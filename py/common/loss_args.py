"""
Nicholas M. Boffi
10/5/25

Code for setting up arguments for loss functions.
"""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from . import state_utils
from . import dist_utils
from . import ot_coupling as ot_coupling_mod


def safe_resize(curr_bs: int, bs: int, x: jnp.ndarray) -> jnp.ndarray:
    """Resize the input array to the current batch size."""
    if curr_bs < bs:
        x = x[:curr_bs]
    return x


def _sample_diagonal(
    key: jnp.ndarray, bs: int, tmin: float, tmax: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample points on the diagonal (s=t)."""
    s = jax.random.uniform(key, shape=(bs,), minval=tmin, maxval=tmax)
    return s, s


def _sample_triangle(
    key1: jnp.ndarray,
    key2: jnp.ndarray,
    bs: int,
    tmin: float,
    tmax: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample uniformly from upper triangle."""
    temp1 = jax.random.uniform(key1, shape=(bs,), minval=tmin, maxval=tmax)
    temp2 = jax.random.uniform(key2, shape=(bs,), minval=tmin, maxval=tmax)
    s = jnp.minimum(temp1, temp2)
    t = jnp.maximum(temp1, temp2)
    return s, t


def _get_diag_offdiag_bs(cfg: config_dict.ConfigDict, bs: int) -> Tuple[int, int]:
    """Get diagonal and off-diagonal batch sizes."""
    if hasattr(cfg.optimization, "diag_fraction"):
        diag_bs = max(1, int(bs * cfg.optimization.diag_fraction))
    elif hasattr(cfg.optimization, "diag_bs"):
        diag_bs = cfg.optimization.diag_bs
    else:
        raise ValueError("Either diag_fraction or diag_bs must be specified")

    offdiag_bs = bs - diag_bs

    return diag_bs, offdiag_bs


def _concat_diag_offdiag(
    s_diag: jnp.ndarray,
    t_diag: jnp.ndarray,
    s_offdiag: jnp.ndarray,
    t_offdiag: jnp.ndarray,
    ndevices: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Concatenate diagonal and off-diagonal samples.

    With ndevices > 1, dist_utils.replicate_batch shards by reshaping to
    (ndevices, -1, ...), so each device gets a *contiguous* block. A plain
    [all diag; all offdiag] concatenation therefore hands whole devices a
    single kind of sample, while losses.py re-derives the split from the
    per-device batch size and slices positionally -- so most off-diagonal
    slots end up holding s == t samples, where lsd_term is identically zero
    (X_{s,s} = x and dt_X|_{t=s} = calc_b(s, x) by construction). At bs=512
    on 4 devices only 32/128 off-diagonal slots carried real samples.

    Interleaving in per-device blocks makes each shard hold the intended
    mixture.
    """
    if ndevices > 1:
        s_diag = s_diag.reshape(ndevices, -1)
        t_diag = t_diag.reshape(ndevices, -1)
        s_offdiag = s_offdiag.reshape(ndevices, -1)
        t_offdiag = t_offdiag.reshape(ndevices, -1)
        sbatch = jnp.concatenate([s_diag, s_offdiag], axis=1).reshape(-1)
        tbatch = jnp.concatenate([t_diag, t_offdiag], axis=1).reshape(-1)
    else:
        sbatch = jnp.concatenate([s_diag, s_offdiag])
        tbatch = jnp.concatenate([t_diag, t_offdiag])
    return sbatch, tbatch


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_loss_fn_args_randomness(
    prng_key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
    diag_bs: int,
    offdiag_bs: int,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    (
        tkey,
        skey,
        ukey,
        x0key,
        tkey2,
        mg_key,
        mg_x0key,
    ) = jax.random.split(prng_key, num=7)
    x0batch = sample_rho0(cfg.optimization.bs, x0key)

    bs = cfg.optimization.bs
    tmin = cfg.training.tmin
    tmax = cfg.training.tmax

    # If offdiag_bs is 0, use full batch on diagonal
    if offdiag_bs == 0:
        sbatch, tbatch = _sample_diagonal(skey, bs, tmin, tmax)
    else:
        # sample diagonal and off-diagonal points
        s_diag, t_diag = (
            _sample_diagonal(skey, diag_bs, tmin, tmax)
            if diag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )
        s_offdiag, t_offdiag = (
            _sample_triangle(tkey, tkey2, offdiag_bs, tmin, tmax)
            if offdiag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )

        sbatch, tbatch = _concat_diag_offdiag(s_diag, t_diag, s_offdiag, t_offdiag, cfg.training.ndevices)

    if cfg.training.psd_type == "midpoint":
        ubatch = 0.5 * (sbatch + tbatch)
        hbatch = None  # Not used for midpoint interpolation
    elif cfg.training.psd_type == "uniform":
        minval = 0.0
        maxval = 1.0

        hbatch = jax.random.uniform(
            ukey, shape=(cfg.optimization.bs,), minval=minval, maxval=maxval
        )

        ubatch = hbatch * sbatch + (1 - hbatch) * tbatch
    elif cfg.training.psd_type == None:
        ubatch = None
        hbatch = None
    else:
        raise ValueError(f"Unknown psd_type: {cfg.training.psd_type}")

    dropout_keys = jax.random.split(tkey, num=cfg.optimization.bs).reshape(
        (cfg.optimization.bs, -1)
    )
    prng_key = jax.random.split(dropout_keys[0])[0]

    # Sample K independent (s, t) pairs for the Monge gap, plus its own base
    # point cloud shared across all K pairs. lambda_reg is static (cfg is a
    # static jit arg), so when the regularizer is disabled this whole block
    # is never traced -- mg_x0's base-distribution sample is the expensive
    # part, but mg_s_vec/mg_t_vec are gated too since they're only ever
    # meaningful alongside mg_x0/mg_x1. Length-1 dummies stand in instead.
    if cfg.training.lambda_reg > 0.0:
        mg_x0 = sample_rho0(cfg.training.mg_batch_size, mg_x0key)
        mg_s_vec, mg_t_vec = _sample_mg_pairs(
            mg_key, cfg.training.monge_num_pairs, tmin, tmax
        )
    else:
        mg_x0 = jnp.zeros((1,) + x0batch.shape[1:], dtype=x0batch.dtype)
        mg_s_vec = jnp.zeros((1,))
        mg_t_vec = jnp.zeros((1,))

    return (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
        mg_x0,
        mg_s_vec,
        mg_t_vec,
    )


def get_batch(
    cfg: config_dict.ConfigDict, statics: state_utils.StaticArgs, prng_key: jnp.ndarray
) -> int:
    """Extract a batch based on the structure expected for image
    or non-image datasets."""
    is_image_dataset = (cfg.problem.target in ["cifar10", "celeb_a"]) or (
        "afhq" in cfg.problem.target
    )

    batch = next(statics.ds)
    if is_image_dataset:
        x1batch = batch["image"]
        label_batch = batch["label"]
    else:
        x1batch = batch
        label_batch = None

    # add droput to randomly replace fraction cfg.class_dropout of labels by num_classes
    # if not conditional, we don't need the labels
    if not cfg.training.conditional:
        label_batch = None

    elif cfg.training.class_dropout > 0:
        assert cfg.network.use_cfg  # class dropout doesn't make sense without cfg
        mask = jax.random.bernoulli(
            prng_key, cfg.training.class_dropout, shape=(cfg.optimization.bs,)
        )
        mask = mask > 0
        label_batch = label_batch.at[mask].set(cfg.problem.num_classes)
        prng_key = jax.random.split(prng_key)[0]

    return x1batch, label_batch, prng_key


def get_loss_fn_args(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> Tuple:

    # Determine batch sizes based on splitting configuration
    bs = cfg.optimization.bs

    # Normal batch splitting
    diag_bs, offdiag_bs = _get_diag_offdiag_bs(cfg, bs)

    nd = cfg.training.ndevices
    if nd > 1:
        assert bs % nd == 0, f"bs ({bs}) not divisible by ndevices ({nd})"
        assert diag_bs % nd == 0 and offdiag_bs % nd == 0, (
            f"diag_bs ({diag_bs}) and offdiag_bs ({offdiag_bs}) must each be "
            f"divisible by ndevices ({nd}) for per-device interleaving"
        )
        local_diag_bs, _ = _get_diag_offdiag_bs(cfg, bs // nd)
        assert local_diag_bs == diag_bs // nd, (
            f"per-device split mismatch: losses.py uses {local_diag_bs} diagonal "
            f"samples per device, interleaving supplies {diag_bs // nd}"
        )

    # drew randomness needed for the objective
    (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
        mg_x0,
        mg_s_vec,
        mg_t_vec,
    ) = get_loss_fn_args_randomness(
        prng_key,
        cfg,
        statics.sample_rho0,
        diag_bs,
        offdiag_bs,
    )

    # grab next batch of samples and labels
    x1batch, label_batch, prng_key = get_batch(cfg, statics, prng_key)

    # Optional minibatch OT coupling (OT-CFM / SB-CFM): reorder x1 to pair
    # with x0 via per-shard, per-chunk minibatch OT instead of the
    # independent pairing from the data loader. Must run here, outside
    # jit/pmap and before replication -- "exact" uses scipy's
    # linear_sum_assignment, which is not jittable. Recomputed fresh every
    # step; default is None, which leaves the existing independent-pairing
    # behavior unchanged.
    # Diagnostics are returned rather than logged here: a bare wandb.log
    # call would advance the global step counter on its own, so metrics
    # logged later in train_loop (via logging.log_metrics) would land on a
    # different wandb step than this one.
    ot_metrics = {}
    if cfg.training.ot_coupling is not None:
        if cfg.training.ot_coupling not in ("exact", "sinkhorn"):
            raise ValueError(f"Unknown training.ot_coupling: {cfg.training.ot_coupling!r}")
        ot_key, prng_key = jax.random.split(prng_key)
        (
            x1batch,
            ot_idx,
            ot_cost_before,
            ot_cost_after,
            ot_frac_moved,
            ot_frac_unique,
            ot_converged,
            ot_n_iters,
            ot_frac_repaired,
        ) = (
            ot_coupling_mod.reorder_minibatch_ot(
                x0batch,
                x1batch,
                cfg.training.ndevices,
                cfg.training.ot_chunk_size,
                method=cfg.training.ot_coupling,
                n_jobs=getattr(cfg.training, "ot_n_jobs", 1),
                prng_key=ot_key,
                # reuse the Monge gap regularizer's entropic-OT settings so
                # the two features share one set of Sinkhorn hyperparameters
                sinkhorn_epsilon=cfg.training.sinkhorn_eps,
                sinkhorn_relative_epsilon=cfg.training.sinkhorn_relative_epsilon,
                sinkhorn_max_iter=cfg.training.sinkhorn_max_iter,
            )
        )
        # keep any per-sample array paired with x1 (e.g. class labels)
        # consistent with the reordered x1
        if label_batch is not None:
            label_batch = np.asarray(label_batch)[ot_idx]
        ot_metrics = {
            "ot_coupling/mean_cost_before": ot_cost_before,
            "ot_coupling/mean_cost_after": ot_cost_after,
            "ot_coupling/frac_moved": ot_frac_moved,
            "ot_coupling/frac_unique": ot_frac_unique,
            "ot_coupling/converged": ot_converged,
            "ot_coupling/n_iters": ot_n_iters,
            "ot_coupling/frac_repaired": ot_frac_repaired,
        }

    # set up the teacher (uses current params for self-distillation)
    teacher_params = train_state.params

    # Monge gap target slice: take mg_batch_size samples from the dataset batch.
    # NOTE: this is capped by the training batch size, so mg_batch_size > bs
    # silently yields fewer points than configured. Assert rather than truncate,
    # since a smaller batch changes the entropic gap estimate.
    # Regularizer disabled: skip the slice entirely and hand losses.py a
    # length-1 dummy instead. It is never read -- losses.py's own
    # `if cfg.training.lambda_reg > 0.0` guards the only place mg_x1 is
    # used -- this just avoids slicing/broadcasting/host-transferring an
    # mg_batch_size-sized array on every step for no reason.
    if cfg.training.lambda_reg > 0.0:
        assert cfg.training.mg_batch_size <= bs, (
            f"mg_batch_size ({cfg.training.mg_batch_size}) exceeds optimization.bs "
            f"({bs}); mg_x1 is sliced from the training batch and cannot be larger."
        )
        mg_x1 = x1batch[: min(bs, cfg.training.mg_batch_size)]
    else:
        mg_x1 = np.zeros((1,) + x1batch.shape[1:], dtype=x1batch.dtype)

    # for training flow map.
    # NOTE: mg_x0/mg_x1 are deliberately NOT in this tuple. replicate_loss_fn_args
    # shards along the sample axis, which would give each device only
    # mg_batch_size/ndevices points. The Monge gap is a distribution-level
    # quantity and its entropic estimator is strongly batch-size dependent, so
    # averaging ndevices independent small-batch gaps != one full-batch gap.
    # They are broadcast below instead, so every device computes the identical
    # gap on the full mg_batch_size and the result stays comparable to the
    # single-GPU checker runs.
    loss_fn_args = (
        x0batch,
        x1batch,
        label_batch,
        sbatch,
        tbatch,
        ubatch,
        hbatch,
        dropout_keys,
    )
    loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)

    # Group-level (not per-sample) Monge arguments: broadcast a leading device
    # axis so pmap is satisfied without splitting the data.
    # mg_s_vec/mg_t_vec are (K,) and would fail outright whenever K < ndevices.
    ndevices = cfg.training.ndevices
    if ndevices > 1:
        # np.asarray forces a host round-trip first: mg_x0/mg_x1/mg_s_vec/
        # mg_t_vec come out of the jax.jit-decorated randomness function
        # above already committed to a single device, and broadcasting a
        # single-device-committed JAX array straight into a pmap argument
        # (rather than a plain host array) causes pmap to misread it as
        # already device-split -- producing wrong-but-finite collective
        # results downstream (see dist_utils.replicate_batch for the same
        # fix and a fuller explanation).
        mg_x0 = np.broadcast_to(np.asarray(mg_x0), (ndevices,) + mg_x0.shape)
        mg_x1 = np.broadcast_to(np.asarray(mg_x1), (ndevices,) + mg_x1.shape)
        mg_s_vec = np.broadcast_to(np.asarray(mg_s_vec), (ndevices,) + mg_s_vec.shape)
        mg_t_vec = np.broadcast_to(np.asarray(mg_t_vec), (ndevices,) + mg_t_vec.shape)

    # order is unchanged from before: mg_x0/mg_x1 were already the last two
    # entries of the replicated tuple, so losses.py needs no modification.
    loss_fn_args = (teacher_params, *loss_fn_args, mg_x0, mg_x1, mg_s_vec, mg_t_vec)

    return loss_fn_args, prng_key, ot_metrics

def _sample_mg_pairs(
    key: jnp.ndarray, K: int, tmin: float, tmax: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single fixed (s, t) = (tmin, tmax) pair for the Monge gap regularizer.

    The Monge gap only needs to approximate the path between noise (tmin) and
    data (tmax); randomly resampling (s, t) each step adds noise on top of the
    entropic gap estimator's own noise without adding signal. Fixed to the
    endpoints instead. `key` and `K` are unused, kept for call-site compat.
    """
    del key, K
    s_vec = jnp.array([tmin])
    t_vec = jnp.array([tmax])
    return s_vec, t_vec
