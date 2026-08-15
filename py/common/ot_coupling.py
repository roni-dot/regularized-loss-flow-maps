"""
Minibatch optimal transport coupling for OT-CFM / SB-CFM (Tong et al. 2023,
"Conditional flow matching: Simulation-free dynamic optimal transport";
Pooladian et al. 2023, "Multisample flow matching").

Reorders x1 to pair with x0 via minibatch OT instead of the independent
pairing coming out of the data loader. Two methods are supported:

  "exact"    -- exact OT (scipy.optimize.linear_sum_assignment on squared
                Euclidean cost). This is the BatchOT coupling from those
                papers. Not jittable, so it runs on host numpy arrays; the
                per-block solves are independent and are optionally
                parallelized across CPU worker processes.
  "sinkhorn" -- entropic OT (ott-jax's Sinkhorn solver), matching the
                Schrodinger-bridge-flavored SB-CFM coupling and reusing the
                same entropic-regularization machinery already used for the
                Monge gap regularizer (common/monge_gap_reg.py) elsewhere in
                this codebase. Runs as ordinary jitted jax ops (on
                whichever device jax is already placing arrays on), one
                block at a time. Unlike "exact", the resulting per-row
                assignment is not generally a permutation: the entropic
                coupling routinely sends more than one x0 row to the same
                x1 row, so the same x1 sample can appear more than once in
                the reordered batch (and label_batch, if reordered
                alongside it, then also happens to duplicate a label).

Both methods solve their coupling within blocks of the minibatch: a single
global assignment/coupling over the full batch is not feasible at the batch
sizes used here (e.g. the checker config's bs=100_000). Both are also
solved independently within each device's shard, so the result is agnostic
to how many devices are in play, and never disturbs the row ranges that
dist_utils.replicate_batch's reshape later treats as per-device shards.

Recomputed fresh on every call -- no state is cached across steps (aside
from the lazily-created worker pool used by the "exact" method's
parallelization, which is a process pool, not training state).
"""

import atexit
import functools
import multiprocessing
import multiprocessing.pool
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn as ott_sinkhorn
from scipy.optimize import linear_sum_assignment


def _block_ranges(bs: int, ndevices: int, chunk_size: int):
    """Row ranges [start, end) for each (device shard, chunk) block, in the
    same row order that dist_utils.replicate_batch's reshape will later
    split into per-device shards."""
    per_device_bs = bs // ndevices
    ranges = []
    for dev_start in range(0, bs, per_device_bs):
        dev_end = dev_start + per_device_bs
        for blk_start in range(dev_start, dev_end, chunk_size):
            blk_end = min(blk_start + chunk_size, dev_end)
            ranges.append((blk_start, blk_end))
    return ranges


# ----------------------------------------------------------------------------
# "exact" method: scipy Hungarian algorithm, optionally parallelized across
# CPU worker processes (independent blocks -> embarrassingly parallel).
# ----------------------------------------------------------------------------


def _squared_euclidean_cost(x0_flat: np.ndarray, x1_flat: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean cost matrix between two flattened batches."""
    return (
        (x0_flat**2).sum(1)[:, None]
        + (x1_flat**2).sum(1)[None, :]
        - 2.0 * x0_flat @ x1_flat.T
    )


def _process_exact_block(
    x0_blk: np.ndarray, x1_blk: np.ndarray
) -> Tuple[np.ndarray, float, float, int]:
    """Exact OT assignment within one chunk. Module-level (not a closure) so
    it can be pickled and sent to worker processes.

    Returns (perm, cost_before, cost_after, n_moved) where perm is a
    permutation of x1_blk's rows such that x1_blk[perm] is optimally
    coupled to x0_blk, and cost_before/cost_after/n_moved are this block's
    (unnormalized) contributions to the batch-level diagnostics.
    """
    cost = _squared_euclidean_cost(x0_blk, x1_blk)
    _, perm = linear_sum_assignment(cost)
    cost_before = float(((x0_blk - x1_blk) ** 2).sum(1).sum())
    cost_after = float(((x0_blk - x1_blk[perm]) ** 2).sum(1).sum())
    n_moved = int((perm != np.arange(len(perm))).sum())
    return perm, cost_before, cost_after, n_moved


_WORKER_POOLS = {}


def _get_pool(n_jobs: int) -> multiprocessing.pool.Pool:
    """Lazily create (and cache) a worker pool of the requested size.
    Spawning a fresh pool on every training step would swamp any speedup
    from parallelizing the block loop, so the pool is created once per
    distinct n_jobs value and reused for the lifetime of the process."""
    pool = _WORKER_POOLS.get(n_jobs)
    if pool is None:
        pool = multiprocessing.Pool(processes=n_jobs)
        _WORKER_POOLS[n_jobs] = pool
    return pool


@atexit.register
def _close_worker_pools() -> None:
    for pool in _WORKER_POOLS.values():
        pool.close()
        pool.join()


def _reorder_minibatch_exact(
    x0_flat: np.ndarray,
    x1_flat: np.ndarray,
    ranges,
    n_jobs: int,
) -> Tuple[np.ndarray, list]:
    """Solve exact OT independently on each block. Returns the full list of
    per-block (perm, cost_before, cost_after, n_moved) results, in the same
    order as `ranges`."""
    blocks = [(x0_flat[a:b], x1_flat[a:b]) for a, b in ranges]

    if n_jobs is None or n_jobs <= 1 or len(blocks) <= 1:
        return [_process_exact_block(x0_blk, x1_blk) for x0_blk, x1_blk in blocks]

    pool = _get_pool(n_jobs)
    return pool.starmap(_process_exact_block, blocks)


# ----------------------------------------------------------------------------
# "sinkhorn" method: entropic OT via ott-jax, row-sampled into a per-x0
# partner index. Runs as ordinary jitted jax ops, one block at a time.
# ----------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("relative_epsilon", "max_iterations"))
def _sinkhorn_reorder_chunk(
    x0_blk: jnp.ndarray,
    x1_blk: jnp.ndarray,
    key: jnp.ndarray,
    epsilon,
    relative_epsilon,
    max_iterations: int,
) -> jnp.ndarray:
    """Entropic OT within one chunk (uniform marginals), then sample a
    partner for each x0 row from its row of the Sinkhorn coupling matrix
    (row-normalized, since under uniform marginals each row of the raw
    coupling already sums to 1 / chunk_size). Returns a (chunk_size,) index
    array into x1_blk -- generally NOT a permutation, unlike the "exact"
    method: the entropic coupling can (and does) send more than one x0 row
    to the same x1 row.
    """
    geom = pointcloud.PointCloud(
        x0_blk, x1_blk, epsilon=epsilon, relative_epsilon=relative_epsilon
    )
    prob = linear_problem.LinearProblem(geom)
    solver = ott_sinkhorn.Sinkhorn(max_iterations=max_iterations)
    out = solver(prob)
    pi = out.matrix

    row_sums = pi.sum(axis=1, keepdims=True)
    row_probs = pi / jnp.clip(row_sums, 1e-30, None)
    log_probs = jnp.log(jnp.clip(row_probs, 1e-30, None))

    keys = jax.random.split(key, x0_blk.shape[0])
    return jax.vmap(jax.random.categorical)(keys, log_probs)


def _reorder_minibatch_sinkhorn(
    x0_flat: np.ndarray,
    x1_flat: np.ndarray,
    ranges,
    prng_key: jnp.ndarray,
    epsilon,
    relative_epsilon,
    max_iterations: int,
) -> list:
    """Solve entropic OT independently on each block. Returns the full list
    of per-block (idx, cost_before, cost_after, n_moved) results, where idx
    is the (possibly repeating) sampled partner index array, in the same
    order as `ranges`."""
    results = []
    keys = jax.random.split(prng_key, len(ranges))
    for (a, b), block_key in zip(ranges, keys):
        x0_blk = x0_flat[a:b]
        x1_blk = x1_flat[a:b]
        idx = np.asarray(
            _sinkhorn_reorder_chunk(
                x0_blk, x1_blk, block_key, epsilon, relative_epsilon, max_iterations
            )
        )
        cost_before = float(((x0_blk - x1_blk) ** 2).sum(1).sum())
        cost_after = float(((x0_blk - x1_blk[idx]) ** 2).sum(1).sum())
        n_moved = int((idx != np.arange(len(idx))).sum())
        results.append((idx, cost_before, cost_after, n_moved))
    return results


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def reorder_minibatch_ot(
    x0: np.ndarray,
    x1: np.ndarray,
    ndevices: int,
    chunk_size: int,
    method: str = "exact",
    n_jobs: int = 1,
    prng_key: Optional[jnp.ndarray] = None,
    sinkhorn_epsilon=0.1,
    sinkhorn_relative_epsilon=None,
    sinkhorn_max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Reorder x1 to couple with x0 via minibatch optimal transport.

    See the module docstring for the "exact" vs "sinkhorn" tradeoff.

    Args:
        x0: source batch, shape (bs, ...).
        x1: target batch, shape (bs, ...), same leading dimension as x0.
        ndevices: number of local devices the batch will be sharded across.
        chunk_size: block size for the per-block OT solve.
        method: "exact" (scipy Hungarian algorithm) or "sinkhorn" (entropic
            OT via ott-jax).
        n_jobs: number of worker processes for parallelizing the per-block
            solves under method="exact" (ignored for "sinkhorn", which runs
            as ordinary jax ops instead). 1 (default) runs sequentially in
            the calling process, with no pool overhead.
        prng_key: required for method="sinkhorn" (used to sample partners
            from each block's coupling matrix). Unused for "exact".
        sinkhorn_epsilon, sinkhorn_relative_epsilon, sinkhorn_max_iter:
            entropic-regularization / solver settings for method="sinkhorn".

    Returns:
        x1_reordered: x1 with rows reassigned within each (device, block).
        idx: shape (bs,) global index array such that x1_reordered ==
            x1[idx]. Callers that carry any other per-sample array paired
            with x1 (e.g. class labels) must apply this same index array to
            keep pairings consistent. For method="exact" this is a genuine
            permutation (within each block); for method="sinkhorn" it can
            repeat indices, since the entropic coupling is not generally a
            bijection.
        mean_cost_before: mean squared distance of the original pairing.
        mean_cost_after: mean squared distance of the OT-reordered pairing.
        frac_moved: fraction of rows whose index changed under the reorder.
        frac_unique: fraction of distinct x1 rows actually used in the
            reordered batch. 1.0 for method="exact" (each block's coupling
            is a bijection); can be < 1.0 for method="sinkhorn", where the
            entropic coupling can send more than one x0 row to the same
            x1 row.
    """
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    bs = x0.shape[0]
    assert x1.shape[0] == bs, (
        f"x0 and x1 batch sizes must match, got {x0.shape[0]} and {x1.shape[0]}"
    )
    assert bs % ndevices == 0, (
        f"batch size ({bs}) must be divisible by ndevices ({ndevices}) so that "
        "OT coupling blocks never cross a device shard boundary"
    )

    x0_flat = x0.reshape(bs, -1)
    x1_flat = x1.reshape(bs, -1)
    ranges = _block_ranges(bs, ndevices, chunk_size)

    if method == "exact":
        results = _reorder_minibatch_exact(x0_flat, x1_flat, ranges, n_jobs)
    elif method == "sinkhorn":
        assert prng_key is not None, "prng_key is required for method='sinkhorn'"
        results = _reorder_minibatch_sinkhorn(
            x0_flat,
            x1_flat,
            ranges,
            prng_key,
            sinkhorn_epsilon,
            sinkhorn_relative_epsilon,
            sinkhorn_max_iter,
        )
    else:
        raise ValueError(f"Unknown OT coupling method: {method!r}")

    x1_out = np.array(x1)
    global_idx = np.arange(bs)
    cost_before_total = 0.0
    cost_after_total = 0.0
    n_moved = 0

    for (blk_start, blk_end), (idx, cost_before, cost_after, blk_n_moved) in zip(
        ranges, results
    ):
        x1_out[blk_start:blk_end] = x1[blk_start:blk_end][idx]
        global_idx[blk_start:blk_end] = blk_start + idx
        cost_before_total += cost_before
        cost_after_total += cost_after
        n_moved += blk_n_moved

    mean_cost_before = cost_before_total / bs
    mean_cost_after = cost_after_total / bs
    frac_moved = n_moved / bs
    # Blocks never share index ranges (global_idx[a:b] is always shifted
    # into [a, b)), so a duplicate can only occur within a single block --
    # counting uniques over the whole batch is exactly equivalent to
    # summing per-block unique counts. Always 1.0 for method="exact"
    # (each block's coupling is a bijection); can be < 1.0 for
    # method="sinkhorn", where the entropic coupling routinely reuses the
    # same x1 row for more than one x0 row.
    frac_unique = len(np.unique(global_idx)) / bs

    return x1_out, global_idx, mean_cost_before, mean_cost_after, frac_moved, frac_unique
