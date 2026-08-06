"""
Standalone diagnostic: run exactly one pmap'd train_step on however many local
GPUs are visible, then inspect the gradient tree leaf-by-leaf to find which
parameter's gradient first contains a NaN/Inf -- even though the loss scalar
for that same step can come out perfectly finite (a finite loss does not
imply a finite gradient; they are computed together but are separate outputs
of value_and_grad).

Usage (run interactively via srun/salloc, NOT sbatch, so you see output live):
    python -u launchers/debug_grad_nan.py \
        --cfg_path configs.cifar10 \
        --slurm_id 1 \
        --dataset_location /home/ronimaor/regularized-loss-flow-maps/datasets \
        --output_folder /tmp/debug_nan

Adjust the two marked sections below if your checked-out code's train_step /
setup_loss signatures differ slightly from this repo's (e.g. 3-tuple return
instead of 4-tuple, or no monge-gap args) -- the tracebacks you've pasted
show your cluster copy is very close to this one but slightly older.
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import argparse
import importlib

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

import common.datasets as datasets
import common.dist_utils as dist_utils
import common.interpolant as interpolant
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs.cifar10")
    parser.add_argument("--slurm_id", type=int, default=1)
    parser.add_argument("--dataset_location", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="/tmp/debug_nan")
    parser.add_argument(
        "--bs_override",
        type=int,
        default=None,
        help="Shrink cfg.optimization.bs for this diagnostic run to reduce the "
        "memory footprint of JAX_DEBUG_NANS's de-optimized (unfused) rerun. "
        "Must stay a multiple of the device count.",
    )
    return parser.parse_args()


def report_nonfinite(name_prefix, tree):
    """Flatten a (possibly device-leading) pytree and print any leaf containing
    a NaN or Inf, along with how many entries are affected."""
    flat = traverse_util.flatten_dict(tree)
    found_any = False
    for key, val in flat.items():
        val = jnp.asarray(val)
        n_nan = int(jnp.sum(jnp.isnan(val)))
        n_inf = int(jnp.sum(jnp.isinf(val)))
        if n_nan > 0 or n_inf > 0:
            found_any = True
            print(
                f"[{name_prefix}] NON-FINITE leaf: {'/'.join(key)}  "
                f"shape={val.shape}  nan_count={n_nan}  inf_count={n_inf}  "
                f"finite_min={jnp.min(jnp.where(jnp.isfinite(val), val, jnp.inf))}  "
                f"finite_max={jnp.max(jnp.where(jnp.isfinite(val), val, -jnp.inf))}"
            )
    if not found_any:
        print(f"[{name_prefix}] all leaves finite.")
    return found_any


def main():
    args = parse_args()
    print(f"JAX devices ({len(jax.devices())}): {jax.devices()}")

    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    cfg.training.ndevices = jax.device_count()

    if args.bs_override is not None:
        assert args.bs_override % cfg.training.ndevices == 0, (
            f"--bs_override ({args.bs_override}) must be a multiple of "
            f"ndevices ({cfg.training.ndevices})"
        )
        print(f"Overriding cfg.optimization.bs: {cfg.optimization.bs} -> {args.bs_override}")
        cfg.optimization.bs = args.bs_override

    prng_key = jax.random.PRNGKey(cfg.training.seed)

    cfg, ds, prng_key = datasets.setup_target(cfg, prng_key)
    ex_input = next(ds)
    ex_input = ex_input["image"][0] if isinstance(ex_input, dict) else ex_input[0]
    interp = interpolant.setup_interpolant(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)

    train_state, net, schedule, prng_key = state_utils.setup_training_state(
        cfg, ex_input, prng_key
    )
    loss = losses.setup_loss(cfg, net, interp)

    statics = state_utils.StaticArgs(
        net=net,
        schedule=schedule,
        loss=loss,
        get_loss_fn_args=loss_args.get_loss_fn_args,
        train_step=updates.setup_train_step(cfg),
        update_ema_params=updates.setup_ema_update(cfg),
        ds=ds,
        interp=interp,
        sample_rho0=datasets.setup_base(cfg, ex_input),
        inception_fn=None,
    )

    train_state = dist_utils.safe_replicate(cfg, train_state)

    # --- check initial (post sphere-projection) params are clean ---
    init_params = dist_utils.safe_unreplicate(cfg, train_state.params)
    print("\n=== Checking initial params (before any step) ===")
    report_nonfinite("init_params", init_params)

    loss_fn_args, prng_key = statics.get_loss_fn_args(cfg, statics, train_state, prng_key)

    # --- check sharding of every argument about to go into train_step ---
    print("\n=== Sharding check before train_step ===")

    def report_sharding(name_prefix, tree):
        flat = traverse_util.flatten_dict(tree) if isinstance(tree, dict) else {(name_prefix,): tree}
        for key, val in flat.items():
            if val is None:
                continue
            val = jnp.asarray(val)
            label = "/".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
            print(f"[{name_prefix}] {label}: shape={val.shape}  sharding={val.sharding}")

    report_sharding("train_state.params", train_state.params)
    for i, arg in enumerate(loss_fn_args):
        if isinstance(arg, dict):
            report_sharding(f"loss_fn_args[{i}]", arg)
        elif arg is not None:
            # report the REAL type/sharding without converting via jnp.asarray
            # first -- that conversion itself commits a plain numpy array to
            # a single device, which would misleadingly show up as
            # SingleDeviceSharding even when the actual `arg` being passed
            # into train_step is a correctly-unplaced plain numpy array.
            if isinstance(arg, np.ndarray):
                print(f"[loss_fn_args[{i}]] type=plain numpy.ndarray (unplaced, good)  shape={arg.shape}  dtype={arg.dtype}")
            elif hasattr(arg, "sharding"):
                print(f"[loss_fn_args[{i}]] type={type(arg).__name__}  shape={arg.shape}  dtype={arg.dtype}  sharding={arg.sharding}")
            else:
                print(f"[loss_fn_args[{i}]] type={type(arg).__name__}  value/shape={getattr(arg, 'shape', arg)}")
        else:
            print(f"[loss_fn_args[{i}]] None")

    # NOTE: adjust this unpacking if your train_step returns a 3-tuple
    # (state, loss_value, grads) instead of this repo's 4-tuple with aux.
    result = statics.train_step(train_state, statics.loss, loss_fn_args)
    if len(result) == 4:
        new_state, loss_value, grads, aux = result
    else:
        new_state, loss_value, grads = result
        aux = None

    print("\n=== Step 0 result ===")
    print("loss_value:", loss_value)
    if aux is not None:
        print("aux:", aux)

    grads_host = dist_utils.safe_unreplicate(cfg, grads) if cfg.training.ndevices > 1 else grads
    new_params_host = (
        dist_utils.safe_unreplicate(cfg, new_state.params)
        if cfg.training.ndevices > 1
        else new_state.params
    )

    print("\n=== Checking grads from step 0 ===")
    grads_bad = report_nonfinite("grads", grads_host)

    print("\n=== Checking params AFTER step 0's update ===")
    params_bad = report_nonfinite("post_update_params", new_params_host)

    print("\n=== Summary ===")
    print(f"loss finite: {bool(jnp.isfinite(jnp.asarray(loss_value)).all())}")
    print(f"grads contain nan/inf: {grads_bad}")
    print(f"post-update params contain nan/inf: {params_bad}")

    # --- isolate: does EACH device's own data slice reproduce the NaN on its
    # own, with a PLAIN single-device jax.value_and_grad (no pmap, no
    # shard_map, no pmean)? If exactly one device's local shard is bad, pmean
    # would spread that single device's NaN to all devices identically --
    # matching everything we've observed. ---
    if cfg.training.ndevices > 1:
        any_device_bad = False
        for dev_idx in range(cfg.training.ndevices):
            print(
                f"\n=== Re-running device {dev_idx}'s exact data slice with "
                "plain single-device grad ==="
            )

            target_device = jax.devices()[0]  # compute on device 0 regardless

            def take_device_idx(x, dev_idx=dev_idx):
                # index out this device's slice, then force it onto a single
                # physical device explicitly -- pmap-produced arrays keep
                # multi-device sharding metadata even after indexing, which
                # otherwise conflicts with plain single-device arrays (like
                # init_params) in one jit.
                return jax.device_put(jnp.asarray(x)[dev_idx], target_device)

            single_device_args = jax.tree_util.tree_map(take_device_idx, loss_fn_args)
            init_params_1dev = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, target_device), init_params
            )

            # NOTE: adjust has_aux if your cluster's statics.loss returns a
            # plain scalar instead of this repo's (total, aux_dict) tuple.
            try:
                (loss_val_1dev, _aux_1dev), grads_1dev = jax.value_and_grad(
                    statics.loss, has_aux=True
                )(init_params_1dev, *single_device_args)
            except TypeError:
                loss_val_1dev, grads_1dev = jax.value_and_grad(statics.loss)(
                    init_params_1dev, *single_device_args
                )
            print(f"device {dev_idx} loss_value:", loss_val_1dev)
            bad_1dev = report_nonfinite(f"device_{dev_idx}_grads", grads_1dev)
            any_device_bad = any_device_bad or bad_1dev

        print(f"\n==> any single device's own data reproduces NaN in isolation: {any_device_bad}")
        if any_device_bad:
            print(
                "    At least one device's local data shard is the true root "
                "cause -- pmean then spreads its NaN to every device."
            )
        else:
            print(
                "    Every device's data is individually fine -- the NaN is "
                "purely a property of pmap/pmean/collectives themselves, not "
                "any specific device's data."
            )


if __name__ == "__main__":
    main()
