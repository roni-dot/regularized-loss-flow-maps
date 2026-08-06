"""
Nicholas M. Boffi
10/5/25

Simple utilities for single-node multi-GPU data parallelism.
"""

import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import unreplicate
from ml_collections import config_dict


def safe_index(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Extract first element if using multiple devices, otherwise return as-is."""
    if cfg.training.ndevices > 1:
        return x[0]
    else:
        return x


@functools.partial(jax.pmap, axis_name="data")
def _replicate_via_pmap(x: Any) -> Any:
    """Identity pmap used purely to create a properly device-scattered copy
    of x, tagged with axis_name="data" -- the same mesh/axis identity that
    setup_train_step's own pmap uses. flax.jax_utils.replicate (built on
    jax.device_put_replicated) instead produces a generically-named
    "_device_put_sharded" mesh that does not match train_step's mesh, which
    requires an implicit reshard when passed into train_step -- the same
    class of bug already found and fixed for replicate_batch's inputs (see
    that function below for the fuller explanation)."""
    return x


def safe_replicate(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Replicate data across devices if using multiple GPUs."""
    if cfg.training.ndevices > 1:
        # Broadcast a plain host-numpy copy to a leading device axis, then
        # let an actual pmap call (not flax.jax_utils.replicate) scatter it
        # -- mirrors the same "always hand pmap a plain host array" fix
        # applied to replicate_batch below.
        broadcast = jax.tree_util.tree_map(
            lambda a: np.broadcast_to(
                np.asarray(a), (cfg.training.ndevices,) + np.asarray(a).shape
            ),
            x,
        )
        return _replicate_via_pmap(broadcast)
    else:
        return x


def safe_unreplicate(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Unreplicate data from devices if using multiple GPUs."""
    if cfg.training.ndevices > 1:
        return unreplicate(x)
    else:
        return x


def replicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Shard batch across local devices for data parallelism."""
    if cfg.training.ndevices > 1 and x is not None:
        # Force a plain host-numpy round-trip before reshaping. If x is a JAX
        # array already committed to a single device (e.g. anything produced
        # by a jax.jit-decorated function, like get_loss_fn_args_randomness),
        # that device placement survives .reshape() and pmap silently
        # misreads the single-device buffer as if it were already split
        # per-device -- producing wrong-but-finite collective results
        # downstream (confirmed via a minimal jnp.arange-vs-np.arange repro).
        x = np.asarray(x).reshape((cfg.training.ndevices, -1, *x.shape[1:]))
    return x


def unreplicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Merge batch from local devices."""
    if cfg.training.ndevices > 1 and x is not None:
        x = x.reshape((-1, *x.shape[2:]))
    return x


def replicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
    """Replicate all loss function arguments for data parallelism."""
    return tuple(replicate_batch(cfg, arg) for arg in loss_fn_args)


def unreplicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
    """Unreplicate all loss function arguments."""
    return tuple(unreplicate_batch(cfg, arg) for arg in loss_fn_args)
