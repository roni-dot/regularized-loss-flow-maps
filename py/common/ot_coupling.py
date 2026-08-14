"""
Minibatch exact optimal transport coupling for OT-CFM (Tong et al. 2023,
"Conditional flow matching: Simulation-free dynamic optimal transport";
Pooladian et al. 2023, "Multisample flow matching").

Reorders x1 to pair with x0 via exact OT (linear_sum_assignment on squared
Euclidean cost) instead of the independent pairing coming out of the data
loader. This is the BatchOT coupling from those papers: solved exactly
within blocks of the minibatch (a single global assignment over the full
batch is not feasible at the batch sizes used here), and independently
within each device's shard so the result is agnostic to how many devices
are in play.

linear_sum_assignment is not jittable, so this module runs on host numpy
arrays, outside jax.jit/pmap, in the data path -- before the batch is handed
to dist_utils.replicate_loss_fn_args.
"""

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _squared_euclidean_cost(x0_flat: np.ndarray, x1_flat: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean cost matrix between two flattened batches."""
    return (
        (x0_flat**2).sum(1)[:, None]
        + (x1_flat**2).sum(1)[None, :]
        - 2.0 * x0_flat @ x1_flat.T
    )


def _ot_reorder_chunk(x0_flat: np.ndarray, x1_flat: np.ndarray) -> np.ndarray:
    """Exact OT assignment within one chunk. Returns a permutation of x1's rows
    such that x1_flat[perm] is optimally coupled to x0_flat."""
    cost = _squared_euclidean_cost(x0_flat, x1_flat)
    _, col = linear_sum_assignment(cost)
    return col


def reorder_minibatch_ot(
    x0: np.ndarray,
    x1: np.ndarray,
    ndevices: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Reorder x1 to couple with x0 via exact minibatch optimal transport.

    The coupling is solved independently within each device's shard (rows
    [0, bs/ndevices) belong to device 0, etc. -- the same shard boundaries
    that dist_utils.replicate_batch imposes downstream), and within each
    shard, independently within blocks of `chunk_size` rows. This keeps the
    O(k^3) assignment problem small and ensures the coupling never crosses a
    device boundary.

    Recomputed fresh on every call -- no state is cached across steps.

    Args:
        x0: source batch, shape (bs, ...).
        x1: target batch, shape (bs, ...), same leading dimension as x0.
        ndevices: number of local devices the batch will be sharded across.
        chunk_size: block size for the per-block exact OT solve.

    Returns:
        x1_reordered: x1 with rows permuted within each (device, block).
        perm: shape (bs,) global index array such that
            x1_reordered == x1[perm]. Callers that carry any other
            per-sample array paired with x1 (e.g. class labels) must apply
            this same permutation to keep pairings consistent.
        mean_cost_before: mean squared distance of the original pairing.
        mean_cost_after: mean squared distance of the OT-reordered pairing.
        frac_moved: fraction of rows whose index changed under the reorder.
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

    per_device_bs = bs // ndevices

    x0_flat = x0.reshape(bs, -1)
    x1_flat = x1.reshape(bs, -1)

    x1_out = np.array(x1)
    global_perm = np.arange(bs)
    n_moved = 0
    cost_before_total = 0.0
    cost_after_total = 0.0

    for dev_start in range(0, bs, per_device_bs):
        dev_end = dev_start + per_device_bs
        for blk_start in range(dev_start, dev_end, chunk_size):
            blk_end = min(blk_start + chunk_size, dev_end)

            x0_blk = x0_flat[blk_start:blk_end]
            x1_blk = x1_flat[blk_start:blk_end]

            perm = _ot_reorder_chunk(x0_blk, x1_blk)

            cost_before_total += float(((x0_blk - x1_blk) ** 2).sum(1).sum())
            cost_after_total += float(((x0_blk - x1_blk[perm]) ** 2).sum(1).sum())
            n_moved += int((perm != np.arange(len(perm))).sum())

            x1_out[blk_start:blk_end] = x1[blk_start:blk_end][perm]
            global_perm[blk_start:blk_end] = blk_start + perm

    mean_cost_before = cost_before_total / bs
    mean_cost_after = cost_after_total / bs
    frac_moved = n_moved / bs

    return x1_out, global_perm, mean_cost_before, mean_cost_after, frac_moved
