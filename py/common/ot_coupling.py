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
                this codebase. Solved once over the *entire* minibatch (not
                chunked into per-device/per-block pieces like "exact"), as
                ordinary jitted jax ops on whichever device jax is already
                placing arrays on (GPU/TPU if available -- running the
                O(bs^2) solve on CPU was tried and was too slow in
                practice). Each x0 row independently samples a partner
                from its row of the coupling matrix, which on its own
                routinely sends more than
                one x0 row to the same x1 row (biasing training toward
                whichever x1 sample got duplicated). A duplicate-repair
                pass then fixes this: for every x1 row claimed by more than
                one x0 row, only the highest-coupling-mass claim is kept,
                and every "losing" x0 row is reassigned via an exact
                (Hungarian) assignment against the x1 rows nobody claimed.
                The final idx is therefore a genuine permutation, same as
                "exact" -- just with the bulk of the pairing coming from
                the (faster-converging) entropic solve and only the
                collisions patched up exactly.

The "exact" method solves its coupling within blocks of the minibatch: a
single global assignment over the full batch is not feasible at the batch
sizes used here (e.g. the checker config's bs=100_000 for other datasets).
It is solved independently within each device's shard, so the result is
agnostic to how many devices are in play, and never disturbs the row
ranges that dist_utils.replicate_batch's reshape later treats as
per-device shards. The "sinkhorn" method instead solves one coupling over
the whole batch (see above): its cost is dominated by an O(bs^2) pairwise
distance matrix computed as ordinary jax ops, which is cheap to run on
whatever accelerator is already handling the training step.

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
) -> Tuple[np.ndarray, float, float, int, float, float, float]:
    """Exact OT assignment within one chunk. Module-level (not a closure) so
    it can be pickled and sent to worker processes.

    Returns (perm, cost_before, cost_after, n_moved, converged, n_iters,
    frac_repaired) where perm is a permutation of x1_blk's rows such that
    x1_blk[perm] is optimally coupled to x0_blk, and
    cost_before/cost_after/n_moved are this block's (unnormalized)
    contributions to the batch-level diagnostics. The Hungarian algorithm
    is an exact combinatorial solve, not an iterative one and never
    produces duplicates, so converged/n_iters/frac_repaired are trivial
    placeholders (1.0 / 0.0 / 0.0), included only so
    reorder_minibatch_ot's aggregation loop can treat both methods
    uniformly.
    """
    cost = _squared_euclidean_cost(x0_blk, x1_blk)
    _, perm = linear_sum_assignment(cost)
    cost_before = float(((x0_blk - x1_blk) ** 2).sum(1).sum())
    cost_after = float(((x0_blk - x1_blk[perm]) ** 2).sum(1).sum())
    n_moved = int((perm != np.arange(len(perm))).sum())
    return perm, cost_before, cost_after, n_moved, 1.0, 0.0, 0.0


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
# partner index. Solved once over the whole batch, as ordinary jitted jax
# ops on whichever device jax is already placing arrays on (GPU/TPU if
# available) -- running the O(bs^2) solve on CPU was tried and was too
# slow in practice.
# ----------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("relative_epsilon", "max_iterations"))
def _sinkhorn_reorder_chunk(
    x0_blk: jnp.ndarray,
    x1_blk: jnp.ndarray,
    key: jnp.ndarray,
    epsilon,
    relative_epsilon,
    max_iterations: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Entropic OT within one chunk (uniform marginals), then sample a
    partner for each x0 row from its row of the Sinkhorn coupling matrix
    (row-normalized, since under uniform marginals each row of the raw
    coupling already sums to 1 / chunk_size). Returns (idx, score,
    converged, n_iters): idx is a (chunk_size,) index array into x1_blk --
    on its own generally NOT a permutation, since the entropic coupling
    can (and does) send more than one x0 row to the same x1 row (duplicate
    repair happens on the host afterwards, see
    _repair_duplicate_assignments). score is the raw coupling mass
    pi[i, idx[i]] for each row i, used by the repair pass to decide which
    of several x0 rows claiming the same x1 row "wins" that claim.
    converged / n_iters are the underlying ott-jax SinkhornOutput's own
    convergence diagnostics.
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
    idx = jax.vmap(jax.random.categorical)(keys, log_probs)
    score = jnp.take_along_axis(pi, idx[:, None], axis=1)[:, 0]
    return idx, score, out.converged, out.n_iters


def _repair_duplicate_assignments(
    x0_flat: np.ndarray, x1_flat: np.ndarray, idx: np.ndarray, score: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Turn the (possibly repeating) sampled idx into a genuine permutation.

    For every x1 row claimed by more than one x0 row, keeps only the claim
    with the highest coupling mass (`score`) and treats every other x0 row
    in that group as a "loser" that needs a new partner. Losers are then
    matched, via exact Hungarian assignment on squared Euclidean cost,
    against exactly the x1 rows nobody claimed ("orphans") -- these two
    sets are always the same size, since bs = (# distinct claimed rows) +
    (# orphans) = (# winners) + (# losers).

    Returns (repaired_idx, n_repaired) where n_repaired is the number of
    x0 rows that were reassigned (i.e. the number of losers).
    """
    bs = idx.shape[0]
    # Process rows in descending score order so that, for each x1 column,
    # np.unique's "first occurrence" is the highest-scoring claim on it.
    order = np.argsort(-score, kind="stable")
    _, winner_pos = np.unique(idx[order], return_index=True)
    winner_rows = order[winner_pos]

    is_winner = np.zeros(bs, dtype=bool)
    is_winner[winner_rows] = True
    loser_rows = np.nonzero(~is_winner)[0]

    claimed_cols = np.zeros(bs, dtype=bool)
    claimed_cols[idx[winner_rows]] = True
    orphan_cols = np.nonzero(~claimed_cols)[0]

    if len(loser_rows) == 0:
        return idx, 0

    cost = _squared_euclidean_cost(x0_flat[loser_rows], x1_flat[orphan_cols])
    row_ind, col_ind = linear_sum_assignment(cost)
    idx = idx.copy()
    idx[loser_rows[row_ind]] = orphan_cols[col_ind]
    return idx, len(loser_rows)


def _reorder_minibatch_sinkhorn(
    x0_flat: np.ndarray,
    x1_flat: np.ndarray,
    prng_key: jnp.ndarray,
    epsilon,
    relative_epsilon,
    max_iterations: int,
) -> list:
    """Solve entropic OT once over the whole batch, then repair duplicate
    assignments (see _repair_duplicate_assignments) so the result is a
    genuine permutation, like "exact". Returns a single-element list with
    (idx, cost_before, cost_after, n_moved, converged, n_iters,
    frac_repaired) -- kept as a list so callers can treat "exact"
    (multi-block) and "sinkhorn" (single whole-batch block) uniformly."""
    idx, score, converged, n_iters = _sinkhorn_reorder_chunk(
        x0_flat, x1_flat, prng_key, epsilon, relative_epsilon, max_iterations
    )
    idx = np.asarray(idx)
    score = np.asarray(score)
    idx, n_repaired = _repair_duplicate_assignments(x0_flat, x1_flat, idx, score)
    cost_before = float(((x0_flat - x1_flat) ** 2).sum(1).sum())
    cost_after = float(((x0_flat - x1_flat[idx]) ** 2).sum(1).sum())
    n_moved = int((idx != np.arange(len(idx))).sum())
    frac_repaired = n_repaired / len(idx)
    return [
        (idx, cost_before, cost_after, n_moved, float(converged), float(n_iters), frac_repaired)
    ]


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
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float, float]:
    """Reorder x1 to couple with x0 via minibatch optimal transport.

    See the module docstring for the "exact" vs "sinkhorn" tradeoff.

    Args:
        x0: source batch, shape (bs, ...).
        x1: target batch, shape (bs, ...), same leading dimension as x0.
        ndevices: number of local devices the batch will be sharded across.
            Only used by method="exact" (to pick block boundaries that
            never cross a device shard); method="sinkhorn" solves the
            whole batch as one block regardless of ndevices.
        chunk_size: block size for the per-block OT solve under
            method="exact". Unused for "sinkhorn", which always solves the
            entire batch in a single block.
        method: "exact" (scipy Hungarian algorithm) or "sinkhorn" (entropic
            OT via ott-jax, solved over the full batch).
        n_jobs: number of worker processes for parallelizing the per-block
            solves under method="exact" (ignored for "sinkhorn"). 1
            (default) runs sequentially in the calling process, with no
            pool overhead.
        prng_key: required for method="sinkhorn" (used to sample partners
            from the batch's coupling matrix). Unused for "exact".
        sinkhorn_epsilon, sinkhorn_relative_epsilon, sinkhorn_max_iter:
            entropic-regularization / solver settings for method="sinkhorn".

    Returns:
        x1_reordered: x1 with rows reassigned by the OT coupling (within
            each device/block for "exact"; over the whole batch for
            "sinkhorn").
        idx: shape (bs,) global index array such that x1_reordered ==
            x1[idx]. Callers that carry any other per-sample array paired
            with x1 (e.g. class labels) must apply this same index array to
            keep pairings consistent. This is a genuine permutation for
            both methods: "exact" is a bijection within each block by
            construction, and "sinkhorn"'s raw per-row sampling (which can
            send more than one x0 row to the same x1 row) is patched into
            a permutation by a duplicate-repair pass -- see the module
            docstring and _repair_duplicate_assignments.
        mean_cost_before: mean squared distance of the original pairing.
        mean_cost_after: mean squared distance of the OT-reordered pairing.
        frac_moved: fraction of rows whose index changed under the reorder.
        frac_unique: fraction of distinct x1 rows actually used in the
            reordered batch. Always 1.0 for both methods now that
            "sinkhorn"'s duplicates are repaired -- kept as a sanity-check
            diagnostic; frac_repaired below is the metric that actually
            reflects how much repair "sinkhorn" needed.
        converged: 1.0 if every block's solve converged, else 0.0. Trivial
            (always 1.0) for method="exact", which is an exact combinatorial
            solve rather than an iterative one; meaningful for "sinkhorn",
            where it reflects the underlying ott-jax SinkhornOutput.converged.
        n_iters: worst-case (max over blocks) iteration count the solver
            used. Trivial (always 0.0) for method="exact"; meaningful for
            "sinkhorn".
        frac_repaired: fraction of x0 rows whose sampled partner was a
            duplicate claim and had to be reassigned by the exact repair
            pass. Trivial (always 0.0) for method="exact"; for "sinkhorn"
            it's a proxy for how much the entropic sampling deviated from
            a clean bijection before repair (larger sinkhorn_epsilon ->
            more spread-out coupling rows -> more duplicate claims ->
            larger frac_repaired).
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

    if method == "exact":
        ranges = _block_ranges(bs, ndevices, chunk_size)
        results = _reorder_minibatch_exact(x0_flat, x1_flat, ranges, n_jobs)
    elif method == "sinkhorn":
        assert prng_key is not None, "prng_key is required for method='sinkhorn'"
        # Whole batch as a single block: unlike "exact", sinkhorn's cost is
        # an O(bs^2) pairwise distance matrix rather than a combinatorial
        # assignment, and it runs as ordinary jax ops (see
        # _sinkhorn_reorder_chunk), so there's no need to shard it
        # per-device/per-chunk.
        ranges = [(0, bs)]
        results = _reorder_minibatch_sinkhorn(
            x0_flat,
            x1_flat,
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
    converged = 1.0  # AND across blocks: 1.0 only if every block converged
    n_iters = 0.0  # MAX across blocks: worst-case iteration count
    n_repaired = 0

    for (blk_start, blk_end), (
        idx,
        cost_before,
        cost_after,
        blk_n_moved,
        blk_converged,
        blk_n_iters,
        blk_frac_repaired,
    ) in zip(ranges, results):
        x1_out[blk_start:blk_end] = x1[blk_start:blk_end][idx]
        global_idx[blk_start:blk_end] = blk_start + idx
        cost_before_total += cost_before
        cost_after_total += cost_after
        n_moved += blk_n_moved
        converged = min(converged, blk_converged)
        n_iters = max(n_iters, blk_n_iters)
        n_repaired += round(blk_frac_repaired * (blk_end - blk_start))

    mean_cost_before = cost_before_total / bs
    mean_cost_after = cost_after_total / bs
    frac_moved = n_moved / bs
    frac_repaired = n_repaired / bs
    # Blocks never share index ranges (global_idx[a:b] is always shifted
    # into [a, b)), so a duplicate can only occur within a single block --
    # counting uniques over the whole batch is exactly equivalent to
    # summing per-block unique counts. Always 1.0: for method="exact" each
    # block's coupling is already a bijection; for method="sinkhorn" the
    # duplicate-repair pass (see _repair_duplicate_assignments) turns the
    # raw entropic sampling into a bijection too, so this is now a sanity
    # check rather than a live diagnostic -- frac_repaired below is the
    # metric that actually varies for "sinkhorn".
    frac_unique = len(np.unique(global_idx)) / bs

    return (
        x1_out,
        global_idx,
        mean_cost_before,
        mean_cost_after,
        frac_moved,
        frac_unique,
        converged,
        n_iters,
        frac_repaired,
    )
