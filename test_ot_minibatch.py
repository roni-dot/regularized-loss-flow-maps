"""
Pipeline check for minibatch OT.

Drop-in test of the exact reorder you would put in the data path:
sample (x0, x1), solve exact OT within chunks, reorder x1, and see how
much the mean transport cost actually drops.

  reduction ~  0%  -> concentration regime, OT permutation is near-arbitrary
  reduction >> 0%  -> real structure for OT-CFM to exploit

Also reports how many pairs actually move, which is the more direct
question: if the OT plan is close to the identity permutation on an
already-random pairing, nothing has changed.

Usage
-----
    python ot_pipeline_check.py                     # 2D checker reference only
    python ot_pipeline_check.py /path/to/cifar.npy  # + real CIFAR

The .npy should be (N, 32, 32, 3) or (N, 3072), any scaling; it is
rescaled to [-1, 1] automatically if it looks like uint8 range.
"""
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(0)


def ot_reorder(x0, x1):
    """Exact OT within one chunk. Returns permutation for x1."""
    C = (x0**2).sum(1)[:, None] + (x1**2).sum(1)[None, :] - 2.0 * x0 @ x1.T
    _, col = linear_sum_assignment(C)
    return col


def check(name, sample_x0, sample_x1, chunk_sizes, n_rep=3):
    print(f"\n=== {name} ===")
    print(f"{'chunk':>7} {'cost before':>13} {'cost after':>12} "
          f"{'reduction':>11} {'pairs moved':>12}")
    for k in chunk_sizes:
        before, after, moved = [], [], []
        for _ in range(n_rep):
            x0, x1 = sample_x0(k), sample_x1(k)
            perm = ot_reorder(x0, x1)
            before.append(((x0 - x1) ** 2).sum(1).mean())
            after.append(((x0 - x1[perm]) ** 2).sum(1).mean())
            moved.append((perm != np.arange(k)).mean())
        b, a, m = np.mean(before), np.mean(after), np.mean(moved)
        print(f"{k:>7} {b:>13.3f} {a:>12.3f} "
              f"{100*(1-a/b):>10.1f}% {100*m:>11.1f}%")


# ---------------- 2D checkerboard: reference regime ----------------
def sample_checker(n):
    out = []
    while len(out) < n:
        p = rng.uniform(-1, 1, size=(4 * n, 2))
        keep = (np.floor((p[:, 0] + 1) * 2).astype(int)
                + np.floor((p[:, 1] + 1) * 2).astype(int)) % 2 == 0
        out.extend(p[keep])
    return np.asarray(out[:n])


check("checker d=2  (OT-CFM is known to help here)",
      lambda n: 0.5 * rng.normal(size=(n, 2)),
      sample_checker,
      [128, 256, 512])


# ---------------- CIFAR: the regime you care about ----------------
if len(sys.argv) > 1:
    data = np.load(sys.argv[1])
    data = data.reshape(len(data), -1).astype(np.float32)
    if data.max() > 2.0:                      # looks like 0..255
        data = data / 127.5 - 1.0
    print(f"\nloaded {data.shape}  std={data.std():.4f}  "
          f"range=[{data.min():.2f}, {data.max():.2f}]")
    check("CIFAR-10 d=3072  (REAL DATA)",
          lambda n: rng.normal(size=(n, 3072)),
          lambda n: data[rng.choice(len(data), n, replace=False)],
          [128, 256, 512])
else:
    print("\n[no data path given - skipping CIFAR. pass your .npy to test it]")

print("\nRead the CIFAR reduction against the checker reduction.")
print("A small reduction does not by itself mean OT-CFM will fail -- the")
print("mechanism is variance reduction in the regression target, not cost --")
print("but it tells you how much geometric structure the coupling has.")