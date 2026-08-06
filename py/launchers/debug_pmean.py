"""
Minimal test: does jax.lax.pmean across all local GPUs even work correctly?
No model, no data pipeline -- just a basic multi-device collective all-reduce.
If THIS produces NaN/wrong results, the problem is NCCL/GPU-interconnect/driver
on this node, not anything in the training code.

Usage:
    python -u launchers/debug_pmean.py
"""

import functools

import jax
import jax.numpy as jnp


def main():
    devices = jax.devices()
    ndevices = len(devices)
    print(f"JAX devices ({ndevices}): {devices}")

    if ndevices < 2:
        print("Only 1 device visible -- this test needs >1 GPU to be meaningful.")
        return

    # each device gets a different, known, finite value
    per_device_values = jnp.arange(1, ndevices + 1, dtype=jnp.float32)
    print(f"Per-device input values: {per_device_values}")
    expected_mean = float(jnp.mean(per_device_values))
    print(f"Expected pmean result (same on every device): {expected_mean}")

    @functools.partial(jax.pmap, axis_name="d")
    def pmean_fn(x):
        return jax.lax.pmean(x, axis_name="d")

    result = pmean_fn(per_device_values)
    print(f"Actual pmean result per device: {result}")
    print(f"All finite: {bool(jnp.isfinite(result).all())}")
    print(f"All equal to expected: {bool(jnp.allclose(result, expected_mean))}")

    # also test psum, and a pmean over something the same shape/dtype as a
    # real gradient leaf, to rule out shape/dtype-specific issues
    @functools.partial(jax.pmap, axis_name="d")
    def psum_fn(x):
        return jax.lax.psum(x, axis_name="d")

    psum_result = psum_fn(per_device_values)
    expected_sum = float(jnp.sum(per_device_values))
    print(f"\npsum result per device: {psum_result}")
    print(f"All finite: {bool(jnp.isfinite(psum_result).all())}")
    print(f"All equal to expected sum ({expected_sum}): {bool(jnp.allclose(psum_result, expected_sum))}")

    # larger random tensor, closer to a real conv weight's size/shape
    key = jax.random.PRNGKey(0)
    big = jax.random.normal(key, (ndevices, 256, 256, 3, 3), dtype=jnp.float32)
    big_result = pmean_fn(big)
    print(f"\nLarge-tensor pmean all finite: {bool(jnp.isfinite(big_result).all())}")
    print(f"nan count: {int(jnp.sum(jnp.isnan(big_result)))}")
    print(f"inf count: {int(jnp.sum(jnp.isinf(big_result)))}")


if __name__ == "__main__":
    main()
