"""
Same collective-correctness test as debug_pmean.py, but using JAX's modern
shard_map/Mesh API instead of the legacy jax.pmap. If THIS gives the correct
answer while jax.pmap gives the wrong one, the bug is in jax.pmap's
compatibility shim in this JAX version -- fixable by switching the training
code's pmap calls to shard_map, no cluster/IT fix needed.

Usage:
    python -u launchers/debug_shardmap.py
"""

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


def main():
    devices = jax.devices()
    ndevices = len(devices)
    print(f"JAX devices ({ndevices}): {devices}")

    if ndevices < 2:
        print("Only 1 device visible -- this test needs >1 GPU to be meaningful.")
        return

    mesh = Mesh(devices, axis_names=("d",))

    per_device_values = jnp.arange(1, ndevices + 1, dtype=jnp.float32)
    print(f"Per-device input values: {per_device_values}")
    expected_mean = float(jnp.mean(per_device_values))
    expected_sum = float(jnp.sum(per_device_values))
    print(f"Expected pmean: {expected_mean}, expected psum: {expected_sum}")

    def pmean_body(x):
        return jax.lax.pmean(x, axis_name="d")

    def psum_body(x):
        return jax.lax.psum(x, axis_name="d")

    pmean_shmap = shard_map(
        pmean_body, mesh=mesh, in_specs=P("d"), out_specs=P("d")
    )
    psum_shmap = shard_map(
        psum_body, mesh=mesh, in_specs=P("d"), out_specs=P("d")
    )

    mean_result = jax.jit(pmean_shmap)(per_device_values)
    print(f"\nshard_map pmean result: {mean_result}")
    print(f"All finite: {bool(jnp.isfinite(mean_result).all())}")
    print(f"All equal to expected ({expected_mean}): {bool(jnp.allclose(mean_result, expected_mean))}")

    sum_result = jax.jit(psum_shmap)(per_device_values)
    print(f"\nshard_map psum result: {sum_result}")
    print(f"All finite: {bool(jnp.isfinite(sum_result).all())}")
    print(f"All equal to expected ({expected_sum}): {bool(jnp.allclose(sum_result, expected_sum))}")


if __name__ == "__main__":
    main()
