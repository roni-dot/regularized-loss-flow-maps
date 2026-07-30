"""
Monge gap regularizer using OTT-JAX's built-in monge_gap_from_samples,
averaged over K independent (s, t) pairs sharing one base point cloud.
"""

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.neural.methods import monge_gap
from flax import linen as nn

from . import interpolant as interpolant


def compute_monge_gap_reg(
    params,
    net: nn.Module,
    interp: interpolant.Interpolant,
    mg_x0: jnp.ndarray,
    mg_x1: jnp.ndarray,
    s_vec: jnp.ndarray,
    t_vec: jnp.ndarray,
    epsilon: float,
    relative_epsilon: bool,
    cost_fn=None,
     **sinkhorn_kwargs,
) -> jnp.ndarray:
    """
    K independent (s, t) pairs, each applied to the SAME (mg_x0, mg_x1) base
    batch.
    """

    def single_pair(s, t):
        I_s = jax.vmap(lambda x0i, x1i: interp.calc_It(s, x0i, x1i))(mg_x0, mg_x1)
        X_st = jax.vmap(
            lambda xi: net.apply(params, s, t, xi, None, train=False)
        )(I_s)

        gap = monge_gap.monge_gap_from_samples(
            I_s, X_st,
            cost_fn=cost_fn,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon,
            return_output=False,
            **sinkhorn_kwargs,
        )
        return gap
    
    gaps = jax.vmap(single_pair)(s_vec, t_vec)
    return jnp.mean(gaps)