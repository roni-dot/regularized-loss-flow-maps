"""
Monge gap regularizer using OTT-JAX's built-in monge_gap_from_samples,
averaged over K independent (s, t) pairs sharing one base point cloud.
"""

from typing import Literal

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
    relative_epsilon: Literal["mean", "std"] = "mean",
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

        # OTT's PointCloud/Sinkhorn solver expects 2D (num_points, feature_dim)
        # inputs. I_s/X_st are images with shape (batch, C, H, W); passing
        # them in unflattened causes OTT to misread a spatial dimension as a
        # point count, producing a shape mismatch against the true batch size.
        n_points = I_s.shape[0]
        I_s_flat = I_s.reshape(n_points, -1)
        X_st_flat = X_st.reshape(n_points, -1)

        gap, out = monge_gap.monge_gap_from_samples(
            I_s_flat, X_st_flat,
            cost_fn=cost_fn,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon,
            return_output=True,
            **sinkhorn_kwargs,
        )
        if cost_fn is None:
            mean_cost = jnp.mean(((I_s_flat - X_st_flat) ** 2).sum(-1))
        else:
            mean_cost = jnp.mean(jax.vmap(cost_fn)(I_s_flat, X_st_flat))
        return gap, out.reg_ot_cost, mean_cost, out.converged, out.n_iters

    gaps, reg_ot_costs, mean_costs, converged, n_iters = jax.vmap(single_pair)(s_vec, t_vec)
    return (
        jnp.mean(gaps),
        jnp.mean(reg_ot_costs),
        jnp.mean(mean_costs),
        jnp.mean(converged.astype(jnp.float32)),
        jnp.max(n_iters).astype(jnp.float32),
    )