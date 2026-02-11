"""Host-side diagnostics helpers.

These utilities are designed to help debug training/inference while keeping
all Python logging outside any JAX-jitted function.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp

from .trainer import batch_elbo, batch_loss


@dataclass(frozen=True)
class LossStats:
    loss: float
    loss_is_finite: bool
    posterior_has_nonfinite: bool
    prior_has_nonfinite: bool


def compute_loss_stats(model, batch, *, key: Array) -> LossStats:
    """Compute a few scalar diagnostics for a given batch.

    Notes
    -----
    This function performs computation and returns scalars/flags only. It does
    not log or print.
    """

    @eqx.filter_jit
    def _stats(m, b, k):
        times, observations, controls, covariates = b
        k, model_key = jax.random.split(k)
        _, posterior_moments, prior_moments = m(
            times, observations, controls, covariates, key=model_key
        )

        k, elbo_key = jax.random.split(k)
        free_energy = -batch_elbo(
            m, elbo_key, times, posterior_moments, prior_moments, observations
        )
        loss_val = jnp.mean(free_energy) + m.conf.noise_penalty * m.forward.loss()
        loss_finite = jnp.isfinite(loss_val)
        posterior_nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(posterior_moments)))
        prior_nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(prior_moments)))
        return loss_val, loss_finite, posterior_nonfinite, prior_nonfinite

    loss_val, loss_finite, posterior_nonfinite, prior_nonfinite = _stats(
        model, batch, key
    )
    return LossStats(
        loss=float(loss_val.item()),
        loss_is_finite=bool(loss_finite.item()),
        posterior_has_nonfinite=bool(posterior_nonfinite.item()),
        prior_has_nonfinite=bool(prior_nonfinite.item()),
    )


@dataclass(frozen=True)
class GradStats:
    grad_global_norm: float
    grad_has_nonfinite: bool


def compute_grad_stats(model, batch, *, key: Array) -> GradStats:
    """Compute gradient norm and non-finite flag for a batch."""

    @eqx.filter_jit
    def _grad_stats(m, b, k):
        grads = eqx.filter_grad(batch_loss)(m, b, k)
        grads = eqx.filter(grads, eqx.is_inexact_array)
        leaves = jax.tree.leaves(grads)
        sq = jnp.sum(jnp.array([jnp.vdot(g, g).real for g in leaves]))
        norm = jnp.sqrt(sq)
        nonfinite = jnp.any(
            jnp.array([jnp.any(jnp.logical_not(jnp.isfinite(g))) for g in leaves])
        )
        return norm, nonfinite

    norm, nonfinite = _grad_stats(model, batch, key)
    return GradStats(
        grad_global_norm=float(norm.item()),
        grad_has_nonfinite=bool(nonfinite.item()),
    )
