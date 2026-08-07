"""Concrete trainer-owned closed-form M-step transformations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax
from jax import numpy as jnp

from ..constraints import _EPS, _MIN_VARIANCE, unconstrain_positive
from ..distributions.mvn import MVN


def gaussian_observation_stat(t, y, moment, approx, readout):
    unpack = jax.vmap(jax.vmap(approx.unpack))
    mean_z, cov_z = unpack(moment)
    mean_y = jax.vmap(jax.vmap(readout))(t, mean_z)
    residual_sq = (y - mean_y) ** 2
    propagated_var = jnp.einsum(
        "dj,...jk,dk->...d", readout.weight, cov_z, readout.weight
    )
    return residual_sq + propagated_var


class GaussianObservationMstep:
    """Update diagonal Gaussian observation covariance from posterior moments."""

    def initialize(self, model, *, key):
        del key
        return model

    def __call__(self, model, batch, forward, *, key):
        del key
        t, y, _u, _c = batch
        _natural, moment, _predictive = forward
        observation = model.observation
        likelihood = observation.likelihood
        if not hasattr(likelihood, "unconstrained_cov"):
            return model
        stat = gaussian_observation_stat(
            t, y, moment, model.approx, observation.readout
        )
        valid = jnp.isfinite(y)
        sums = jnp.sum(jnp.where(valid, stat, 0), axis=(0, 1))
        counts = jnp.sum(valid, axis=(0, 1))
        r_new = sums / jnp.maximum(counts, 1)
        free = unconstrain_positive(jnp.maximum(r_new - _MIN_VARIANCE, _EPS))
        new_likelihood = eqx.tree_at(
            lambda leaf: leaf.unconstrained_cov, likelihood, free
        )
        new_observation = eqx.tree_at(
            lambda leaf: leaf.likelihood, observation, new_likelihood
        )
        return eqx.tree_at(lambda leaf: leaf.observation, model, new_observation)

    def frozen_paths(self, model) -> list[str]:
        likelihood = getattr(model.observation, "likelihood", None)
        if likelihood is None or not hasattr(likelihood, "unconstrained_cov"):
            return []
        return ["observation.likelihood.unconstrained_cov"]


@dataclass(frozen=True)
class MVNNoiseMstep:
    """Apply an isotropic Q prior and update Q from predictive moments."""

    q_scale: float = 1.0
    q_prior_fraction: float = 0.1

    def __post_init__(self):
        if self.q_scale <= 0 or not isfinite(self.q_scale):
            raise ValueError("q_scale must be finite and positive")
        if self.q_prior_fraction < 0 or not isfinite(self.q_prior_fraction):
            raise ValueError("q_prior_fraction must be finite and nonnegative")

    def initialize(self, model, *, key):
        del key
        approx = model.approx
        if not isinstance(approx, MVN):
            raise NotImplementedError(
                "MVNNoiseMstep requires model.approx to be an MVN instance"
            )
        free = approx.free_from_kw(scale=self.q_scale)
        return eqx.tree_at(lambda leaf: leaf.noise, model, free)

    def __call__(self, model, batch, forward, *, key):
        del batch, key
        approx = model.approx
        if not isinstance(approx, MVN):
            raise NotImplementedError(
                "MVNNoiseMstep requires model.approx to be an MVN instance"
            )
        _natural, moment, predictive_moment = forward
        unpack = jax.vmap(jax.vmap(approx.unpack))
        mean_t, cov_t = unpack(moment[:, 1:])
        mean_f, cov_pred = unpack(predictive_moment[:, 1:])
        _, q = approx.unpack(approx.canon_to_moment(approx.free_to_canon(model.noise)))
        cov_f = cov_pred - q
        cov_f = 0.5 * (cov_f + jnp.swapaxes(cov_f, -1, -2))
        residual = mean_t - mean_f
        raw = jnp.einsum("...i,...j->...ij", residual, residual) + cov_t + cov_f
        q_hat = jnp.mean(raw, axis=(0, 1))
        d = q_hat.shape[-1]
        prior = self.q_scale * jnp.eye(d, dtype=q_hat.dtype)
        cov = (q_hat + self.q_prior_fraction * prior) / (1.0 + self.q_prior_fraction)
        cov = 0.5 * (cov + cov.T)
        if approx._layout.is_diag:
            cov = jnp.diag(jnp.diagonal(cov))
        cov = cov + _EPS * jnp.eye(d, dtype=cov.dtype)
        free = approx.canon_to_free(
            approx.moment_to_canon(approx.pack(jnp.zeros(d, dtype=cov.dtype), cov))
        )
        return eqx.tree_at(lambda leaf: leaf.noise, model, free)

    def frozen_paths(self, model) -> list[str]:
        if not isinstance(model.approx, MVN):
            raise NotImplementedError(
                "MVNNoiseMstep requires model.approx to be an MVN instance"
            )
        return ["noise"]
