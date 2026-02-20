"""
Dynamics models for XFADS.

This module implements concrete dynamics models for state transitions in
XFADS. Abstract interfaces are defined in ``jaxfads.base``.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jr

from .base import Noise
from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx


def predict_moment(
    z: Array,
    u: Array,
    c: Array,
    f: Callable[..., Array],
    noise: Noise,
    approx: type[Approx],
    *,
    key: Array | None = None,
) -> Array:
    """
    Predict moment parameters for next state given current state.

    Computes the moment parameters of p(z_{t+1} | z_t, u_t, c_t) by
    applying the dynamics function and incorporating process noise.

    Parameters
    ----------
    z : Array, shape (state_dim,)
        Current state vector.
    u : Array, shape (input_dim,)
        Control/input vector.
    c : Array, shape (covariate_dim,)
        Covariate vector.
    f : Callable
        Dynamics function mapping (z, u, c) -> z_next.
    noise : Noise
        Noise model providing covariance structure.
    approx : type[Approx]
        Exponential family approximation class.
    key : PRNGKeyArray, optional
        Random key for stochastic dynamics.

    Returns
    -------
    Array
        Moment parameters of the predictive distribution p(z_{t+1} | z_t, u_t, c_t).

    Notes
    -----
    The predictive distribution is constructed as:
    p(z_{t+1} | z_t) = N(f(z_t, u_t, c_t), Σ_noise)

    where f is the deterministic dynamics and Σ_noise is the process noise.
    """
    ztp1 = f(z, u, c, key=key)
    M2 = approx.noise_moment(noise.cov())
    moment = approx.canon_to_moment(ztp1, M2)
    return moment


def sample_expected_moment(
    key: Array,
    moment: Array,
    u: Array,
    c: Array,
    f: Callable[..., Array],
    noise: Noise,
    approx: type[Approx],
    mc_size: int,
) -> Array:
    """
    Compute expected moment parameters via Monte Carlo sampling.

    Approximates E_{p(z_t)}[μ(z_t, u_t, c_t)] where μ(·) gives the moment
    parameters of the predictive distribution p(z_{t+1} | z_t, u_t, c_t).
    This expectation is intractable for nonlinear dynamics, so we use
    Monte Carlo approximation.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for sampling.
    moment : Array
        Moment parameters of current state distribution p(z_t).
    u : Array, shape (input_dim,)
        Control/input vector.
    c : Array, shape (covariate_dim,)
        Covariate vector.
    f : Callable
        Dynamics function.
    noise : Noise
        Process noise model.
    approx : type[Approx]
        Exponential family approximation.
    mc_size : int
        Number of Monte Carlo samples.

    Returns
    -------
    Array
        Expected moment parameters E_{p(z_t)}[μ(z_t, u_t, c_t)].

    Notes
    -----
    The Monte Carlo approximation is:
    E[μ(z_t)] ≈ (1/K) Σ_{k=1}^K μ(z_t^{(k)})

    where z_t^{(k)} ~ p(z_t) are samples from the current state distribution.

    The same ``key`` is intentionally reused for every MC sample when
    evaluating ``f``.  This keeps any stochastic regularisation inside
    ``f`` (e.g. dropout) fixed within the expectation: the MC estimate
    integrates over latent uncertainty z ~ q(z_t) only, not over
    dropout randomness.

    Non-finite handling:
    After computing per-sample moments, any sample containing NaN or Inf
    values is masked out. The mean is taken only over valid (all-finite)
    samples. If *every* sample is non-finite, the function falls back to
    the deterministic prediction at the posterior mean ``z_mean`` (no MC).
    This makes the MC estimate robust against rare dynamics blow-ups
    (e.g. stiff ODEs, overflow in ``exp``) without requiring users to
    clamp their dynamics functions.
    """
    key, subkey = jr.split(key)
    z = approx.sample_by_moment(subkey, moment, mc_size)
    u = jnp.broadcast_to(u, shape=(mc_size,) + u.shape)
    c = jnp.broadcast_to(c, shape=(mc_size,) + c.shape)
    f_vmap_sample_axis = jax.vmap(
        partial(predict_moment, f=f, noise=noise, approx=approx, key=key),
        in_axes=(0, 0, 0),
    )
    moments = f_vmap_sample_axis(z, u, c)  # (mc_size, param_size)

    # --- nonfinite-safe aggregation ---
    # valid_k: True when all moment entries for sample k are finite
    valid = jnp.all(jnp.isfinite(moments), axis=-1)  # (mc_size,)
    n_valid = jnp.sum(valid)
    # Masked mean (zero out entire invalid rows so NaN/Inf don't poison the sum)
    safe_moments = jnp.where(valid[:, None], moments, 0.0)
    mc_mean = jnp.sum(safe_moments, axis=0) / jnp.maximum(n_valid, 1.0)

    # Fallback: deterministic prediction at the posterior mean (deferred)
    def _fallback(_):
        z_mean, _ = approx.moment_to_canon(moment)
        return predict_moment(
            z_mean, u[0], c[0], f=f, noise=noise, approx=approx, key=key
        )

    return jax.lax.cond(n_valid > 0, lambda _: mc_mean, _fallback, None)


class DiagGaussian(eqx.Module, strict=True):
    """
    Diagonal Gaussian noise model for dynamics systems.

    Implements process noise with diagonal covariance structure, assuming
    independence between state dimensions. More efficient than full
    covariance but less expressive.

    Parameters
    ----------
    cov : ArrayLike
        Initial covariance value (Array applied to all dimensions).
    size : int
        Dimensionality of the noise (should match state dimension).

    Attributes
    ----------
    unconstrained_cov : Array, shape (size,)
        Unconstrained covariance parameters for optimization.

    Notes
    -----
    The covariance is parameterized in unconstrained space to ensure
    positive values during optimization. The actual covariance is
    obtained via constrain_positive() transformation.
    """

    unconstrained_cov: Array

    def __init__(self, cov: Array, size: int):  # pyright: ignore[reportMissingSuperCall]
        self.unconstrained_cov = jnp.full(size, fill_value=unconstrain_positive(cov))

    def cov(self) -> Array:
        """
        Get the diagonal covariance vector.

        Returns
        -------
        Array, shape (size,)
            Diagonal elements of the covariance matrix.

        Notes
        -----
        Applies positive constraint to ensure valid covariance values.
        """
        return constrain_positive(self.unconstrained_cov)

    # def set_static(self, static=True) -> None:
    #     self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}
