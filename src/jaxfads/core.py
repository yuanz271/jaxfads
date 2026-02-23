"""
Core filtering and smoothing algorithms for XFADS.

This module implements the fundamental algorithms for XFADS,
including forward filtering and bidirectional smoothing
using variational inference in exponential family approximations.
"""

from collections.abc import Callable
from enum import StrEnum, auto
from functools import partial

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jrnd
from jax.lax import scan

from .base import Noise
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
    key, subkey = jrnd.split(key)
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


class Mode(StrEnum):
    """
    Enumeration of inference modes for XFADS.

    Attributes
    ----------
    PSEUDO : str
        Pseudo-observation mode using forward filtering.
    BIFILTER : str
        Bidirectional filtering mode (not tested).
    """

    PSEUDO = auto()
    BIFILTER = auto()


def filter(
    model,
    key: Array,
    _t: Array,
    alpha: Array,
    u: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """
    Forward filtering for state estimation in XFADS.

    Performs sequential Bayesian filtering to estimate latent states given
    observations. Uses variational inference with exponential family
    approximations and Monte Carlo sampling for intractable expectations.

    Parameters
    ----------
    model : XFADS
        The XFADS model containing dynamics and hyperparameters.
    key : Array
        JAX random number generator key for stochastic operations.
    _t : Array, shape (T,)
        Time steps for the sequence (unused in current implementation).
    alpha : Array, shape (T, param_dim)
        Information updates from observations in natural parameter form.
    u : Array, shape (T, input_dim)
        External control/input signals.
    c : Array, shape (T, covariate_dim)
        Time-varying covariates.

    Returns
    -------
    nature_f : Array, shape (T, param_dim)
        Filtered natural parameters for each time step.
    moment_f : Array, shape (T, param_dim)
        Filtered moment parameters for each time step.
    moment_p : Array, shape (T, param_dim)
        Predicted moment parameters from dynamics.

    Notes
    -----
    The filtering recursion follows:

    1. Prediction: p(z_t | y_{1:t-1}) from dynamics
    2. Update: p(z_t | y_{1:t}) ∝ p(z_t | y_{1:t-1}) p(y_t | z_t)

    Uses natural parameter representation for numerical stability.
    """
    approx = model.approx
    nature_p_1 = (
        model.prior_natural()
    )  # TODO: where should prior belongs, approx or dynamics?

    expected_moment_forward = partial(
        sample_expected_moment,
        f=model.forward,
        noise=model.forward,
        approx=approx,
        mc_size=model.conf.mc_size,
    )

    nature_f_1 = nature_p_1 + alpha[0]

    def ff(carry, obs, expected_moment):
        key, nature_tm1 = carry
        key, ky = jrnd.split(key)
        a_t, u_tm1, c_tm1 = obs
        moment_tm1 = approx.natural_to_moment(nature_tm1)
        moment_p_t = expected_moment(ky, moment_tm1, u_tm1, c_tm1)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_t = nature_p_t + a_t
        return (key, nature_t), (moment_p_t, nature_p_t, nature_t)

    key, ky = jrnd.split(key)
    scan_body = eqx.filter_checkpoint(
        partial(ff, expected_moment=expected_moment_forward)
    )
    _, (moment_p, _, nature_f) = scan(
        scan_body,
        init=(ky, nature_f_1),
        xs=(alpha[1:], u[:-1], c[:-1]),  # t = 2 ... T+1
    )
    nature_f = jnp.vstack((nature_f_1, nature_f))  # 1...T

    moment_f = jax.vmap(approx.natural_to_moment)(nature_f)
    moment_p = jnp.vstack(
        (approx.natural_to_moment(nature_f_1), moment_p)
    )  # prediction of t=1 is the prior

    return nature_f, moment_f, moment_p


# NOTE: bismooth() requires model.backward (a Dynamics instance) which is
# not yet implemented on XFADS.  Do not call until backward dynamics are added.
def bismooth(
    model,
    key: Array,
    _t: Array,
    alpha: Array,
    u: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """
    Bidirectional filtering for improved state smoothing in XFADS.

    Implements bidirectional variational inference by combining forward
    and backward information. Uses parameterized inverse dynamics to
    propagate information backward in time, resulting in better posterior
    approximations compared to forward filtering alone.

    Parameters
    ----------
    model : XFADS
        The XFADS model containing forward/backward dynamics.
    key : Array
        JAX random number generator key for stochastic operations.
    _t : Array, shape (T,)
        Time steps for the sequence (unused in current implementation).
    alpha : Array, shape (T, param_dim)
        Information updates from observations in natural parameter form.
    u : Array, shape (T, input_dim)
        External control/input signals.
    c : Array, shape (T, covariate_dim)
        Time-varying covariates.

    Returns
    -------
    nature_s : Array, shape (T, param_dim)
        Smoothed natural parameters combining forward and backward passes.
    moment_s : Array, shape (T, param_dim)
        Smoothed moment parameters.
    moment_p : Array, shape (T, param_dim)
        Predicted moment parameters under smoothing distribution.

    Notes
    -----
    The bidirectional combination follows:

    q(z_t|y_{1:T}) = q(z_t|y_{1:t}) q(z_t|y_{t+1:T}) / p(z_t)

    In natural parameters:
    η_s[t] = η_f[t] + η_b[t] - η_0

    where η_f, η_b, η_0 are forward, backward, and prior natural parameters.

    References
    ----------
    Dowling et al. (2023). Linear Time GPs for Inferring Latent Trajectories
        from Neural Spike Trains. https://arxiv.org/abs/2306.01802.
        Equations (21-23).
    """
    mc_size = model.conf.mc_size
    approx = model.approx
    nature_prior = model.prior_natural()

    natural_to_moment = jax.vmap(approx.natural_to_moment)
    expected_moment_forward = partial(
        sample_expected_moment,
        f=model.forward,
        noise=model.forward,
        approx=approx,
        mc_size=mc_size,
    )
    expected_moment_backward = partial(
        sample_expected_moment,
        f=model.backward,
        noise=model.backward,
        approx=approx,
        mc_size=mc_size,
    )

    nature_f_1 = nature_prior + alpha[0]

    def ff(carry, obs, expected_moment):
        key, nature_f_tm1 = carry
        key_tp1, key_t = jrnd.split(key)
        update_obs_t, u, c = obs
        moment_f_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = expected_moment(key_t, moment_f_tm1, u, c)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_f_t = nature_p_t + update_obs_t
        return (key_tp1, nature_f_t), (moment_p_t, nature_p_t, nature_f_t)

    # Forward
    # TODO: checkpoint scan body when bismooth is implemented
    key, forward_key = jrnd.split(key)
    _, (_, _, nature_f) = scan(
        partial(ff, expected_moment=expected_moment_forward),
        init=(forward_key, nature_f_1),
        xs=(alpha[1:], u[:-1], c[:-1]),  # t = 2 ... T+1
    )
    nature_f = jnp.vstack((nature_f_1, nature_f))  # 1...T

    ## Backward
    key, ky = jrnd.split(key)
    (_, nature_b_Tp1), _ = ff(
        (ky, nature_f[-1]),
        (jnp.zeros_like(nature_prior), u[-1], c[-1]),
        expected_moment_forward,
    )

    key, backward_key = jrnd.split(key)
    _, (_, nature_p_b, _) = scan(
        partial(ff, expected_moment=expected_moment_backward),
        init=(backward_key, nature_b_Tp1),
        xs=(alpha, u, c),
        reverse=True,
    )

    nature_s = nature_f + nature_p_b - jnp.expand_dims(nature_prior, axis=0)
    moment_s = natural_to_moment(nature_s)

    # expectation should be under smoothing distribution
    keys = jrnd.split(key, jnp.size(moment_s, 0))
    moment_p = jax.vmap(expected_moment_forward)(keys, moment_s, u, c)
    moment_p = jnp.vstack((moment_s[0], moment_p[:-1]))

    return nature_s, moment_s, moment_p
