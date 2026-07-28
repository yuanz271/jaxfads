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

from .base import Approx

__all__ = [
    "expected_predictive_moment",
    "mstep_transition_stat",
    "Mode",
    "filter",
    "smooth",
    "causal",
    "nofilt",
]


def expected_predictive_moment(
    key: Array,
    moment: Array,
    u: Array,
    c: Array,
    f: Callable[..., Array],
    noise: Array,
    approx: Approx,
    mc_size: int,
) -> Array:
    """
    Compute expected predictive moment via a weighted point set from
    ``approx.transition_points`` (default: Monte Carlo; some ``Approx``
    implementations may override with a deterministic point set, e.g.
    unscented-transform sigma points -- see ``docs/transition_points.md``).

    Implements Eq (12): ``μ̄_t = E_{π(z_{t-1})}[μ_θ(z_{t-1})]``
    where ``μ_θ`` is the predictive moment function (Eq 4)
    ``μ_θ(z_{t-1}) = E_{p(z_t|z_{t-1})}[T(z_t)]``.

    Averaging is performed in moment-parameter space, i.e. the output of
    ``approx.predictive_moment``.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for sampling.
    moment : Array
        Moment parameters of current state distribution π(z_{t-1}).
    u : Array, shape (input_dim,)
        Control/input vector.
    c : Array, shape (covariate_dim,)
        Covariate vector.
    f : Callable
        One-step transition callable.
    noise : Array
        Transition noise parameters passed through to ``approx``.
    approx : Approx
        Exponential family approximation instance.
    mc_size : int
        Requested point/sample count; may be ignored by deterministic
        ``transition_points`` overrides.

    Returns
    -------
    Array
        Expected predictive moment parameters (i.e. weighted-averaged
        ``E[T(z_t) | z_{t-1}]``).

    Notes
    -----
    The same ``key`` is intentionally reused for every point when
    evaluating ``f``.  This keeps any stochastic regularisation inside
    ``f`` (e.g. dropout) fixed within the expectation.

    Non-finite handling:
    After computing per-point predictive moment, any point containing
    NaN or Inf values is masked out (its weight zeroed) before the
    weighted average. If every point is non-finite the result will itself
    be non-finite.
    """
    key, subkey = jrnd.split(key)
    z, weights = approx.transition_points(subkey, moment, mc_size)
    n_points = z.shape[0]  # not mc_size -- deterministic policies may return
    # a different, fixed point count (e.g. unscented-transform sigma points
    # always return 2*dim + 1, regardless of the mc_size argument).
    u_bc = jnp.broadcast_to(u, shape=(n_points,) + u.shape)
    c_bc = jnp.broadcast_to(c, shape=(n_points,) + c.shape)

    # Transition outputs for each point
    zs = jax.vmap(partial(f, key=key), in_axes=(0, 0, 0))(z, u_bc, c_bc)

    # Per-point predictive moment
    predictive_moment_samples = jax.vmap(
        partial(approx.predictive_moment, noise=noise)
    )(zs)

    # Non-finite safe weighted averaging. Strict generalization of the
    # former uniform-weight reduction (sum(safe)/n_valid): for weights
    # w_i = 1/mc_size, sum(w_valid*safe)/sum(w_valid) reduces to that
    # formula exactly.
    valid = jnp.all(jnp.isfinite(predictive_moment_samples), axis=-1)  # (n_points,)
    safe = jnp.where(valid[:, None], predictive_moment_samples, 0.0)
    w_valid = jnp.where(valid, weights, 0.0)
    w_sum = jnp.sum(w_valid)
    avg = jnp.where(
        w_sum > 0,
        jnp.sum(w_valid[:, None] * safe, axis=0) / w_sum,
        jnp.full(predictive_moment_samples.shape[-1:], jnp.nan),
    )

    return avg


def mstep_transition_stat(
    approx: Approx,
    moment_t: Array,
    moment_tm1: Array,
    u: Array,
    c: Array,
    transition_fn: Callable[..., Array],
    *,
    key: Array,
) -> Array:
    """Per-(batch,time) sufficient statistic for the transition-noise
    M-step, computed entirely from smoothed quantities.

    Standalone, not a method on ``Approx`` or ``Dynamics`` -- needs both
    ``approx.unpack`` (distribution-only) and ``transition_fn``
    (dynamics-aware), so belongs to neither; same shape of problem as
    :func:`expected_predictive_moment`, which is the direct precedent for
    solving it this way.

    Deliberately decoupled from ``Approx.transition_points`` (MC or UT):
    that answers a different question ("propagate q(z_{t-1})'s
    uncertainty forward, before seeing y_t"); this statistic is a
    posterior expectation over the *joint* smoothed distribution of
    ``(z_{t-1}, z_t)``, which needs no forward-propagation machinery at
    all.

    v1 approximation: no cross-covariance term (``Cov(z_t, z_{t-1})`` is
    not exposed by this repo's smoothing algorithm -- see
    ``docs/mstep_dynamics_noise.md``). Formula, per (batch, time) pair::

        r = m' - transition_fn(m)
        J = jacrev(transition_fn)(m)
        raw_stat = outer(r, r) + P' + J @ P @ J.T

    where ``m, P = approx.unpack(moment_tm1)`` and
    ``m', P' = approx.unpack(moment_t)``.

    Parameters
    ----------
    approx : Approx
        Exponential family approximation instance.
    moment_t : Array, shape (N, T-1, param_dim)
        Smoothed moment parameters of ``q(z_t)``, ``t = 2 ... T``.
    moment_tm1 : Array, shape (N, T-1, param_dim)
        Smoothed moment parameters of ``q(z_{t-1})``, ``t-1 = 1 ... T-1``
        -- i.e. ``moment_t``/``moment_tm1`` are already the shifted,
        aligned pair (caller's responsibility, matching how ``u``/``c``
        below must be aligned the same way).
    u : Array, shape (N, T-1, input_dim)
    c : Array, shape (N, T-1, covariate_dim)
        Control/covariate at the *source* time step of each pair (i.e. the
        same slice used for ``u_tm1``/``c_tm1`` during filtering -- see
        ``filter()``'s own ``u[:-1], c[:-1]`` convention).
    transition_fn : Callable
        One-step transition callable, ``f(z, u, c, key=...) -> z``.
    key : Array
        JAX PRNG key, reused across all pairs (same rationale as
        :func:`expected_predictive_moment`: keeps any stochastic
        regularization inside ``f`` fixed for this statistic).

    Returns
    -------
    Array, shape (N, T-1, D, D)
        Un-reduced per-(batch,time) raw statistic. Reduction (mean) and
        MAP shrinkage toward a prior are the responsibility of
        ``Approx.shrink``, not this function.
    """

    def _single(moment_t_i, moment_tm1_i, u_i, c_i):
        mean_tm1, cov_tm1 = approx.unpack(moment_tm1_i)
        mean_t, cov_t = approx.unpack(moment_t_i)

        def f(z):
            return transition_fn(z, u_i, c_i, key=key)

        residual = mean_t - f(mean_tm1)
        jac = jax.jacrev(f)(mean_tm1)
        return jnp.outer(residual, residual) + cov_t + jac @ cov_tm1 @ jac.T

    stat_fn = jax.vmap(jax.vmap(_single))
    return stat_fn(moment_t, moment_tm1, u, c)


class Mode(StrEnum):
    """
    Enumeration of inference modes for XFADS.

    Attributes
    ----------
    FILTER : str
        Filtering mode using alpha-only updates.
    SMOOTH : str
        Smoothing mode using additive pseudo-observation updates.
    CAUSAL : str
        Causal Eq. 29 mode using alpha-only filtering plus beta reconstruction.
    """

    FILTER = auto()
    SMOOTH = auto()
    CAUSAL = auto()
    NOFILT = auto()


def _site_filter(
    model,
    key: Array,
    _t: Array,
    site_natural: Array,
    u: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """
    Generic forward recursion over natural-parameter site updates in XFADS.

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
    site_natural : Array, shape (T, param_dim)
        Natural-parameter site updates used in the recursion.
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
    The recursion follows:

    1. Prediction: p(z_t | y_{1:t-1}) from dynamics
    2. Update: p(z_t | y_{1:t}) ∝ p(z_t | y_{1:t-1}) p(y_t | z_t)

    Uses natural parameter representation for numerical stability.
    """
    approx = model.approx
    nature_p_1 = model.prior_natural()

    noise = approx.canon_to_moment(approx.free_to_canon(model.noise_free))

    expected_moment_forward = partial(
        expected_predictive_moment,
        f=model.transition,
        noise=noise,
        approx=approx,
        mc_size=model.conf.mc_size,
    )

    nature_f_1 = nature_p_1 + site_natural[0]

    def ff(carry, obs, expected_moment):
        key, nature_tm1 = carry
        key, ky = jrnd.split(key)
        site_t, u_tm1, c_tm1 = obs
        moment_tm1 = approx.natural_to_moment(nature_tm1)
        moment_p_t = expected_moment(ky, moment_tm1, u_tm1, c_tm1)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_t = nature_p_t + site_t
        return (key, nature_t), (moment_p_t, nature_p_t, nature_t)

    key, ky = jrnd.split(key)
    scan_body = eqx.filter_checkpoint(
        partial(ff, expected_moment=expected_moment_forward)
    )
    _, (moment_p, _, nature_f) = scan(
        scan_body,
        init=(ky, nature_f_1),
        xs=(site_natural[1:], u[:-1], c[:-1]),  # t = 2 ... T+1
    )
    nature_f = jnp.vstack((nature_f_1, nature_f))  # 1...T

    moment_f = jax.vmap(approx.natural_to_moment)(nature_f)
    moment_p = jnp.vstack(
        (approx.natural_to_moment(nature_f_1), moment_p)
    )  # prediction of t=1 is the prior

    return nature_f, moment_f, moment_p


def filter(
    model,
    key: Array,
    _t: Array,
    alpha: Array,
    u: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """
    Alpha-only filtering posterior recursion.

    This returns filtering natural/moment parameters and predictive moments.
    """
    return _site_filter(model, key, _t, alpha, u, c)


def smooth(
    model,
    key: Array,
    _t: Array,
    alpha: Array,
    beta: Array,
    u: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """
    Smoothing-side recursion using additive natural sites ``alpha + beta``.
    """
    return _site_filter(model, key, _t, alpha + beta, u, c)


def causal(
    model,
    key: Array,
    _t: Array,
    alpha: Array,
    beta: Array,
    u: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """
    Causal Eq. 29 inference path using the repository beta indexing convention.

    This mode first computes filtering natural parameters from alpha-only
    updates, then reconstructs the smoothing-side natural parameters by adding
    beta at the same code index:

    ``lambda_t = check_lambda_t + beta_t``.

    Under the paper indexing, this corresponds to
    ``lambda_t = check_lambda_t + beta_{t+1}``.
    """
    check_nature, _, moment_p = filter(model, key, _t, alpha, u, c)
    nature = check_nature + beta
    moment = jax.vmap(model.approx.natural_to_moment)(nature)
    return nature, moment, moment_p


def nofilt(
    model,
    key: Array,
    _t: Array,
    alpha: Array,
    u: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """Non-filter mode: posterior from encoder only, no filtering recursion.

    Posterior natural parameters are set directly by encoder output ``alpha``.
    Predictive moments are computed in parallel for ELBO KL terms.
    """
    approx = model.approx
    noise = approx.canon_to_moment(approx.free_to_canon(model.noise_free))

    nature = alpha
    moment = jax.vmap(approx.natural_to_moment)(nature)

    expected_moment_fn = partial(
        expected_predictive_moment,
        f=model.transition,
        noise=noise,
        approx=approx,
        mc_size=model.conf.mc_size,
    )
    keys = jrnd.split(key, nature.shape[0] - 1)
    moment_p_rest = jax.vmap(expected_moment_fn)(keys, moment[:-1], u[:-1], c[:-1])
    moment_p = jnp.vstack((moment[0:1], moment_p_rest))

    return nature, moment, moment_p


# NOTE: _bismooth() requires model.backward (a callable reverse dynamics) which is
# not yet implemented on XFADS.  Do not call until backward dynamics are added.
def _bismooth(
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

    noise = approx.canon_to_moment(approx.free_to_canon(model.noise_free))

    natural_to_moment = jax.vmap(approx.natural_to_moment)
    expected_moment_forward = partial(
        expected_predictive_moment,
        f=model.transition,
        noise=noise,
        approx=approx,
        mc_size=mc_size,
    )
    expected_moment_backward = partial(
        expected_predictive_moment,
        f=model.backward,
        noise=noise,
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
