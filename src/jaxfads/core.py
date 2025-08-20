"""
Core filtering and smoothing algorithms for XFADS.

This module implements the fundamental algorithms for XFADS,
including forward filtering and bidirectional smoothing
using variational inference in exponential family approximations.

Functions
---------
filter
    Forward filtering for state estimation using variational inference.
bismooth
    Bidirectional filtering for improved state smoothing.

Classes
-------
Mode
    Enumeration of inference modes for XFADS.
"""

from enum import auto, StrEnum
from functools import partial

import jax
from jax import Array, numpy as jnp, random as jrnd
from jax.lax import scan

from .dynamics import sample_expected_moment


class Mode(StrEnum):
    """
    Enumeration of inference modes for XFADS.

    Attributes
    ----------
    PSEUDO : str
        Pseudo-observation mode using forward filtering only.
    BIFILTER : str
        Bidirectional filtering mode for improved smoothing.
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
    _, (moment_p, _, nature_f) = scan(
        partial(ff, expected_moment=expected_moment_forward),
        init=(ky, nature_f_1),
        xs=(alpha[1:], u[:-1], c[:-1]),  # t = 2 ... T+1
    )
    nature_f = jnp.vstack((nature_f_1, nature_f))  # 1...T

    moment_f = jax.vmap(approx.natural_to_moment)(nature_f)
    moment_p = jnp.vstack(
        (approx.natural_to_moment(nature_f_1), moment_p)
    )  # prediction of t=1 is the prior

    return nature_f, moment_f, moment_p


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
