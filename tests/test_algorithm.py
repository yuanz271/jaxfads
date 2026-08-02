"""Algorithm-level tests.

Focus:
- Consistency with the XFADS paper (Eq. 4 / Eq. 12 semantics)
- Mathematical correctness of exp-family conversions
- Numerical stability (nonfinite-safe MC aggregation)
- Basic functionality of filtering/smoothing primitives
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial as fpartial
from types import SimpleNamespace

import chex
import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jr

from jaxfads import core
from jaxfads.base import Dynamics
from jaxfads.core import expected_predictive_moment
from jaxfads.noise import Noise
from jaxfads.vi import elbo

# -----------------------------------------------------------------------------
# Filtering / smoothing shape + finiteness
# -----------------------------------------------------------------------------


class _Identity(Dynamics):
    def __init__(self, state_dim: int):
        self.conf = SimpleNamespace(state_dim=state_dim)

    def eval(self, z, u, c, *, key=None):
        del u, c, key
        return z


class _DummyModel:
    """Minimal model duck-typing XFADS for core.filter / core._bismooth."""

    def __init__(self, approx, state_dim: int, mc_size: int = 1, cov: float = 1.0):
        self.approx = approx
        self.conf = SimpleNamespace(mc_size=mc_size)
        self.transition = _Identity(state_dim)
        self.backward = _Identity(state_dim)
        self.noise = Noise(
            approx=approx,
            free=approx.free_from_kw(scale=cov),
        )

    def prior_natural(self):
        return self.approx.moment_to_natural(
            self.approx.canon_to_moment(
                self.approx.free_to_canon(self.approx.free_from_kw(scale=1.0))
            )
        )


def test_filter_shapes_and_finite(diag):
    state_dim, T = 2, 4
    model = _DummyModel(diag, state_dim)
    key = jr.key(0)
    param_dim = diag.param_size()

    alpha = jnp.zeros((T, param_dim))
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    nature_f, moment_f, moment_p = core.filter(
        model, key, jnp.arange(T), alpha, u, c
    )

    for arr in (nature_f, moment_f, moment_p):
        chex.assert_shape(arr, (T, param_dim))
        chex.assert_tree_all_finite(arr)


def test_bismooth_shapes_and_finite():
    """Pinned to use_sigma_points=False (not the shared `diag` fixture,
    which now defaults to True): _bismooth is documented as not yet
    production-ready (requires real backward dynamics; _DummyModel uses
    Identity for both forward and backward as a placeholder), and
    combining two identical forward/backward filtering passes can produce
    a genuinely negative-definite natural-parameter sum here -- MC's tfd
    sampling tolerates that silently, UT's explicit Cholesky correctly
    does not. Unrelated to transition_points' own correctness; not
    something to fix as part of that work."""
    from jaxfads.distributions import MVN

    state_dim, T = 2, 5
    approx = MVN(dim=state_dim, rank=state_dim, use_sigma_points=False)
    model = _DummyModel(approx, state_dim)
    key = jr.key(1)
    param_dim = approx.param_size()

    alpha = jnp.zeros((T, param_dim))
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    nature_s, moment_s, moment_p = core._bismooth(
        model, key, jnp.arange(T), alpha, u, c
    )

    for arr in (nature_s, moment_s, moment_p):
        chex.assert_shape(arr, (T, param_dim))
        chex.assert_tree_all_finite(arr)


def test_causal_reindexed_identity(diag):
    """Causal mode uses code indexing: lambda_t = check_lambda_t + beta_t."""
    state_dim, T = 2, 6
    model = _DummyModel(diag, state_dim)
    key = jr.key(10)
    param_dim = diag.param_size()

    alpha = jr.normal(jr.key(11), (T, param_dim)) * 0.05
    beta = jr.normal(jr.key(12), (T, param_dim)) * 0.05
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    check_nature, _, _ = core.filter(
        model, key, jnp.arange(T), alpha, u, c
    )
    nature, moment, moment_p = core.causal(
        model, key, jnp.arange(T), alpha, beta, u, c
    )

    chex.assert_trees_all_close(nature, check_nature + beta, atol=1e-6)
    chex.assert_shape(moment, (T, param_dim))
    chex.assert_shape(moment_p, (T, param_dim))
    chex.assert_tree_all_finite(moment)
    chex.assert_tree_all_finite(moment_p)


def test_causal_zero_beta_reduces_to_alpha_filter(diag):
    """If beta is zero, causal matches alpha-only filtering."""
    state_dim, T = 2, 5
    model = _DummyModel(diag, state_dim)
    key = jr.key(20)
    param_dim = diag.param_size()

    alpha = jr.normal(jr.key(21), (T, param_dim)) * 0.05
    beta = jnp.zeros_like(alpha)
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    check_nature, check_moment, check_moment_p = core.filter(
        model, key, jnp.arange(T), alpha, u, c
    )
    nature, moment, moment_p = core.causal(
        model, key, jnp.arange(T), alpha, beta, u, c
    )

    chex.assert_trees_all_close(nature, check_nature, atol=1e-6)
    chex.assert_trees_all_close(moment, check_moment, atol=1e-6)
    chex.assert_trees_all_close(moment_p, check_moment_p, atol=1e-6)


# -----------------------------------------------------------------------------
# Paper semantics: Eq (4) predictive moment and Eq (12) expected predictive moment
# -----------------------------------------------------------------------------


class _Nonlinear(Dynamics):
    """Small nonlinear dynamics used only for algorithm tests."""

    f: Callable[..., Array]

    def __init__(self, conf, key):
        from jaxfads import nn

        self.conf = conf
        self.f = nn.make_mlp(
            conf.state_dim + conf.input_dim,
            conf.state_dim,
            conf.width,
            conf.depth,
            key=key,
            final_bias=False,
            dropout=conf.dropout,
        )

    def eval(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return z + self.f(x, key=key)


def _make_nonlinear(spec, key):
    return Dynamics.get_subclass(_Nonlinear.__name__)(
        SimpleNamespace(
            state_dim=spec["state_dim"],
            input_dim=spec["input_dim"],
            width=spec["width"],
            depth=spec["depth"],
            dropout=None,
        ),
        key=key,
    )


def test_predictive_moment_is_conditional_moment(diag, spec):
    """Eq (4): predictive_moment(mu, noise) encodes E[T(z_t) | z_{t-1}]."""
    key = jr.key(0)
    state_dim = spec["state_dim"]

    f = _make_nonlinear(spec, key)

    # Transition noise in moment form
    noise = diag.canon_to_moment(diag.free_to_canon(diag.free_from_kw(scale=1.0)))

    z = jr.normal(key, (state_dim,))
    u = jr.normal(key, (spec["input_dim"],))
    mu = f(z, u, jnp.zeros((0,)))

    stats = diag.predictive_moment(mu, noise)
    chex.assert_shape(stats, (diag.param_size(),))
    chex.assert_tree_all_finite(stats)

    mean, cov = diag.unpack(stats)
    chex.assert_trees_all_close(mean, mu, atol=1e-6)
    _, Q = diag.unpack(noise)
    chex.assert_trees_all_close(cov, Q, atol=1e-5)


def test_expected_predictive_moment_is_finite(diag, spec):
    """Eq (12): expected predictive moment is finite for a well-behaved model."""
    key = jr.key(0)
    state_dim = spec["state_dim"]

    f = _make_nonlinear(spec, key)
    noise = diag.canon_to_moment(diag.free_to_canon(diag.free_from_kw(scale=1.0)))

    mp = diag.pack(jr.normal(key, (state_dim,)), jnp.eye(state_dim))
    u = jr.normal(key, (spec["input_dim"],))

    result = expected_predictive_moment(
        key,
        mp,
        u,
        jnp.zeros((0,)),
        f,
        noise,
        diag,
        mc_size=10,
    )
    chex.assert_shape(result, (diag.param_size(),))
    chex.assert_tree_all_finite(result)


# -----------------------------------------------------------------------------
# Numerical stability: nonfinite-safe MC aggregation
# -----------------------------------------------------------------------------


def _threshold_dynamics(
    z: Array, u: Array, c: Array, *, key: Array | None = None, radius: float = 3.0
) -> Array:
    """Dynamics that return NaN when any |z_i| > radius."""
    del u, c, key
    safe = jnp.all(jnp.abs(z) <= radius)
    return jnp.where(safe, z * 0.9, jnp.full_like(z, jnp.nan))


def test_expected_predictive_moment_partial_invalid():
    """When some MC samples produce NaN, the output should still be finite."""
    from jaxfads.distributions import MVN

    state_dim, mc_size = 4, 64
    key = jr.key(42)

    # Pinned to use_sigma_points=False: this test is specifically about MC
    # sample masking (see docstring); UT is covered separately by
    # test_expected_predictive_moment_unscented_partial_invalid below.
    approx = MVN(dim=state_dim, rank=state_dim, use_sigma_points=False)
    noise = approx.canon_to_moment(approx.free_to_canon(approx.free_from_kw(scale=1.0)))

    mp = approx.pack(jnp.zeros(state_dim), 25.0 * jnp.eye(state_dim))
    f = fpartial(_threshold_dynamics, radius=3.0)

    result = jax.jit(
        lambda k: expected_predictive_moment(
            k, mp, jnp.zeros(0), jnp.zeros(0), f, noise, approx, mc_size
        )
    )(key)

    assert jnp.all(jnp.isfinite(result))
    chex.assert_shape(result, (approx.param_size(),))


def test_expected_predictive_moment_all_invalid(diag):
    """When all MC samples produce NaN, the result should be non-finite."""
    state_dim, mc_size = 2, 16
    key = jr.key(99)

    noise = diag.canon_to_moment(diag.free_to_canon(diag.free_from_kw(scale=1.0)))

    mp = diag.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    u, c = jnp.zeros(0), jnp.zeros(0)
    f = fpartial(_threshold_dynamics, radius=0.001)

    result = jax.jit(
        lambda k: expected_predictive_moment(k, mp, u, c, f, noise, diag, mc_size)
    )(key)

    chex.assert_shape(result, (diag.param_size(),))
    assert not jnp.all(jnp.isfinite(result))


# -----------------------------------------------------------------------------
# transition_points: unscented sigma points vs. plain MC
# -----------------------------------------------------------------------------


def test_transition_points_unscented_exact_for_linear_transition():
    """Discriminative test for docs/transition_points.md: for a linear
    transition, the unscented transform's predicted mean/covariance
    exactly matches the true linear-Gaussian pushforward
    (mean_true = A @ mean + b, cov_true = A @ cov @ A.T + Q); plain MC at
    a matched point count does not (residual sampling error)."""
    from jaxfads.distributions import MVN

    state_dim = 3
    key = jr.key(7)

    approx_ut = MVN(dim=state_dim, rank=state_dim, use_sigma_points=True)
    approx_mc = MVN(dim=state_dim, rank=state_dim, use_sigma_points=False)

    A = 0.9 * jnp.eye(state_dim) + 0.05 * jr.normal(jr.key(1), (state_dim, state_dim))
    b = jr.normal(jr.key(2), (state_dim,))

    def f(z, u, c, *, key=None):
        del u, c, key
        return A @ z + b

    mean = jr.normal(jr.key(3), (state_dim,))
    cov_free = jr.normal(jr.key(4), (state_dim, state_dim))
    cov = cov_free @ cov_free.T + 0.5 * jnp.eye(state_dim)
    mp = approx_ut.pack(mean, cov)

    noise = approx_ut.canon_to_moment(
        approx_ut.free_to_canon(approx_ut.free_from_kw(scale=0.3))
    )
    _, Q = approx_ut.unpack(noise)

    mean_true = A @ mean + b
    cov_true = A @ cov @ A.T + Q

    u, c = jnp.zeros(0), jnp.zeros(0)
    mc_size_matched = 2 * state_dim + 1  # same f-evaluation count as UT

    result_ut = expected_predictive_moment(
        key, mp, u, c, f, noise, approx_ut, mc_size_matched
    )
    mean_ut, cov_ut = approx_ut.unpack(result_ut)
    chex.assert_trees_all_close(mean_ut, mean_true, atol=1e-4)
    chex.assert_trees_all_close(cov_ut, cov_true, atol=1e-4)

    result_mc = expected_predictive_moment(
        key, mp, u, c, f, noise, approx_mc, mc_size_matched
    )
    _, cov_mc = approx_mc.unpack(result_mc)
    # MC at the same point count has residual sampling error -- it should
    # not land anywhere near UT's tolerance for the same comparison.
    assert not jnp.allclose(cov_mc, cov_true, atol=1e-4)


def _coord_threshold_dynamics(
    z: Array,
    u: Array,
    c: Array,
    *,
    key: Array | None = None,
    coord: int = 0,
    threshold: float = 1.0,
) -> Array:
    """Dynamics that return NaN when z[coord] > threshold."""
    del u, c, key
    invalid = z[coord] > threshold
    return jnp.where(invalid, jnp.full_like(z, jnp.nan), z * 0.9)


def test_expected_predictive_moment_unscented_partial_invalid():
    """When exactly one of the 2*dim+1 deterministic sigma points produces
    NaN, the weighted-masking reduction should still degrade correctly
    (finite output) at this coarser, deterministic point count."""
    from jaxfads.distributions import MVN

    state_dim = 2
    key = jr.key(11)

    approx = MVN(dim=state_dim, rank=state_dim, use_sigma_points=True)
    noise = approx.canon_to_moment(approx.free_to_canon(approx.free_from_kw(scale=1.0)))
    mp = approx.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    u, c = jnp.zeros(0), jnp.zeros(0)

    # ut_alpha=1.0, ut_kappa=0.0 defaults -> c=dim=2, so the sigma points at
    # mean=0, cov=I are (0,0), (+-sqrt(2),0), (0,+-sqrt(2)) -- exactly one
    # of them (+sqrt(2), 0) has coord-0 > 1.0.
    f = fpartial(_coord_threshold_dynamics, coord=0, threshold=1.0)

    result = expected_predictive_moment(key, mp, u, c, f, noise, approx, mc_size=4)
    chex.assert_shape(result, (approx.param_size(),))
    assert jnp.all(jnp.isfinite(result))


# -----------------------------------------------------------------------------
# ELBO correctness
# -----------------------------------------------------------------------------


def _eloglik_stub(key, t, mp, y, approx, mc_size):
    del key, t, mp, approx, mc_size
    return jnp.sum(y)


def test_elbo_matches_manual(diag):
    mean = jnp.array([0.2, -0.1])
    cov = jnp.diag(jnp.array([1.0, 2.0]))
    mp = diag.pack(mean, cov)
    mp_p = diag.pack(jnp.zeros(2), jnp.eye(2))
    y = jnp.array([1.0, 2.0])

    expected = _eloglik_stub(None, None, None, y, None, None) - diag.kl(mp, mp_p)
    value = elbo(
        jr.key(0),
        jnp.array(0),
        mp,
        mp_p,
        y,
        _eloglik_stub,
        diag,
        mc_size=1,
    )

    chex.assert_trees_all_close(value, expected)
