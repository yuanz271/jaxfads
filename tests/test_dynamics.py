from collections.abc import Callable
from functools import partial as fpartial

import jax
from jax import Array, numpy as jnp, random as jrnd
import chex
from omegaconf import OmegaConf

from jaxfads.distributions import DiagMVN
from jaxfads.base import Dynamics
from jaxfads.dynamics import sample_expected_moment
from jaxfads.nn import make_mlp


class Nonlinear(Dynamics):
    f: Callable[..., Array]

    def __init__(
        self,
        conf,
        key,
    ):
        self.conf = conf
        state_dim = self.conf.state_dim
        input_dim = self.conf.input_dim
        width = self.conf.width
        depth = self.conf.depth
        dropout = self.conf.dropout

        self.f = make_mlp(
            state_dim + input_dim,
            state_dim,
            width,
            depth,
            key=key,
            final_bias=False,
            dropout=dropout,
        )

    def forward(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return z + self.f(x, key=key)


def _make_noise_moment(state_dim, cov=1.0):
    """Helper: create a constrained noise moment array for test use."""
    return DiagMVN.constrain_moment(DiagMVN.init_noise(cov, state_dim))


def test_predict_moment(spec):
    """Test Approx.predict_moment returns correct shape (mean parameter)."""
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    input_dim = spec["input_dim"]

    f = Dynamics.get_subclass(Nonlinear.__name__)(
        OmegaConf.create(
            dict(
                state_dim=state_dim,
                input_dim=input_dim,
                width=spec["width"],
                depth=spec["depth"],
                cov=1.0,
                dropout=None,
            )
        ),
        key=key,
    )
    noise_mom = _make_noise_moment(state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    eu = jnp.zeros((0,))

    loc = f(z, u, eu)
    mean_param = DiagMVN.predict_moment(loc, noise_mom)
    chex.assert_shape(mean_param, (DiagMVN.param_size(state_dim),))
    chex.assert_tree_all_finite(mean_param)


def test_mean_param_to_moment_roundtrip(spec):
    """predict_moment → mean_param_to_moment gives valid (mean, cov)."""
    state_dim = spec["state_dim"]
    noise_mom = _make_noise_moment(state_dim)

    loc = jrnd.normal(jrnd.key(0), (state_dim,))
    mean_param = DiagMVN.predict_moment(loc, noise_mom)
    moment = DiagMVN.mean_param_to_moment(mean_param)

    mean, cov = DiagMVN.moment_to_canon(moment)
    chex.assert_trees_all_close(mean, loc, atol=1e-6)
    # cov should be the noise covariance (since Var[f] = 0 for a single point)
    _, Q = DiagMVN.moment_to_canon(noise_mom)
    chex.assert_trees_all_close(cov, Q, atol=1e-5)


def test_sample_expected_moment(spec):
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    input_dim = spec["input_dim"]

    f = Dynamics.get_subclass(Nonlinear.__name__)(
        OmegaConf.create(
            dict(
                state_dim=state_dim,
                input_dim=input_dim,
                width=spec["width"],
                depth=spec["depth"],
                cov=1.0,
                dropout=None,
            )
        ),
        key=key,
    )
    noise_mom = _make_noise_moment(state_dim)

    # Start from a proper (mean, cov) moment for sampling
    moment = DiagMVN.canon_to_moment(
        jrnd.normal(key, (state_dim,)), jnp.ones(state_dim)
    )
    u = jrnd.normal(key, (input_dim,))
    eu = jnp.zeros((0,))

    result = sample_expected_moment(key, moment, u, eu, f, noise_mom, DiagMVN, 10)
    chex.assert_shape(result, (DiagMVN.param_size(state_dim),))
    chex.assert_tree_all_finite(result)


# ---------------------------------------------------------------------------
# Nonfinite-safe MC aggregation tests
# ---------------------------------------------------------------------------


def _threshold_dynamics(
    z: Array, u: Array, c: Array, *, key: Array | None = None, radius: float = 3.0
) -> Array:
    """Dynamics that return NaN when any |z_i| > radius."""
    safe = jnp.all(jnp.abs(z) <= radius)
    return jnp.where(safe, z * 0.9, jnp.full_like(z, jnp.nan))


def test_sample_expected_moment_partial_invalid():
    """When some MC samples produce NaN, the output should still be finite."""
    state_dim = 4
    mc_size = 64
    key = jrnd.key(42)

    noise_mom = _make_noise_moment(state_dim)

    # Wide posterior: N(0, 25I) — many samples will exceed |z|>3
    moment = DiagMVN.canon_to_moment(jnp.zeros(state_dim), jnp.full(state_dim, 25.0))
    u = jnp.zeros(0)
    c = jnp.zeros(0)

    f = fpartial(_threshold_dynamics, radius=3.0)

    result = jax.jit(
        lambda k: sample_expected_moment(k, moment, u, c, f, noise_mom, DiagMVN, mc_size)
    )(key)

    assert jnp.all(jnp.isfinite(result)), (
        f"Expected all-finite output with partial-invalid MC samples, "
        f"got {jnp.sum(~jnp.isfinite(result))} non-finite entries"
    )
    chex.assert_shape(result, (DiagMVN.param_size(state_dim),))


def test_sample_expected_moment_all_invalid():
    """When all MC samples produce NaN, the fallback at z_mean should be used."""
    state_dim = 2
    mc_size = 16
    key = jrnd.key(99)

    noise_mom = _make_noise_moment(state_dim)

    moment = DiagMVN.canon_to_moment(jnp.zeros(state_dim), jnp.full(state_dim, 1.0))
    u = jnp.zeros(0)
    c = jnp.zeros(0)

    f = fpartial(_threshold_dynamics, radius=0.001)

    result = jax.jit(
        lambda k: sample_expected_moment(k, moment, u, c, f, noise_mom, DiagMVN, mc_size)
    )(key)

    assert jnp.all(jnp.isfinite(result)), (
        f"Expected all-finite fallback output when all MC samples are invalid, "
        f"got {jnp.sum(~jnp.isfinite(result))} non-finite entries"
    )
    chex.assert_shape(result, (DiagMVN.param_size(state_dim),))

    # Verify it equals the deterministic prediction at z_mean
    # Fallback: predict_moment at z_mean, converted to code's moment format
    z_mean, _ = DiagMVN.moment_to_canon(moment)
    loc = f(z_mean, u, c)
    expected = DiagMVN.mean_param_to_moment(DiagMVN.predict_moment(loc, noise_mom))
    chex.assert_trees_all_close(result, expected, atol=1e-6)
