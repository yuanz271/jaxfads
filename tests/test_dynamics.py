from collections.abc import Callable
from functools import partial as fpartial

import jax
from jax import Array, numpy as jnp, random as jrnd
import chex
from omegaconf import OmegaConf

from jaxfads.base import Dynamics
from jaxfads.dynamics import sample_expected_moment
from jaxfads.nn import make_mlp
from conftest import make_noise_moment


class Nonlinear(Dynamics):
    f: Callable[..., Array]

    def __init__(self, conf, key):
        self.conf = conf
        self.f = make_mlp(
            conf.state_dim + conf.input_dim,
            conf.state_dim,
            conf.width,
            conf.depth,
            key=key,
            final_bias=False,
            dropout=conf.dropout,
        )

    def forward(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return z + self.f(x, key=key)


def _make_nonlinear(spec, key):
    """Build a Nonlinear dynamics model from spec."""
    return Dynamics.get_subclass(Nonlinear.__name__)(
        OmegaConf.create(
            dict(
                state_dim=spec["state_dim"],
                input_dim=spec["input_dim"],
                width=spec["width"],
                depth=spec["depth"],
                cov=1.0,
                dropout=None,
            )
        ),
        key=key,
    )


def test_predict_moment(diag, spec):
    """predict_moment returns correct shape (mean parameter)."""
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    f = _make_nonlinear(spec, key)
    noise_mom = make_noise_moment(diag, state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (spec["input_dim"],))
    loc = f(z, u, jnp.zeros((0,)))

    mean_param = diag.predict_moment(loc, noise_mom)
    chex.assert_shape(mean_param, (diag.param_size(state_dim),))
    chex.assert_tree_all_finite(mean_param)


def test_mean_param_to_moment_roundtrip(diag, spec):
    """predict_moment → mean_param_to_moment gives valid (mean, cov)."""
    state_dim = spec["state_dim"]
    noise_mom = make_noise_moment(diag, state_dim)

    loc = jrnd.normal(jrnd.key(0), (state_dim,))
    mean_param = diag.predict_moment(loc, noise_mom)
    moment = diag.mean_param_to_moment(mean_param)

    mean, cov = diag.moment_to_canon(moment)
    chex.assert_trees_all_close(mean, loc, atol=1e-6)
    _, Q = diag.moment_to_canon(noise_mom)
    chex.assert_trees_all_close(cov, Q, atol=1e-5)


def test_sample_expected_moment(diag, spec):
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    f = _make_nonlinear(spec, key)
    noise_mom = make_noise_moment(diag, state_dim)

    moment = diag.canon_to_moment(
        jrnd.normal(key, (state_dim,)), jnp.ones(state_dim)
    )
    u = jrnd.normal(key, (spec["input_dim"],))

    result = sample_expected_moment(key, moment, u, jnp.zeros((0,)), f, noise_mom, diag, 10)
    chex.assert_shape(result, (diag.param_size(state_dim),))
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


def test_sample_expected_moment_partial_invalid(diag):
    """When some MC samples produce NaN, the output should still be finite."""
    state_dim, mc_size = 4, 64
    key = jrnd.key(42)
    noise_mom = make_noise_moment(diag, state_dim)

    moment = diag.canon_to_moment(jnp.zeros(state_dim), jnp.full(state_dim, 25.0))
    f = fpartial(_threshold_dynamics, radius=3.0)

    result = jax.jit(
        lambda k: sample_expected_moment(
            k, moment, jnp.zeros(0), jnp.zeros(0), f, noise_mom, diag, mc_size
        )
    )(key)

    assert jnp.all(jnp.isfinite(result))
    chex.assert_shape(result, (diag.param_size(state_dim),))


def test_sample_expected_moment_all_invalid(diag):
    """When all MC samples produce NaN, the fallback at z_mean should be used."""
    state_dim, mc_size = 2, 16
    key = jrnd.key(99)
    noise_mom = make_noise_moment(diag, state_dim)

    moment = diag.canon_to_moment(jnp.zeros(state_dim), jnp.full(state_dim, 1.0))
    u, c = jnp.zeros(0), jnp.zeros(0)
    f = fpartial(_threshold_dynamics, radius=0.001)

    result = jax.jit(
        lambda k: sample_expected_moment(k, moment, u, c, f, noise_mom, diag, mc_size)
    )(key)

    assert jnp.all(jnp.isfinite(result))
    chex.assert_shape(result, (diag.param_size(state_dim),))

    z_mean, _ = diag.moment_to_canon(moment)
    loc = f(z_mean, u, c)
    expected = diag.mean_param_to_moment(diag.predict_moment(loc, noise_mom))
    chex.assert_trees_all_close(result, expected, atol=1e-6)
