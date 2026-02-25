from collections.abc import Callable
from functools import partial as fpartial

import jax
from jax import Array, numpy as jnp, random as jrnd
import chex
from omegaconf import OmegaConf

from jaxfads.base import Dynamics
from jaxfads.core import expected_predictive_moment
from jaxfads.nn import make_mlp
from conftest import make_noise


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


def test_predictive_moment(diag, spec):
    """predictive_moment returns the conditional exp-family moment."""
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    f = _make_nonlinear(spec, key)
    noise = make_noise(diag, state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (spec["input_dim"],))
    mu = f(z, u, jnp.zeros((0,)))

    stats = diag.predictive_moment(mu, noise)
    chex.assert_shape(stats, (state_dim + state_dim * state_dim,))
    chex.assert_tree_all_finite(stats)

    mean, cov = diag.unpack(stats)
    chex.assert_trees_all_close(mean, mu, atol=1e-6)
    _, Q = diag.unpack(noise)
    chex.assert_trees_all_close(cov, Q, atol=1e-5)


def test_expected_predictive_moment(diag, spec):
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    f = _make_nonlinear(spec, key)
    noise = make_noise(diag, state_dim)

    mp = diag.pack(
        jrnd.normal(key, (state_dim,)), jnp.eye(state_dim)
    )
    u = jrnd.normal(key, (spec["input_dim"],))

    result = expected_predictive_moment(key, mp, u, jnp.zeros((0,)), f, noise, diag, 10)
    chex.assert_shape(result, (state_dim + state_dim * state_dim,))
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


def test_expected_predictive_moment_partial_invalid():
    """When some MC samples produce NaN, the output should still be finite."""
    state_dim, mc_size = 4, 64
    key = jrnd.key(42)
    from jaxfads.distributions import MVN
    diag4 = MVN(dim=state_dim)
    noise = make_noise(diag4, state_dim)

    mp = diag4.pack(jnp.zeros(state_dim), 25.0 * jnp.eye(state_dim))
    f = fpartial(_threshold_dynamics, radius=3.0)

    result = jax.jit(
        lambda k: expected_predictive_moment(
            k, mp, jnp.zeros(0), jnp.zeros(0), f, noise, diag4, mc_size
        )
    )(key)

    assert jnp.all(jnp.isfinite(result))
    chex.assert_shape(result, (state_dim + state_dim * state_dim,))


def test_expected_predictive_moment_all_invalid(diag):
    """When all MC samples produce NaN, the result should be non-finite."""
    state_dim, mc_size = 2, 16
    key = jrnd.key(99)
    noise = make_noise(diag, state_dim)

    mp = diag.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    u, c = jnp.zeros(0), jnp.zeros(0)
    f = fpartial(_threshold_dynamics, radius=0.001)

    result = jax.jit(
        lambda k: expected_predictive_moment(k, mp, u, c, f, noise, diag, mc_size)
    )(key)

    chex.assert_shape(result, (state_dim + state_dim * state_dim,))
    assert not jnp.all(jnp.isfinite(result))
