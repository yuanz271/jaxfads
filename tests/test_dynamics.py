from collections.abc import Callable
from functools import partial as fpartial

import jax
from jax import Array, numpy as jnp, random as jrnd
import chex
from omegaconf import OmegaConf

from jaxfads.base import Dynamics
from jaxfads.dynamics import sample_expected_mean
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


def test_predict_mean(diag, spec):
    """predict_mean returns correct shape and recovers (loc, Q)."""
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    f = _make_nonlinear(spec, key)
    noise = make_noise(diag, state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (spec["input_dim"],))
    loc = f(z, u, jnp.zeros((0,)))

    # Single loc: predict_mean → from_sufficient_stats recovers (loc, Q_diag)
    expanded = diag.predict_mean(loc, noise)
    mp = diag.from_sufficient_stats(expanded)
    chex.assert_shape(mp, (diag.mean_size(state_dim),))
    chex.assert_tree_all_finite(mp)

    mean, cov = diag.unpack(mp)
    chex.assert_trees_all_close(mean, loc, atol=1e-6)
    _, Q = diag.unpack(noise)
    chex.assert_trees_all_close(cov, Q, atol=1e-5)


def test_sample_expected_mean(diag, spec):
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    f = _make_nonlinear(spec, key)
    noise = make_noise(diag, state_dim)

    mp = diag.pack(
        jrnd.normal(key, (state_dim,)), jnp.ones(state_dim)
    )
    u = jrnd.normal(key, (spec["input_dim"],))

    result = sample_expected_mean(key, mp, u, jnp.zeros((0,)), f, noise, diag, 10)
    chex.assert_shape(result, (diag.mean_size(state_dim),))
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


def test_sample_expected_mean_partial_invalid():
    """When some MC samples produce NaN, the output should still be finite."""
    state_dim, mc_size = 4, 64
    key = jrnd.key(42)
    from jaxfads.distributions import MVN
    diag4 = MVN(dim=state_dim, rank=0)
    noise = make_noise(diag4, state_dim)

    mp = diag4.pack(jnp.zeros(state_dim), jnp.full(state_dim, 25.0))
    f = fpartial(_threshold_dynamics, radius=3.0)

    result = jax.jit(
        lambda k: sample_expected_mean(
            k, mp, jnp.zeros(0), jnp.zeros(0), f, noise, diag4, mc_size
        )
    )(key)

    assert jnp.all(jnp.isfinite(result))
    chex.assert_shape(result, (diag4.mean_size(state_dim),))


def test_sample_expected_mean_all_invalid(diag):
    """When all MC samples produce NaN, the result should be non-finite."""
    state_dim, mc_size = 2, 16
    key = jrnd.key(99)
    noise = make_noise(diag, state_dim)

    mp = diag.pack(jnp.zeros(state_dim), jnp.full(state_dim, 1.0))
    u, c = jnp.zeros(0), jnp.zeros(0)
    f = fpartial(_threshold_dynamics, radius=0.001)

    result = jax.jit(
        lambda k: sample_expected_mean(k, mp, u, c, f, noise, diag, mc_size)
    )(key)

    chex.assert_shape(result, (diag.mean_size(state_dim),))
    assert not jnp.all(jnp.isfinite(result))
