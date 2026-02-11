from collections.abc import Callable
from functools import partial as fpartial

import jax
from jax import Array, numpy as jnp, random as jrnd
import chex
from omegaconf import OmegaConf

from jaxfads.distributions import DiagMVN
from jaxfads.dynamics import (
    Dynamics,
    predict_moment,
    sample_expected_moment,
    DiagGaussian,
)
from jaxfads.nn import make_mlp
from jaxfads.dynamics import Noise


class Nonlinear(Dynamics):
    noise: Noise
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
        cov = self.conf.cov
        dropout = self.conf.dropout

        self.noise = DiagGaussian(cov, state_dim)
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

    def loss(self):
        return jnp.mean(self.cov())


def test_predict_moment(spec):
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
    noise = DiagGaussian(jnp.array(1.0), state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    eu = jnp.zeros((0,))  # empty eu for this test

    moment = predict_moment(z, u, eu, f, noise, DiagMVN)
    chex.assert_shape(moment, (DiagMVN.param_size(state_dim),))


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
    noise = DiagGaussian(jnp.array(1.0), state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    eu = jnp.zeros((0,))  # empty eu for this test

    moment = predict_moment(z, u, eu, f, noise, DiagMVN)
    moment = sample_expected_moment(key, moment, u, eu, f, noise, DiagMVN, 10)
    chex.assert_shape(moment, (DiagMVN.param_size(state_dim),))


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
    """When some MC samples produce NaN, the output should still be finite.

    We use a wide posterior (cov=25) so many samples exceed radius=3.0
    and trigger NaN, but the mean is at 0 so some samples remain valid.
    The masked mean over valid samples should be all-finite.
    """
    state_dim = 4
    mc_size = 64
    key = jrnd.key(42)

    noise = DiagGaussian(jnp.array(1.0), state_dim)

    # Wide posterior: N(0, 25I) — many samples will exceed |z|>3
    moment = DiagMVN.canon_to_moment(jnp.zeros(state_dim), jnp.full(state_dim, 25.0))
    u = jnp.zeros(0)
    c = jnp.zeros(0)

    f = fpartial(_threshold_dynamics, radius=3.0)

    result = jax.jit(
        lambda k: sample_expected_moment(k, moment, u, c, f, noise, DiagMVN, mc_size)
    )(key)

    assert jnp.all(jnp.isfinite(result)), (
        f"Expected all-finite output with partial-invalid MC samples, "
        f"got {jnp.sum(~jnp.isfinite(result))} non-finite entries"
    )
    chex.assert_shape(result, (DiagMVN.param_size(state_dim),))


def test_sample_expected_moment_all_invalid():
    """When all MC samples produce NaN, the fallback at z_mean should be used.

    Uses threshold dynamics with radius=0.001 and posterior N(0, I).
    All MC samples exceed the threshold, but z_mean=0 is within radius,
    so the deterministic fallback at z_mean produces finite output.
    """
    state_dim = 2
    mc_size = 16
    key = jrnd.key(99)

    noise = DiagGaussian(jnp.array(1.0), state_dim)

    moment = DiagMVN.canon_to_moment(jnp.zeros(state_dim), jnp.full(state_dim, 1.0))
    u = jnp.zeros(0)
    c = jnp.zeros(0)

    f = fpartial(_threshold_dynamics, radius=0.001)

    result = jax.jit(
        lambda k: sample_expected_moment(k, moment, u, c, f, noise, DiagMVN, mc_size)
    )(key)

    # The posterior mean z_mean = 0 is within radius=0.001, so fallback
    # predict_moment(z_mean=0, ...) should produce finite output.
    assert jnp.all(jnp.isfinite(result)), (
        f"Expected all-finite fallback output when all MC samples are invalid, "
        f"got {jnp.sum(~jnp.isfinite(result))} non-finite entries"
    )
    chex.assert_shape(result, (DiagMVN.param_size(state_dim),))

    # Verify it equals the deterministic prediction at z_mean
    z_mean, _ = DiagMVN.moment_to_canon(moment)
    expected = predict_moment(z_mean, u, c, f, noise, DiagMVN)
    chex.assert_trees_all_close(result, expected, atol=1e-6)
