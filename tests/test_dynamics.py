from jax import numpy as jnp, random as jrnd
import equinox as eqx
import chex
from jaxtyping import Array

from xfads.distributions import DiagMVN
from xfads.dynamics import (
    AbstractDynamics,
    predict_moment,
    sample_expected_moment,
    DiagGaussian,
)
from xfads.nn import make_mlp


class Nonlinear(AbstractDynamics):
    noise: eqx.Module
    f: eqx.Module

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key,
        dropout: float | None = None,
        **kwargs,
    ):
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

    def __call__(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return z + self.f(x, key=key)

    def loss(self):
        return jnp.mean(self.cov())


def test_predict_moment(spec):
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    input_dim = spec["input_dim"]

    f = Nonlinear(state_dim, input_dim, spec["width"], spec["depth"], key=key, cov=1.0)
    noise = DiagGaussian(1, state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    eu = jnp.zeros((0,))  # empty eu for this test

    moment = predict_moment(z, u, eu, f, noise, DiagMVN)
    chex.assert_shape(moment, (DiagMVN.param_size(state_dim),))


def test_sample_expected_moment(spec):
    key = jrnd.key(0)
    state_dim = spec["state_dim"]
    input_dim = spec["input_dim"]

    f = Nonlinear(state_dim, input_dim, spec["width"], spec["depth"], key=key, cov=1.0)
    noise = DiagGaussian(1, state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    eu = jnp.zeros((0,))  # empty eu for this test

    moment = predict_moment(z, u, eu, f, noise, DiagMVN)
    moment = sample_expected_moment(key, moment, u, eu, f, noise, DiagMVN, 10)
    chex.assert_shape(moment, (DiagMVN.param_size(state_dim),))
