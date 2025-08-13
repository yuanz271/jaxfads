from collections.abc import Callable
from jax import numpy as jnp, random as jrnd
import equinox as eqx
import chex
from jaxtyping import Array
from omegaconf import OmegaConf

from xfads.distributions import DiagMVN
from xfads.dynamics import (
    Dynamics,
    predict_moment,
    sample_expected_moment,
    DiagGaussian,
)
from xfads.nn import make_mlp
from xfads.dynamics import Noise


class Nonlinear(Dynamics):
    noise: Noise = eqx.field(init=False)
    f: Callable = eqx.field(init=False)

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
    noise = DiagGaussian(1, state_dim)

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    eu = jnp.zeros((0,))  # empty eu for this test

    moment = predict_moment(z, u, eu, f, noise, DiagMVN)
    moment = sample_expected_moment(key, moment, u, eu, f, noise, DiagMVN, 10)
    chex.assert_shape(moment, (DiagMVN.param_size(state_dim),))
