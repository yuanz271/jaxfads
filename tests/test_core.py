from types import SimpleNamespace

import chex
from jax import numpy as jnp
from jax import random as jrnd

from jaxfads import core
from jaxfads.distributions import MVN
from jaxfads.base import Dynamics

approx = MVN(rank=0)


class IdentityDynamics(Dynamics):

    def __init__(self, state_dim: int):
        self.conf = SimpleNamespace(state_dim=state_dim)

    def forward(self, z, u, c, *, key=None):
        del u, c, key
        return z


class DummyModel:
    """Minimal model duck-typing XFADS for core.filter / core.bismooth."""

    def __init__(self, state_dim: int, mc_size: int = 1, cov: float = 1.0):
        self.approx = approx
        self.conf = SimpleNamespace(mc_size=mc_size)
        self.forward = IdentityDynamics(state_dim)
        self.backward = IdentityDynamics(state_dim)
        self._state_dim = state_dim
        self._unconstrained_noise = approx.init_noise(cov, state_dim)

    def prior_natural(self):
        return approx.prior_natural(self._state_dim)

    def noise_moment(self):
        return approx.constrain_moment(self._unconstrained_noise)


def test_filter_shapes_and_finite():
    state_dim = 2
    T = 4
    model = DummyModel(state_dim)
    key = jrnd.key(0)
    param_dim = approx.param_size(state_dim)

    alpha = jnp.zeros((T, param_dim))
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    nature_f, moment_f, moment_p = core.filter(model, key, jnp.arange(T), alpha, u, c)

    chex.assert_shape(nature_f, (T, param_dim))
    chex.assert_shape(moment_f, (T, param_dim))
    chex.assert_shape(moment_p, (T, param_dim))
    chex.assert_tree_all_finite(nature_f)
    chex.assert_tree_all_finite(moment_f)
    chex.assert_tree_all_finite(moment_p)


def test_bismooth_shapes_and_finite():
    state_dim = 2
    T = 5
    model = DummyModel(state_dim)
    key = jrnd.key(1)
    param_dim = approx.param_size(state_dim)

    alpha = jnp.zeros((T, param_dim))
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    nature_s, moment_s, moment_p = core.bismooth(model, key, jnp.arange(T), alpha, u, c)

    chex.assert_shape(nature_s, (T, param_dim))
    chex.assert_shape(moment_s, (T, param_dim))
    chex.assert_shape(moment_p, (T, param_dim))
    chex.assert_tree_all_finite(nature_s)
    chex.assert_tree_all_finite(moment_s)
    chex.assert_tree_all_finite(moment_p)
