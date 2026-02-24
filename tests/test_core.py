from types import SimpleNamespace

import chex
from jax import numpy as jnp
from jax import random as jrnd

from jaxfads import core
from jaxfads.base import Dynamics


class IdentityDynamics(Dynamics):

    def __init__(self, state_dim: int):
        self.conf = SimpleNamespace(state_dim=state_dim)

    def forward(self, z, u, c, *, key=None):
        del u, c, key
        return z


class DummyModel:
    """Minimal model duck-typing XFADS for core.filter / core.bismooth."""

    def __init__(self, approx, state_dim: int, mc_size: int = 1, cov: float = 1.0):
        self.approx = approx
        self.conf = SimpleNamespace(mc_size=mc_size)
        self.forward = IdentityDynamics(state_dim)
        self.backward = IdentityDynamics(state_dim)
        self._state_dim = state_dim
        self._unconstrained_noise = approx.init_noise(cov, state_dim)

    def prior_natural(self):
        return self.approx.prior_natural(self._state_dim)

    def noise_mean(self):
        return self.approx.constrain_mean(self._unconstrained_noise)


def test_filter_shapes_and_finite(diag):
    state_dim, T = 2, 4
    model = DummyModel(diag, state_dim)
    key = jrnd.key(0)
    param_dim = diag.param_size(state_dim)

    alpha = jnp.zeros((T, param_dim))
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    nature_f, moment_f, moment_p = core.filter(model, key, jnp.arange(T), alpha, u, c)

    for arr in (nature_f, moment_f, moment_p):
        chex.assert_shape(arr, (T, param_dim))
        chex.assert_tree_all_finite(arr)


def test_bismooth_shapes_and_finite(diag):
    state_dim, T = 2, 5
    model = DummyModel(diag, state_dim)
    key = jrnd.key(1)
    param_dim = diag.param_size(state_dim)

    alpha = jnp.zeros((T, param_dim))
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    nature_s, moment_s, moment_p = core.bismooth(model, key, jnp.arange(T), alpha, u, c)

    for arr in (nature_s, moment_s, moment_p):
        chex.assert_shape(arr, (T, param_dim))
        chex.assert_tree_all_finite(arr)
