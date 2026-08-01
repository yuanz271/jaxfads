from types import SimpleNamespace

import chex
from jax import numpy as jnp
from jax import random as jr

from jaxfads import core
from jaxfads.distributions import MVN
from jaxfads.noise import Noise


class _DummyModel:
    def __init__(self, state_dim: int, mc_size: int = 1):
        self.approx = MVN(dim=state_dim, rank=state_dim)
        self.conf = SimpleNamespace(mc_size=mc_size)
        self.noise = Noise(
            approx=self.approx,
            q_scale=1.0,
            q_prior_fraction=0.1,
            state_dim=state_dim,
            mstep_enabled=False,
            free=self.approx.free_from_kw(scale=1.0),
        )

    def transition(self, z, u, c, *, key=None):
        del u, c, key
        return z

    def prior_natural(self):
        return self.approx.moment_to_natural(
            self.approx.canon_to_moment(
                self.approx.free_to_canon(self.approx.free_from_kw(scale=1.0))
            )
        )


def test_nofilt_shapes_and_finite():
    state_dim, T = 3, 6
    model = _DummyModel(state_dim=state_dim, mc_size=2)

    param_dim = model.approx.param_size()
    alpha = jnp.vstack([model.prior_natural()] * T)
    u = jnp.zeros((T, 0))
    c = jnp.zeros((T, 0))

    nature, moment, moment_p, _transition_stat = core.nofilt(
        model, jr.key(1), jnp.arange(T), alpha, u, c
    )

    for arr in (nature, moment, moment_p):
        chex.assert_shape(arr, (T, param_dim))
        chex.assert_tree_all_finite(arr)

    chex.assert_trees_all_close(nature, alpha, atol=1e-6)
