import chex
from jax import numpy as jnp
from jax import random as jrnd

from jaxfads.util import vmap_with_key


def test_vmap_with_key_shape_and_randomness():
    @vmap_with_key
    def add_noise(x, *, key):
        return x + jrnd.normal(key, x.shape)

    key = jrnd.key(0)
    x = jnp.ones((6, 3))
    out = add_noise(x, key=key)

    chex.assert_shape(out, (6, 3))
    assert jnp.var(out, axis=0).sum() > 0.0
