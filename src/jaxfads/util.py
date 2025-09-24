from collections.abc import Callable
from jax import numpy as jnp
from jax import random as jr
from jax import vmap


def vmap_with_key(fun: Callable) -> Callable:
    def mapped(x, key):
        n = jnp.size(x, 0)
        keys = jr.split(key, n)
        return vmap(fun)(x, key=keys)

    return mapped
