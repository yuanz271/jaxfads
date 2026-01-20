"""
Utility functions for XFADS.

This module provides helper functions for JAX operations, particularly
for handling random key management in vectorized computations.

Functions
---------
vmap_with_key
    Wrap a function to vmap with independent random keys per element.
"""

from collections.abc import Callable
from jax import numpy as jnp
from jax import random as jr
from jax import vmap


def vmap_with_key(fun: Callable) -> Callable:
    """
    Wrap a function to vmap with independent random keys per element.

    Takes a function that accepts a `key` keyword argument and returns
    a vectorized version that automatically splits the key for each
    element in the batch.

    Parameters
    ----------
    fun : Callable
        Function with signature `fun(x, *, key) -> y` where `x` is the
        input and `key` is a JAX PRNG key.

    Returns
    -------
    Callable
        Vectorized function with signature `mapped(x, key) -> y` where
        `x` has shape `(N, ...)` and `key` is split into `N` independent
        subkeys.

    Notes
    -----
    This utility simplifies the common pattern of applying a stochastic
    function across a batch while ensuring each element receives an
    independent random stream:

    .. math::

        \\text{mapped}(x_{1:N}, k) = [f(x_1, k_1), ..., f(x_N, k_N)]

    where :math:`k_i = \\text{split}(k, N)[i]`.

    Examples
    --------
    >>> @vmap_with_key
    ... def sample(x, *, key):
    ...     return x + jr.normal(key, x.shape)
    >>> samples = sample(jnp.ones((10, 3)), key=jr.key(0))
    """

    def mapped(x, key):
        n = jnp.size(x, 0)
        keys = jr.split(key, n)
        return vmap(fun)(x, key=keys)

    return mapped
