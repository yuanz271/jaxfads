import chex
import jax
from jax import numpy as jnp

from jaxfads.constraints import (
    constrain_positive,
    softplus_inverse,
    unconstrain_positive,
)


def test_constrain_positive_roundtrip():
    x = jnp.array([-2.0, -0.5, 0.0, 1.5])
    constrained = constrain_positive(x)
    recovered = unconstrain_positive(constrained)
    chex.assert_trees_all_close(recovered, x, atol=1e-5)


def test_constrain_positive_monotonic_nonnegative():
    x = jnp.linspace(0.0, 3.0, 16)
    constrained = constrain_positive(x)
    diffs = jnp.diff(constrained)
    assert jnp.all(diffs >= 0.0)


def test_softplus_inverse_roundtrip():
    x = jnp.array([0.1, 0.5, 1.0, 3.0])
    y = softplus_inverse(x)
    recon = jax.nn.softplus(y)
    chex.assert_trees_all_close(recon, x, atol=1e-6, rtol=1e-6)
