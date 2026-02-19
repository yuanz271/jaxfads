from jax import numpy as jnp, random as jrnd
import equinox as eqx
from jaxfads.nn import StationaryLinear, VariantBiasLinear


def test_VariantBiasLinear():
    key = jrnd.key(0)
    vblin = eqx.filter_jit(VariantBiasLinear(2, 4, 5, key=key))
    x = jnp.ones((2,))
    vblin(0, x)


def test_StationaryLinear():
    key = jrnd.key(1)
    lin = StationaryLinear(2, 4, key=key)
    x = jnp.ones((2,))
    out = lin(0, x)
    assert out.shape == (4,)
    assert lin.weight.shape == (4, 2)
