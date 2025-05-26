from jax import numpy as jnp, random as jrnd
import equinox as eqx
from xfads.nn import VariantBiasLinear


def test_VariantBiasLinear():
    key = jrnd.key(0)
    vblin = eqx.filter_jit(VariantBiasLinear(2, 4, 5, key=key))
    x = jnp.ones((2,))
    vblin(0, x)
