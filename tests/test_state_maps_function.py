from __future__ import annotations

import sys
import types

import chex
import jax
from jax import numpy as jnp
from jax import random as jr
from omegaconf import OmegaConf

from jaxfads.state_maps import FunctionStateMap


def plain_fn(z, u, c):
    return z + u - c


def key_fn(z, u, c, *, key=None, gain: float = 2.0):
    del key
    return z + gain * u + c


class CallableObject:
    def __call__(self, z, u, c):
        return z + u + c


_TEST_FN_MODULE = "tests_state_map_fns"
_mod = types.ModuleType(_TEST_FN_MODULE)
_mod.plain_fn = plain_fn
_mod.key_fn = key_fn
_mod.CallableObject = CallableObject
sys.modules[_TEST_FN_MODULE] = _mod


def test_function_state_map_accepts_plain_function():
    conf = OmegaConf.create(
        dict(system_type="discrete", fn_path=f"{_TEST_FN_MODULE}:plain_fn")
    )
    sm = FunctionStateMap(conf, key=jr.key(0))

    z = jnp.array([1.0, 2.0])
    u = jnp.array([0.5, 0.5])
    c = jnp.array([0.1, 0.2])
    out = sm.eval(z, u, c)
    chex.assert_trees_all_close(out, z + u - c, atol=1e-7)


def test_function_state_map_accepts_partial_and_key_callable():
    conf = OmegaConf.create(
        dict(
            system_type="continuous",
            fn_path=f"{_TEST_FN_MODULE}:key_fn",
            fn_kwargs=dict(gain=2.0),
        )
    )
    sm = FunctionStateMap(conf, key=jr.key(0))

    z = jnp.array([1.0, 2.0])
    u = jnp.array([0.5, 0.5])
    c = jnp.array([0.1, 0.2])
    out = sm.eval(z, u, c, key=jr.key(1))
    chex.assert_trees_all_close(out, z + 2.0 * u + c, atol=1e-7)


def test_function_state_map_rejects_callable_object():
    conf = OmegaConf.create(
        dict(
            system_type="discrete",
            fn_path=f"{_TEST_FN_MODULE}:CallableObject",
        )
    )
    try:
        FunctionStateMap(conf, key=jr.key(0))
        raise AssertionError("Expected callable object rejection.")
    except TypeError:
        pass


def test_function_state_map_jit_smoke():
    conf = OmegaConf.create(
        dict(system_type="discrete", fn_path=f"{_TEST_FN_MODULE}:plain_fn")
    )
    sm = FunctionStateMap(conf, key=jr.key(0))
    z = jnp.array([1.0, 2.0])
    u = jnp.array([0.5, 0.5])
    c = jnp.array([0.1, 0.2])
    out = jax.jit(lambda zz, uu, cc: sm.eval(zz, uu, cc))(z, u, c)
    chex.assert_trees_all_close(out, z + u - c, atol=1e-7)
