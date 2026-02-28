from types import SimpleNamespace

import chex
from jax import numpy as jnp
from jax import random as jr

from jaxfads.base import StateMap
from jaxfads.steppers import DiscreteStepper, EulerStepper, RK4Stepper


class _LinearField(StateMap):
    a: float

    def __init__(self, conf, key):
        del key
        self.conf = conf
        self.a = float(conf.a)

    def eval(self, z, u, c, *, key=None):
        del u, c, key
        return self.a * z


class _DiscreteMap(StateMap):
    def __init__(self, conf, key):
        del key
        self.conf = conf

    def eval(self, z, u, c, *, key=None):
        del key
        return z + u + c


def test_euler_stepper_linear_field():
    conf = SimpleNamespace(system_type="continuous", dt=0.1, a=2.0)
    field = _LinearField(conf, key=jr.key(0))
    stepper = EulerStepper(conf, key=jr.key(1))

    z = jnp.array([1.0, -2.0])
    out = stepper.step(z, jnp.zeros_like(z), jnp.zeros_like(z), field)
    expected = z + conf.dt * conf.a * z
    chex.assert_trees_all_close(out, expected, atol=1e-7)


def test_rk4_stepper_linear_field_matches_expansion():
    conf = SimpleNamespace(system_type="continuous", dt=0.1, a=1.5)
    field = _LinearField(conf, key=jr.key(0))
    stepper = RK4Stepper(conf, key=jr.key(1))

    z = jnp.array([0.5, -1.0])
    out = stepper.step(z, jnp.zeros_like(z), jnp.zeros_like(z), field)

    x = conf.a * conf.dt
    # RK4 stability polynomial for y' = a y
    poly = 1.0 + x + x**2 / 2.0 + x**3 / 6.0 + x**4 / 24.0
    expected = poly * z
    chex.assert_trees_all_close(out, expected, atol=1e-7)


def test_discrete_stepper_passthrough():
    conf = SimpleNamespace(system_type="discrete")
    state_map = _DiscreteMap(conf, key=jr.key(0))
    stepper = DiscreteStepper(conf, key=jr.key(1))

    z = jnp.array([1.0, 2.0])
    u = jnp.array([0.5, -0.5])
    c = jnp.array([0.1, 0.2])
    out = stepper.step(z, u, c, state_map)
    chex.assert_trees_all_close(out, z + u + c, atol=1e-7)


def test_stepper_system_type_mismatch_raises():
    bad_conf = SimpleNamespace(system_type="discrete", dt=0.1)
    try:
        EulerStepper(bad_conf, key=jr.key(0))
        raise AssertionError("EulerStepper should fail for discrete system_type.")
    except ValueError:
        pass

    bad_conf2 = SimpleNamespace(system_type="continuous", dt=0.1)
    try:
        DiscreteStepper(bad_conf2, key=jr.key(0))
        raise AssertionError("DiscreteStepper should fail for continuous system_type.")
    except ValueError:
        pass
