from types import SimpleNamespace

import chex
from jax import numpy as jnp
from jax import random as jr

from jaxfads.base import Dynamics
from jaxfads.integrators import Identity, Euler, RK4


class _LinearField(Dynamics):
    a: float

    def __init__(self, conf, key):
        del key
        self.conf = conf
        self.a = float(conf.a)

    def eval(self, z, u, c, *, key=None):
        del u, c, key
        return self.a * z


class _DiscreteMap(Dynamics):
    def __init__(self, conf, key):
        del key
        self.conf = conf

    def eval(self, z, u, c, *, key=None):
        del key
        return z + u + c


def test_euler_integrator_linear_field():
    conf = SimpleNamespace(system_type="continuous", dt=0.1, a=2.0)
    field = _LinearField(conf, key=jr.key(0))
    integrator = Euler(conf)

    z = jnp.array([1.0, -2.0])
    out = integrator.step(z, jnp.zeros_like(z), jnp.zeros_like(z), field)
    expected = z + conf.dt * conf.a * z
    chex.assert_trees_all_close(out, expected, atol=1e-7)


def test_rk4_integrator_linear_field_matches_expansion():
    conf = SimpleNamespace(system_type="continuous", dt=0.1, a=1.5)
    field = _LinearField(conf, key=jr.key(0))
    integrator = RK4(conf)

    z = jnp.array([0.5, -1.0])
    out = integrator.step(z, jnp.zeros_like(z), jnp.zeros_like(z), field)

    x = conf.a * conf.dt
    # RK4 stability polynomial for y' = a y
    poly = 1.0 + x + x**2 / 2.0 + x**3 / 6.0 + x**4 / 24.0
    expected = poly * z
    chex.assert_trees_all_close(out, expected, atol=1e-7)


def test_identity_integrator_passthrough():
    conf = SimpleNamespace(system_type="discrete")
    dynamics = _DiscreteMap(conf, key=jr.key(0))
    integrator = Identity(conf)

    z = jnp.array([1.0, 2.0])
    u = jnp.array([0.5, -0.5])
    c = jnp.array([0.1, 0.2])
    out = integrator.step(z, u, c, dynamics)
    chex.assert_trees_all_close(out, z + u + c, atol=1e-7)


def test_integrator_system_type_mismatch_raises():
    bad_conf = SimpleNamespace(system_type="discrete", dt=0.1)
    try:
        Euler(bad_conf)
        raise AssertionError("Euler should fail for discrete system_type.")
    except ValueError:
        pass

    bad_conf2 = SimpleNamespace(system_type="continuous", dt=0.1)
    try:
        Identity(bad_conf2)
        raise AssertionError("Identity should fail for continuous system_type.")
    except ValueError:
        pass
