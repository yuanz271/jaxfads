import chex
import pytest
from jax import numpy as jnp
from jax import random as jr
from omegaconf import OmegaConf

import jaxfads.dynamics  # noqa: F401  (register built-in Dynamics)
import jaxfads.integrators  # noqa: F401  (register built-in Integrator)
from jaxfads.base import Dynamics, Integrator


def test_ou_dynamics_registry_and_euler_integrator():
    dynamics_cls = Dynamics.get_subclass("OUDynamics")
    integrator_cls = Integrator.get_subclass("EulerIntegrator")

    theta = 2.0
    dt = 0.1
    state_dim = 4

    conf = OmegaConf.create(
        dict(
            state_dim=state_dim,
            observation_dim=1,
            input_dim=0,
            context_dim=0,
            theta=theta,
            dt=dt,
            system_type="continuous",
        )
    )

    dynamics = dynamics_cls(conf, key=jr.key(0))
    integrator = integrator_cls(conf)

    z = jr.normal(jr.key(1), (state_dim,))
    out = integrator.step(z, jnp.zeros((0,)), jnp.zeros((0,)), dynamics)

    expected = (1.0 - dynamics.theta * dt) * z
    chex.assert_trees_all_close(out, expected, atol=1e-6)
    chex.assert_shape(out, (state_dim,))
    chex.assert_tree_all_finite(out)


def test_identity_dynamics_registry_and_identity_integrator():
    dynamics_cls = Dynamics.get_subclass("IdentityDynamics")
    integrator_cls = Integrator.get_subclass("IdentityIntegrator")

    state_dim = 4
    conf = OmegaConf.create(
        dict(
            state_dim=state_dim,
            observation_dim=1,
            input_dim=0,
            context_dim=0,
            system_type="discrete",
        )
    )

    dynamics = dynamics_cls(conf, key=jr.key(0))
    integrator = integrator_cls(conf)

    z = jr.normal(jr.key(1), (state_dim,))
    out = integrator.step(z, jnp.zeros((0,)), jnp.zeros((0,)), dynamics)

    chex.assert_trees_all_close(out, z, atol=1e-6)
    chex.assert_shape(out, (state_dim,))
    chex.assert_tree_all_finite(out)


def test_identity_dynamics_rejects_continuous_system_type():
    conf = OmegaConf.create(
        dict(
            state_dim=2,
            observation_dim=1,
            input_dim=0,
            context_dim=0,
            system_type="continuous",
        )
    )

    with pytest.raises(
        ValueError, match="IdentityDynamics requires dyn_conf.system_type='discrete'."
    ):
        Dynamics.get_subclass("IdentityDynamics")(conf, key=jr.key(0))
