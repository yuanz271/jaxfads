import chex
from jax import numpy as jnp
from jax import random as jr
from omegaconf import OmegaConf

import jaxfads.dynamics  # noqa: F401  (register built-in Dynamics)
from jaxfads.base import Dynamics


def test_ou_dynamics_registry_and_forward():
    cls = Dynamics.get_subclass("OUDynamics")

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
        )
    )

    dyn = cls(conf, key=jr.key(0))

    z = jr.normal(jr.key(1), (state_dim,))
    out = dyn.forward(z, jnp.zeros((0,)), jnp.zeros((0,)))

    expected = (1.0 - dyn.theta * dt) * z
    chex.assert_trees_all_close(out, expected, atol=1e-6)
    chex.assert_shape(out, (state_dim,))
    chex.assert_tree_all_finite(out)
