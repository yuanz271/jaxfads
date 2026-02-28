import chex
from jax import numpy as jnp
from jax import random as jr
from omegaconf import OmegaConf

import jaxfads.state_maps  # noqa: F401  (register built-in StateMap)
import jaxfads.steppers  # noqa: F401  (register built-in Stepper)
from jaxfads.base import StateMap, Stepper


def test_ou_state_map_registry_and_euler_step():
    map_cls = StateMap.get_subclass("OUStateMap")
    stepper_cls = Stepper.get_subclass("EulerStepper")

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

    state_map = map_cls(conf, key=jr.key(0))
    stepper = stepper_cls(conf)

    z = jr.normal(jr.key(1), (state_dim,))
    out = stepper.step(z, jnp.zeros((0,)), jnp.zeros((0,)), state_map)

    expected = (1.0 - state_map.theta * dt) * z
    chex.assert_trees_all_close(out, expected, atol=1e-6)
    chex.assert_shape(out, (state_dim,))
    chex.assert_tree_all_finite(out)
