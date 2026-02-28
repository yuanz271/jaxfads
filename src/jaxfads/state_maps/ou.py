"""Ready-to-use diffusion-style state maps.

This module provides a stationary Ornstein–Uhlenbeck (OU) state map.

Notes
-----
This codebase keeps process noise on `XFADS` (configured via
`dyn_conf.state_noise`). `OUStateMap` implements only the continuous-time drift
term ``dz/dt = -theta * z``.
"""

from __future__ import annotations

from jax import Array
from jax import numpy as jnp

from ..base import StateMap
from ..constraints import constrain_positive, unconstrain_positive


class OUStateMap(StateMap):
    """Zero-mean Ornstein–Uhlenbeck (OU) continuous-time state map.

    Continuous-time form (mean zero):

    ``dz = -theta * z * dt + sigma dW``

    ``dz/dt = -theta * z``.

    Parameters (dyn_conf)
    ---------------------
    - `theta`: Mean-reversion rate.
    Notes
    -----
    Controls/covariates `u` and `c` are ignored. This map requires
    ``dyn_conf.system_type='continuous'``.
    """

    theta_free: Array

    def __init__(self, conf, key: Array):
        del key
        self.conf = conf
        if str(conf.system_type) != "continuous":
            raise ValueError("OUStateMap requires dyn_conf.system_type='continuous'.")
        theta0 = jnp.asarray(conf.theta)
        self.theta_free = unconstrain_positive(theta0)

    @property
    def theta(self) -> Array:
        """Constrained (positive) mean-reversion rate."""
        return constrain_positive(self.theta_free)

    def eval(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        """Continuous-time drift: ``dz/dt = -theta * z``."""
        del u, c, key
        return -self.theta * z
