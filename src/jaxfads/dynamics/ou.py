"""Ready-to-use diffusion-style dynamics.

This module provides a stationary Ornstein–Uhlenbeck (OU) drift dynamics.

Notes
-----
This codebase keeps `Dynamics.forward` deterministic; process noise is owned by
`XFADS` (configured via `dyn_conf.state_noise`). `OUDynamics` implements only
the OU drift.
"""

from __future__ import annotations

from jax import Array
from jax import numpy as jnp

from ..base import Dynamics
from ..constraints import constrain_positive, unconstrain_positive


class OUDynamics(Dynamics):
    """Zero-mean Ornstein–Uhlenbeck (OU) drift with Euler discretization.

    Continuous-time form (mean zero):

    ``dz = -theta * z * dt + sigma dW``

    Here we implement only the drift:

    ``z_next = z + dt * (-theta * z)``.

    Parameters (dyn_conf)
    ---------------------
    - `theta`: Mean-reversion rate.
    - `dt`: Constant step size.

    Notes
    -----
    Controls/covariates `u` and `c` are ignored.
    """

    theta_free: Array
    dt: float

    def __init__(self, conf, key: Array):
        del key
        self.conf = conf
        theta0 = jnp.asarray(conf.theta)
        self.theta_free = unconstrain_positive(theta0)
        self.dt = float(conf.dt)

    @property
    def theta(self) -> Array:
        """Constrained (positive) mean-reversion rate."""
        return constrain_positive(self.theta_free)

    def rhs(self, z: Array) -> Array:
        """Continuous-time drift: -theta * z."""
        return -self.theta * z

    def forward(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        del u, c, key
        return z + self.dt * self.rhs(z)
