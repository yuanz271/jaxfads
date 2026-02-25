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

from ..base import Dynamics


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

    theta: float
    dt: float

    def __init__(self, conf, key: Array):
        del key
        self.conf = conf
        self.theta = float(conf.theta)
        self.dt = float(conf.dt)

    def rhs(self, z: Array) -> Array:
        """Continuous-time drift: -theta * z."""
        return -self.theta * z

    def forward(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        del u, c, key
        return z + self.dt * self.rhs(z)
