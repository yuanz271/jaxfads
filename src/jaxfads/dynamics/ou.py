"""Ready-to-use diffusion-style dynamics."""

from __future__ import annotations

from jax import Array
from jax import numpy as jnp

from ..base import Dynamics
from ..constraints import constrain_positive, unconstrain_positive


class OUDynamics(Dynamics):
    """Zero-mean Ornstein–Uhlenbeck (OU) continuous-time dynamics.

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
            raise ValueError("OUDynamics requires dyn_conf.system_type='continuous'.")
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


__all__ = ["OUDynamics"]
