"""Ready-to-use diffusion-style dynamics."""

from __future__ import annotations

from jax import Array
from jax import numpy as jnp

from ..base import Dynamics
from ..constraints import constrain_positive, unconstrain_positive


class OU(Dynamics):
    """Zero-mean Ornstein–Uhlenbeck (OU) continuous-time dynamics.

    Continuous-time form (mean zero):

    ``dz = -theta * z * dt + sigma dW``

    ``dz/dt = -theta * z``.

    Parameters (dyn_conf)
    ---------------------
    - `theta`: Mean-reversion rate.
    Notes
    -----
    Controls/covariates `u` and `c` are ignored.
    """

    theta_free: Array

    def __init__(self, conf, key: Array):
        self.conf = conf
        theta0 = jnp.asarray(conf.theta)
        self.theta_free = unconstrain_positive(theta0)

    @property
    def theta(self) -> Array:
        """Constrained (positive) mean-reversion rate."""
        return constrain_positive(self.theta_free)

    def eval(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        """Continuous-time drift: ``dz/dt = -theta * z`` (``u``, ``c``, ``key`` unused)."""
        return -self.theta * z


__all__ = ["OU"]
