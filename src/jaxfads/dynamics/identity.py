"""Minimal discrete dynamics."""

from __future__ import annotations

from jax import Array

from ..base import Dynamics


class Identity(Dynamics):
    """Discrete random-walk mean dynamics: ``z_{t+1} = z_t + noise``.

    This dynamics contributes the deterministic identity mean map
    ``f(z_t) = z_t``; process noise remains configured on `XFADS` via
    ``dyn_conf.state_noise``.

    Notes
    -----
    Controls/covariates `u` and `c` are ignored. This map requires
    ``dyn_conf.system_type='discrete'``.
    """

    def __init__(self, conf, key: Array):
        del key
        self.conf = conf
        if str(conf.system_type) != "discrete":
            raise ValueError("Identity requires dyn_conf.system_type='discrete'.")

    def eval(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        """Discrete identity mean map: ``f(z_t) = z_t``."""
        del u, c, key
        return z


IdentityDynamics = Identity
Dynamics._subclasses["IdentityDynamics"] = Identity

__all__ = ["Identity", "IdentityDynamics"]
