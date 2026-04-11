"""Minimal discrete state maps.

This module provides an identity/random-walk mean state map.

Notes
-----
This codebase keeps process noise on `XFADS` (configured via
`dyn_conf.state_noise`). `IdentityStateMap` implements only the deterministic
mean map ``z_{t+1} = z_t``; the full latent evolution is therefore the random
walk ``z_{t+1} = z_t + noise``.
"""

from __future__ import annotations

from jax import Array

from ..base import StateMap


class IdentityStateMap(StateMap):
    """Discrete random-walk mean map: ``z_{t+1} = z_t + noise``.

    This state map contributes the deterministic identity mean map
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
            raise ValueError(
                "IdentityStateMap requires dyn_conf.system_type='discrete'."
            )

    def eval(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        """Discrete identity mean map: ``f(z_t) = z_t``."""
        del u, c, key
        return z
