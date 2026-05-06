"""
Exponential Family Dynamical Systems (XFADS).

A JAX-based library for Bayesian state-space modeling using variational inference
with exponential family approximations. XFADS implements flexible nonlinear
dynamical systems with neural network parameterizations for dynamics,
observations, and variational approximations.
"""

from __future__ import annotations

import sys

from . import dynamics as _dynamics
from . import integrators as _integrators
from .dynamics import function as _dynamics_function
from .dynamics import identity as _dynamics_identity
from .dynamics import ou as _dynamics_ou
from .logging import configure_logging
from .smoother import XFADS
from .trainer import train

# Legacy import-path aliases. New public paths are `jaxfads.dynamics` and
# `jaxfads.integrators`; older `state_maps` / `steppers` imports remain usable
# during the deprecation window.
sys.modules.setdefault(__name__ + ".state_maps", _dynamics)
sys.modules.setdefault(__name__ + ".state_maps.identity", _dynamics_identity)
sys.modules.setdefault(__name__ + ".state_maps.ou", _dynamics_ou)
sys.modules.setdefault(__name__ + ".state_maps.function", _dynamics_function)
sys.modules.setdefault(__name__ + ".steppers", _integrators)

__all__ = ["XFADS", "train", "configure_logging"]
