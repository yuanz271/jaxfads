"""Public dynamics namespace.

Preferred import path for built-in dynamics classes.
"""

from __future__ import annotations

import sys

from . import functional as _functional
from .functional import FunctionalDynamics, FunctionDynamics
from .identity import IdentityDynamics
from .ou import OUDynamics

sys.modules.setdefault(__name__ + ".function", _functional)

__all__ = ["IdentityDynamics", "OUDynamics", "FunctionalDynamics", "FunctionDynamics"]
