"""
Exponential Family Dynamical Systems (XFADS).

A JAX-based library for Bayesian state-space modeling using variational inference
with exponential family approximations. XFADS implements flexible nonlinear
dynamical systems with neural network parameterizations for dynamics,
observations, and variational approximations.
"""

from __future__ import annotations

from .logging import configure_logging
from .smoother import XFADS
from .trainer import train

__all__ = ["XFADS", "train", "configure_logging"]
