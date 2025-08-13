"""
eXponential FAmily Dynamical Systems (XFADS).

A JAX-based library for Bayesian state-space modeling using variational inference
with exponential family approximations. XFADS implements flexible nonlinear
dynamical systems with neural network parameterizations for dynamics,
observations, and variational approximations.

Main Classes
------------
XFADS : class
    Main model class orchestrating variational inference for XFADS.
"""

from .smoother import XFADS


__all__ = ["XFADS"]
