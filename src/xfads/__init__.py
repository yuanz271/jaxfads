"""
eXponential FAmily Dynamical Systems (XFADS).

A JAX-based library for Bayesian state-space modeling using variational inference
with exponential family approximations. XFADS implements flexible nonlinear
dynamical systems with neural network parameterizations for dynamics,
observations, and variational approximations.

Classes
-------
XFADS
    Main model class orchestrating variational inference for XFADS.

Notes
-----
XFADS provides a unified framework for Bayesian state-space modeling with:
- Neural network parameterizations for dynamics and observations
- Variational inference with exponential family approximations
- Support for various observation models (Poisson, Gaussian)
- Efficient JAX-based implementation with automatic differentiation

Examples
--------
>>> import jax.random as jrnd
>>> from omegaconf import DictConfig
>>> 
>>> # Create model configuration
>>> conf = DictConfig({
...     'state_dim': 10,
...     'observation_dim': 50,
...     'mc_size': 100,
...     'approx': 'DiagMVN',
...     'forward': 'Linear',
...     'observation': 'Poisson'
... })
>>> 
>>> # Initialize model
>>> key = jrnd.key(42)
>>> model = XFADS(conf, key)
"""

from .smoother import XFADS


__all__ = ["XFADS"]
