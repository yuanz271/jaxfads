"""
Exponential Family Dynamical Systems (XFADS).

A JAX-based library for Bayesian state-space modeling using variational inference
with exponential family approximations. XFADS implements flexible nonlinear
dynamical systems with neural network parameterizations for dynamics,
observations, and variational approximations.

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
...     'obs_conf': {
...         'model': 'GLM',
...         'observation_dim': 50,
...         'state_dim': 10,
...         'likelihood': 'Poisson',
...         'cov': [1.0] * 50,
...         'norm_readout': False,
...     },
... })
>>>
>>> # Initialize model
>>> key = jrnd.key(42)
>>> model = XFADS(conf, key)
"""

from .base import Dynamics, Noise, ObservationModel
from .distributions import Approx, DiagMVN, FullMVN
from .logging import configure_logging, get_logger
from .smoother import XFADS
from .trainer import train

__all__ = [
    # Core model
    "XFADS",
    "Dynamics",
    "Noise",
    "ObservationModel",

    # Distributions
    "Approx",
    "DiagMVN",
    "FullMVN",
    # Training
    "train",
    # Logging
    "configure_logging",
    "get_logger",
]
