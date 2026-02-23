"""
Noise models for XFADS dynamics.

This module provides concrete noise implementations used by dynamics models
to represent process noise. The abstract :class:`~jaxfads.base.Noise`
protocol is defined in :mod:`jaxfads.base`.
"""

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from .constraints import constrain_positive, unconstrain_positive


class DiagGaussian(eqx.Module, strict=True):
    """
    Diagonal Gaussian noise model for dynamics systems.

    Implements process noise with diagonal covariance structure, assuming
    independence between state dimensions. More efficient than full
    covariance but less expressive.

    Parameters
    ----------
    cov : ArrayLike
        Initial covariance value (Array applied to all dimensions).
    size : int
        Dimensionality of the noise (should match state dimension).

    Attributes
    ----------
    unconstrained_cov : Array, shape (size,)
        Unconstrained covariance parameters for optimization.

    Notes
    -----
    The covariance is parameterized in unconstrained space to ensure
    positive values during optimization. The actual covariance is
    obtained via constrain_positive() transformation.
    """

    unconstrained_cov: Array

    def __init__(self, cov: Array, size: int):  # pyright: ignore[reportMissingSuperCall]
        self.unconstrained_cov = jnp.full(size, fill_value=unconstrain_positive(cov))

    def cov(self) -> Array:
        """
        Get the diagonal covariance vector.

        Returns
        -------
        Array, shape (size,)
            Diagonal elements of the covariance matrix.

        Notes
        -----
        Applies positive constraint to ensure valid covariance values.
        """
        return constrain_positive(self.unconstrained_cov)


__all__ = ["DiagGaussian"]
