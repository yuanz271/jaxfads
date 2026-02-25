"""
Parameter constraint transformations for XFADS.

This module provides bijective transformations between constrained and
unconstrained parameter spaces, enabling gradient-based optimization
of parameters with domain restrictions (e.g., positive variances).

Uses softplus / inverse-softplus for positivity constraints, which have
non-zero gradients everywhere.
"""

import jax
from jax import Array
from jax import numpy as jnp

#: Machine epsilon for float32, used for numerical stability (e.g., matrix
#: damping, covariance floors, Cholesky regularisation).
_EPS: float = float(jnp.finfo(jnp.float32).eps)


def constrain_positive(x: Array) -> Array:
    """
    Constrain values to be positive using softplus transformation.

    Parameters
    ----------
    x : Array
        Input values to constrain.

    Returns
    -------
    Array
        Positive values computed as log(1 + exp(x)).

    Notes
    -----
    Uses softplus, which has non-zero gradient everywhere (sigmoid(x)),
    avoiding the zero-gradient problem of the square transformation at
    the origin.
    """
    return jax.nn.softplus(x)


def unconstrain_positive(x: Array) -> Array:
    """
    Unconstrain positive values using inverse softplus.

    Parameters
    ----------
    x : Array
        Positive input values to unconstrain.

    Returns
    -------
    Array
        Unconstrained values such that softplus(result) ≈ x.

    Notes
    -----
    This is the inverse of constrain_positive, mapping positive values
    back to the unconstrained space for optimization.
    """
    return softplus_inverse(x)


def softplus_inverse(x: Array):
    """
    Compute the inverse of the softplus function.

    Parameters
    ----------
    x : Array
        Input values (should be positive).

    Returns
    -------
    Array
        Inverse softplus values.

    Notes
    -----
    The softplus function is softplus(y) = log(1 + exp(y)).
    This function computes y given softplus(y) = x.

    For numerical stability, special handling is applied for very small
    and very large input values to avoid overflow/underflow issues.
    """
    threshold = jnp.log(jnp.finfo(jnp.asarray(x).dtype).eps) + 2.0

    is_too_small = x < jnp.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = threshold
    too_large_value = x

    x = jnp.where(is_too_small | is_too_large, 1.0, x)
    y = x + jnp.log(-jnp.expm1(-x))  # == log(expm1(x))
    return jnp.where(
        is_too_small, too_small_value, jnp.where(is_too_large, too_large_value, y)
    )
