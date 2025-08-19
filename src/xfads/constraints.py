import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike


# _MIN_NORM = 1e-6
MAX_EXP = 5.0
EPS = np.finfo(np.float32).eps


def constrain_positive(x):
    """
    Constrain values to be positive using square transformation.

    Parameters
    ----------
    x
        Input values to constrain.

    Returns
    -------
    Positive values computed as x^2 + eps for numerical stability.

    Notes
    -----
    This function ensures all output values are strictly positive by
    applying a square transformation and adding a small epsilon value
    to prevent numerical issues at zero.
    """
    # x0 = MAX_EXP
    # is_too_large = x > x0
    # expx0 = jnp.exp(x0)
    # clipped = jnp.where(is_too_large, x0, x)
    # expx = jnp.exp(clipped)
    # taylor = expx0 + expx0 * (x - x0)
    # return jnp.where(is_too_large, taylor, expx)
    return jnp.square(x) + EPS


def unconstrain_positive(x):
    """
    Unconstrain positive values using square root transformation.

    Parameters
    ----------
    x
        Positive input values to unconstrain.

    Returns
    -------
    Unconstrained values computed as sqrt(x).

    Notes
    -----
    This is the inverse of constrain_positive, mapping positive values
    back to the unconstrained space for optimization.
    """
    # return jnp.log(x + EPS)
    return jnp.sqrt(x)


def softplus_inverse(x: ArrayLike):
    """
    Compute the inverse of the softplus function.

    Parameters
    ----------
    x : ArrayLike
        Input values (should be positive).

    Returns
    -------
    ArrayLike
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


class AbstractConstraint(eqx.Module):
    """
    Abstract base class for parameter constraints.

    This class defines the interface for constraint transformations that
    map between constrained and unconstrained parameter spaces.

    Methods
    -------
    constrain(unconstrained)
        Transform unconstrained parameters to constrained space.
    unconstrain(constrained)
        Transform constrained parameters to unconstrained space.
    __call__(unconstrained)
        Convenience method that calls constrain().
    """
    def constrain(self, unconstrained: ArrayLike) -> ArrayLike: ...

    def unconstrain(self, constrained: ArrayLike) -> ArrayLike: ...

    def __call__(self, unconstrained: ArrayLike) -> ArrayLike:
        return self.constrain(unconstrained)


class Positivity(AbstractConstraint):
    """
    Positivity constraint using softplus transformation.

    This constraint ensures parameters remain positive by using the
    softplus function: softplus(x) = log(1 + exp(x)).

    Methods
    -------
    constrain(unconstrained)
        Apply softplus to ensure positivity.
    unconstrain(constrained)
        Apply inverse softplus to map back to unconstrained space.

    Notes
    -----
    The softplus function is smooth and differentiable everywhere,
    making it suitable for gradient-based optimization while ensuring
    all outputs are strictly positive.
    """
    def constrain(self, unconstrained: ArrayLike) -> ArrayLike:
        return jax.nn.softplus(unconstrained)

    def unconstrain(self, constrained: ArrayLike) -> ArrayLike:
        return softplus_inverse(constrained)
