import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike


_MIN_NORM = 1e-6
MAX_EXP = 5.0
EPS = np.finfo(np.float32).eps


def constrain_positive(x):
    # x0 = MAX_EXP
    # is_too_large = x > x0
    # expx0 = jnp.exp(x0)
    # clipped = jnp.where(is_too_large, x0, x)
    # expx = jnp.exp(clipped)
    # taylor = expx0 + expx0 * (x - x0)
    # return jnp.where(is_too_large, taylor, expx)
    return jnp.square(x) + EPS


def unconstrain_positive(x):
    # return jnp.log(x + EPS)
    return jnp.sqrt(x)


def softplus_inverse(x: ArrayLike):
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
    def constrain(self, unconstrained: ArrayLike) -> ArrayLike: ...

    def unconstrain(self, constrained: ArrayLike) -> ArrayLike: ...

    def __call__(self, unconstrained: ArrayLike) -> ArrayLike:
        return self.constrain(unconstrained)


class Positivity(AbstractConstraint):
    def constrain(self, unconstrained: ArrayLike) -> ArrayLike:
        return jax.nn.softplus(unconstrained)

    def unconstrain(self, constrained: ArrayLike) -> ArrayLike:
        return softplus_inverse(constrained)
