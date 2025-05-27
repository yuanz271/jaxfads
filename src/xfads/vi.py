from collections.abc import Callable

from jaxtyping import Array, PRNGKeyArray, Scalar

from .distributions import Approx


def elbo(
    key: PRNGKeyArray,
    t: Array,
    moment: Array,
    moment_p: Array,
    y: Array,
    eloglik: Callable[..., Scalar],
    approx: type[Approx],
    *,
    mc_size: int,
) -> Scalar:
    """Single time point"""
    ell: Scalar = eloglik(key, t, moment, y, approx, mc_size)
    kl: Scalar = approx.kl(moment, moment_p)

    lval: Scalar = ell - kl

    return lval
