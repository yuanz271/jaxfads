"""
Variational inference utilities for XFADS.

This module implements the Evidence Lower Bound (ELBO) computation for
variational inference in XFADS. The ELBO provides a tractable
lower bound on the log marginal likelihood that can be optimized during training.
"""

from collections.abc import Callable

from jaxtyping import Array, Scalar

from .distributions import Approx


def elbo(
    key: Array,
    t: Array,
    moment: Array,
    moment_p: Array,
    y: Array,
    eloglik: Callable[..., Scalar],
    approx: type[Approx],
    *,
    mc_size: int,
) -> Scalar:
    """
    Compute Evidence Lower Bound (ELBO) for a single time point.

    The ELBO provides a tractable lower bound on the log marginal likelihood
    by decomposing it into expected log-likelihood and KL divergence terms.
    This is the fundamental objective function optimized in variational inference.

    Parameters
    ----------
    key : Array
        JAX random key for stochastic computations.
    t : Array
        Time index for the current observation.
    moment : Array
        Moment parameters of the posterior distribution q(z_t | y_{1:T}).
    moment_p : Array
        Moment parameters of the prior/predictive distribution p(z_t | y_{1:t-1}).
    y : Array, shape (observation_dim,)
        Observed data at time t.
    eloglik : Callable
        Function computing expected log-likelihood E_q[log p(y_t | z_t)].
    approx : type[Approx]
        Exponential family approximation class for KL computation.
    mc_size : int
        Number of Monte Carlo samples for expectation approximation.

    Returns
    -------
    Scalar
        ELBO value for the single time point.

    Notes
    -----
    The ELBO decomposes as:

    ELBO_t = E_{q(z_t)}[log p(y_t | z_t)] - KL(q(z_t | y_{1:T}) || p(z_t | y_{1:t-1}))

    where:
    - First term: Expected log-likelihood (reconstruction term)
    - Second term: KL divergence (regularization term)

    The KL term encourages the posterior to stay close to the prior/predictive
    distribution, preventing overfitting and ensuring smooth state trajectories.

    For the full sequence, the total ELBO is the sum over all time points:
    ELBO = Σ_t ELBO_t

    Maximizing the ELBO is equivalent to minimizing the negative ELBO, which
    is commonly used as the loss function during training.

    Examples
    --------
    >>> # Compute ELBO for Poisson observations
    >>> key = jrnd.key(42)
    >>> t = 0
    >>> moment_post = jnp.array([0.1, 0.2, 0.05])  # posterior moments
    >>> moment_prior = jnp.array([0.0, 0.1, 0.02])  # prior moments
    >>> y = jnp.array([5, 3, 1])  # count observations
    >>>
    >>> elbo_val = elbo(
    ...     key, t, moment_post, moment_prior, y,
    ...     eloglik=poisson_model.eloglik,
    ...     approx=DiagMVN,
    ...     mc_size=100
    ... )
    """
    ell: Scalar = eloglik(key, t, moment, y, approx, mc_size)
    kl: Scalar = approx.kl(moment, moment_p)
    return ell - kl
