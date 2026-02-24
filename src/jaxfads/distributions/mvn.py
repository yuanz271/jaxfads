"""
Exponential-family variational distributions for XFADS.

This module provides a unified multivariate normal (MVN) distribution with
natural and moment parameterizations for variational inference in XFADS.
The covariance structure is ``diag(d) + U U^T`` with configurable rank,
subsuming diagonal and low-rank cases.
"""

from jax import Array
from jax import numpy as jnp
from jax import random as jrnd
from tensorflow_probability.substrates.jax import distributions as tfd

from ..base import Approx
from ..constraints import _EPS, constrain_positive, unconstrain_positive


def _damping_inv(a: Array, damping: float = _EPS) -> Array:
    """
    Compute the inverse of a matrix with damping for numerical stability.

    Uses ``jnp.linalg.solve`` instead of explicit ``inv`` for better
    numerical conditioning.

    Parameters
    ----------
    a : Array, shape (..., D, D)
        Input square matrix to be inverted.
    damping : float, default=_EPS
        Damping factor added to the diagonal for numerical stability.

    Returns
    -------
    Array, shape (..., D, D)
        Inverse of (a + damping * I).

    Notes
    -----
    The damping term prevents singular matrices and improves numerical
    stability by adding a small positive value to the diagonal elements.
    """
    eye = jnp.eye(a.shape[-1])
    return jnp.linalg.solve(a + damping * eye, eye)


# ---------------------------------------------------------------------------
# Unified MVN implementation
# ---------------------------------------------------------------------------


class MVN(Approx):
    """
    Multivariate normal with configurable covariance structure.

    The covariance is ``Σ = diag(d) + U Uᵀ`` where ``d`` is a positive
    diagonal and ``U`` is a ``(D, rank)`` factor matrix.

    Parameters
    ----------
    dim : int
        State dimensionality.
    rank : int, optional
        Covariance rank.  ``0`` = diagonal, ``> 0`` = diag + low-rank
        factor.  Must satisfy ``0 <= rank <= dim``.  Default is ``0``.

    Moment layout : ``[loc(D), cov_diag(D), cov_factor(D×r)]`` = D(2+r).
    Natural layout: ``[η₁(D), η₂(D)]`` (rank 0) or
    ``[η₁(D), η₂_flat(D²)]`` (rank > 0).

    Notes
    -----
    Uses ``tfd.MultivariateNormalDiagPlusLowRankCovariance`` as backend
    for sampling (rank > 0) and
    ``tfd.MultivariateNormalFullCovariance`` for KL computation.
    """

    def __init__(self, dim: int, rank: int = 0):
        if not (0 <= rank <= dim):
            raise ValueError(
                f"rank must satisfy 0 <= rank <= dim, got rank={rank}, dim={dim}"
            )
        self._dim = dim
        self._rank = rank

    # -- helpers -------------------------------------------------------------

    def _decompose_cov(self, sigma: Array) -> tuple[Array, Array]:
        """Decompose Σ into ``diag(cov_diag) + cov_factor @ cov_factor.T``.

        The off-diagonal matrix ``Σ − diag(diag(Σ))`` is eigendecomposed
        and the top-*r* positive eigenvalues/vectors become the low-rank
        factor.  The diagonal residual is then ``diag(Σ) − diag(U Uᵀ)``.

        Parameters
        ----------
        sigma : Array, shape (D, D)
            Symmetric PSD covariance matrix.

        Returns
        -------
        cov_diag : Array, shape (D,)
            Positive diagonal residual.
        cov_factor : Array, shape (D, r)
            Low-rank factor.
        """
        r = self._rank
        sigma_sym = 0.5 * (sigma + sigma.T)
        diag_sigma = jnp.diag(sigma_sym)
        # Off-diagonal matrix (zero diagonal, keeps off-diag correlations)
        off_diag = sigma_sym - jnp.diag(diag_sigma)
        eigenvalues, eigenvectors = jnp.linalg.eigh(off_diag)
        # Take top *r* eigenvalues (largest / most positive)
        top_vals = jnp.maximum(eigenvalues[-r:], _EPS)
        top_vecs = eigenvectors[:, -r:]
        cov_factor = top_vecs * jnp.sqrt(top_vals)          # (D, r)
        low_rank_diag = jnp.sum(cov_factor ** 2, axis=1)    # diag(U Uᵀ)
        cov_diag = jnp.maximum(diag_sigma - low_rank_diag, _EPS)
        return cov_diag, cov_factor

    def _split_moment(self, moment: Array) -> tuple[Array, Array, Array]:
        """Unpack structured moment ``[loc, cov_diag, cov_factor_flat]``.

        Returns
        -------
        loc : Array, shape (D,)
        cov_diag : Array, shape (D,)
        cov_factor : Array, shape (D, r)
        """
        d, r = self._dim, self._rank
        loc = moment[:d]
        cov_diag = moment[d : 2 * d]
        cov_factor = moment[2 * d :].reshape(d, r)
        return loc, cov_diag, cov_factor

    def _build_cov(self, cov_diag: Array, cov_factor: Array) -> Array:
        """Materialize full covariance from structured parts."""
        return jnp.diag(cov_diag) + cov_factor @ cov_factor.T

    def _pack_moment(self, loc: Array, cov_diag: Array, cov_factor: Array) -> Array:
        """Concatenate structured moment."""
        return jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    # -- Approx interface ----------------------------------------------------

    def param_size(self, state_dim: int) -> int:
        """See base class. Natural parameter size."""
        d = self._dim
        if self._rank == 0:
            return 2 * d
        return d + d * d

    def moment_size(self, state_dim: int) -> int:
        """See base class. Moment parameter size: D(2 + r)."""
        return self._dim * (2 + self._rank)

    # -- natural ↔ moment ---------------------------------------------------

    def natural_to_moment(self, natural: Array) -> Array:
        """See base class."""
        d, r = self._dim, self._rank
        if r == 0:
            eta1, eta2 = jnp.split(natural, 2)
            cov_diag = -0.5 / eta2
            loc = -0.5 * eta1 / eta2
            return jnp.concatenate((loc, cov_diag))

        eta1, eta2_flat = jnp.split(natural, [d])
        P = -2.0 * jnp.reshape(eta2_flat, (d, d))
        loc = jnp.linalg.solve(P, eta1)
        sigma = _damping_inv(P)
        cov_diag, cov_factor = self._decompose_cov(sigma)
        return self._pack_moment(loc, cov_diag, cov_factor)

    def moment_to_natural(self, moment: Array) -> Array:
        """See base class.

        Notes
        -----
        A minimum floor of ``_EPS`` is applied to the covariance
        diagonal before inversion to prevent extreme natural
        parameters.
        """
        if self._rank == 0:
            mean, cov_diag = jnp.split(moment, 2)
            cov_diag = jnp.maximum(cov_diag, _EPS)
            eta2 = -0.5 / cov_diag
            eta1 = mean / cov_diag
            return jnp.concatenate((eta1, eta2))

        loc, cov_diag, cov_factor = self._split_moment(moment)
        sigma = self._build_cov(jnp.maximum(cov_diag, _EPS), cov_factor)
        P = _damping_inv(sigma)
        eta1 = P @ loc
        eta2 = -0.5 * P
        return jnp.concatenate((eta1, eta2.flatten()))

    # -- sampling -----------------------------------------------------------

    def sample_by_moment(
        self, key: Array, moment: Array, mc_size: int | None = None
    ) -> Array:
        """See base class."""
        if self._rank == 0:
            mean, cov_diag = jnp.split(moment, 2)
            std = jnp.sqrt(jnp.maximum(cov_diag, _EPS))
            shape = mean.shape if mc_size is None else (mc_size,) + mean.shape
            return mean + std * jrnd.normal(key, shape)

        loc, cov_diag, cov_factor = self._split_moment(moment)
        dist = tfd.MultivariateNormalDiagPlusLowRankCovariance(
            loc=loc,
            cov_diag_factor=jnp.maximum(cov_diag, _EPS),
            cov_perturb_factor=cov_factor,
        )
        return dist.sample(mc_size, seed=key)

    # -- KL divergence ------------------------------------------------------

    def kl(self, moment1: Array, moment2: Array) -> Array:
        """See base class.

        Uses ``MultivariateNormalFullCovariance`` for all ranks to
        ensure KL is always registered.
        """
        m1, cov1 = self.moment_to_canon(moment1)
        m2, cov2 = self.moment_to_canon(moment2)
        if self._rank == 0:
            # moment_to_canon returns diagonal vectors
            cov1_full = jnp.diag(jnp.maximum(cov1, _EPS))
            cov2_full = jnp.diag(jnp.maximum(cov2, _EPS))
        else:
            cov1_full = cov1
            cov2_full = cov2
        return tfd.kl_divergence(
            tfd.MultivariateNormalFullCovariance(m1, cov1_full),
            tfd.MultivariateNormalFullCovariance(m2, cov2_full),
            allow_nan_stats=False,
        )

    # -- canonical form -----------------------------------------------------

    def moment_to_canon(self, moment: Array) -> tuple[Array, Array]:
        """See base class.

        Returns
        -------
        mean : Array, shape (D,)
        cov : Array
            Diagonal vector (D,) for rank 0, full matrix (D, D) otherwise.
        """
        if self._rank == 0:
            mean, cov_diag = jnp.split(moment, 2, -1)
            return mean, cov_diag

        loc, cov_diag, cov_factor = self._split_moment(moment)
        cov = self._build_cov(cov_diag, cov_factor)
        return loc, cov

    def canon_to_moment(self, mean: Array, cov: Array) -> Array:
        """See base class.

        Parameters
        ----------
        mean : Array, shape (D,)
        cov : Array
            Diagonal vector (D,) for rank 0, full matrix (D, D) otherwise.
        """
        if self._rank == 0:
            return jnp.concatenate((mean, cov))

        cov_diag, cov_factor = self._decompose_cov(cov)
        return self._pack_moment(mean, cov_diag, cov_factor)

    def full_cov(self, cov: Array) -> Array:
        """See base class."""
        if self._rank == 0:
            return jnp.diag(cov)
        return cov

    # -- constrain / unconstrain --------------------------------------------

    def constrain_moment(self, unconstrained: Array) -> Array:
        """See base class.

        Applies ``softplus`` to the diagonal covariance entries;
        the low-rank factor (if any) is left unconstrained.
        """
        d = self._dim
        if self._rank == 0:
            loc, v = jnp.split(unconstrained, 2)
            return jnp.concatenate((loc, constrain_positive(v)))

        loc = unconstrained[:d]
        diag_unc = unconstrained[d : 2 * d]
        factor = unconstrained[2 * d :]
        return jnp.concatenate((loc, constrain_positive(diag_unc), factor))

    def constrain_natural(self, unconstrained: Array) -> Array:
        """See base class."""
        d = self._dim
        if self._rank == 0:
            n1, n2 = jnp.split(unconstrained, 2)
            return jnp.concatenate((n1, -constrain_positive(n2)))

        nat1, flat_L = jnp.split(unconstrained, [d])
        L = jnp.tril(jnp.reshape(flat_L, (d, d)))
        return jnp.concatenate((nat1, -(L @ L.T).flatten()))

    def unconstrain_natural(self, natural: Array) -> Array:
        """See base class."""
        d = self._dim
        if self._rank == 0:
            n1, n2 = jnp.split(natural, 2)
            return jnp.concatenate((n1, unconstrain_positive(-n2)))

        nat1, nat2_flat = jnp.split(natural, [d])
        neg_nat2 = jnp.reshape(-nat2_flat, (d, d))
        L = jnp.linalg.cholesky(neg_nat2 + _EPS * jnp.eye(d))
        return jnp.concatenate((nat1, L.flatten()))

    def unconstrain_moment(self, moment: Array) -> Array:
        """See base class.

        Applies ``softplus_inverse`` to the diagonal covariance entries;
        the low-rank factor is left as-is.
        """
        d = self._dim
        if self._rank == 0:
            loc, v = jnp.split(moment, 2)
            return jnp.concatenate((loc, unconstrain_positive(v)))

        loc = moment[:d]
        cov_diag = moment[d : 2 * d]
        factor = moment[2 * d :]
        return jnp.concatenate((loc, unconstrain_positive(cov_diag), factor))

    # -- prior / noise ------------------------------------------------------

    def prior_natural(self, state_dim: int) -> Array:
        """See base class. Returns N(0, I) in natural form."""
        d = self._dim
        if self._rank == 0:
            eta1 = jnp.zeros(d)
            eta2 = jnp.full(d, -0.5)
            return jnp.concatenate((eta1, eta2))

        eta1 = jnp.zeros(d)
        eta2 = -0.5 * jnp.eye(d)
        return jnp.concatenate((eta1, eta2.flatten()))

    def init_noise(self, scale: float, state_dim: int) -> Array:
        """See base class. Isotropic noise N(0, scale·I).

        All variance goes into ``cov_diag``; the low-rank factor
        is initialised to zero.
        """
        d, r = self._dim, self._rank
        loc = jnp.zeros(d)
        cov_diag = jnp.full(d, scale)
        cov_factor = jnp.zeros(d * r)
        moment = jnp.concatenate((loc, cov_diag, cov_factor))
        return self.unconstrain_moment(moment)

    # -- predict_moment / mean_param_to_moment ------------------------------

    def predict_moment(self, loc: Array, noise_moment: Array) -> Array:
        """See base class.

        Returns the full mean parameter ``E[T(z)]``.

        * rank 0: ``[loc, loc² + Q_diag]`` (2D)
        * rank > 0: ``[loc, (loc locᵀ + Σ)_flat]`` (D + D²)
        """
        if self._rank == 0:
            _, cov_diag = jnp.split(noise_moment, 2)
            return jnp.concatenate((loc, loc ** 2 + cov_diag))

        _, cov = self.moment_to_canon(noise_moment)
        second = jnp.outer(loc, loc) + cov
        return jnp.concatenate((loc, second.flatten()))

    def mean_param_to_moment(self, mean_param: Array) -> Array:
        """See base class.

        Converts full mean parameter back to the structured moment.

        * rank 0: ``[m, s] → [m, s − m²]``
        * rank > 0: extract Σ, eigendecompose into diag + factor
        """
        d = self._dim
        if self._rank == 0:
            mean, second = jnp.split(mean_param, 2)
            return jnp.concatenate((mean, second - mean ** 2))

        mean = mean_param[:d]
        second = jnp.reshape(mean_param[d:], (d, d))
        cov = second - jnp.outer(mean, mean)
        cov_diag, cov_factor = self._decompose_cov(cov)
        return self._pack_moment(mean, cov_diag, cov_factor)
