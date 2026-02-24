"""
Exponential-family variational distributions for XFADS.

This module provides a unified multivariate normal (MVN) distribution with
natural and moment parameterizations for variational inference in XFADS.
The covariance structure is ``diag(d) + U U^T`` with configurable rank,
subsuming diagonal, low-rank, and full-covariance cases.
"""

import math

from jax import Array
from jax import numpy as jnp
from jax import random as jrnd
from tensorflow_probability.substrates.jax import distributions as tfd

from .base import Approx
from .constraints import _EPS, constrain_positive, unconstrain_positive


def damping_inv(a: Array, damping: float = _EPS) -> Array:
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

    Subclass and set ``_rank`` to select a covariance structure:

    * ``_rank = 0``: diagonal covariance (2D parameters).
    * ``_rank > 0``: diag + rank-r low-rank factor.
    * ``_rank = -1``: full rank (rank = D at runtime).

    Moment layout : ``[loc(D), cov_diag(D), cov_factor(D×r)]`` = D(2+r).
    Natural layout: ``[η₁(D), η₂(D)]`` (rank 0) or
    ``[η₁(D), η₂_flat(D²)]`` (rank > 0).

    Notes
    -----
    Uses ``tfd.MultivariateNormalDiagPlusLowRankCovariance`` as backend
    for sampling (rank > 0) and
    ``tfd.MultivariateNormalFullCovariance`` for KL computation.
    """

    _rank: int = 0

    # -- helpers -------------------------------------------------------------

    @classmethod
    def _effective_rank(cls, state_dim: int) -> int:
        """Resolve sentinel ``-1`` to *state_dim*."""
        return state_dim if cls._rank == -1 else cls._rank

    @classmethod
    def _dim_from_natural(cls, n: int) -> int:
        """Recover *D* from a natural-parameter array size."""
        if cls._rank == 0:
            return n // 2
        # D + D² → floor(sqrt(n)) = D
        return int(math.sqrt(n))

    @classmethod
    def _dim_from_moment(cls, n: int) -> int:
        """Recover *D* from a structured-moment array size."""
        if cls._rank == 0:
            return n // 2
        if cls._rank == -1:
            # D*(2+D) = D²+2D  →  D = sqrt(n+1) − 1
            return int(math.sqrt(n + 1)) - 1
        return n // (2 + cls._rank)

    @classmethod
    def _decompose_cov(cls, sigma: Array, d: int, r: int) -> tuple[Array, Array]:
        """Decompose Σ into ``diag(cov_diag) + cov_factor @ cov_factor.T``.

        The off-diagonal matrix ``Σ − diag(diag(Σ))`` is eigendecomposed
        and the top-*r* positive eigenvalues/vectors become the low-rank
        factor.  The diagonal residual is then ``diag(Σ) − diag(U Uᵀ)``.

        Parameters
        ----------
        sigma : Array, shape (d, d)
            Symmetric PSD covariance matrix.
        d : int
            State dimension.
        r : int
            Target rank for the low-rank factor.

        Returns
        -------
        cov_diag : Array, shape (d,)
            Positive diagonal residual.
        cov_factor : Array, shape (d, r)
            Low-rank factor.
        """
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

    @classmethod
    def _split_moment(cls, moment: Array) -> tuple[Array, Array, Array]:
        """Unpack structured moment ``[loc, cov_diag, cov_factor_flat]``.

        Returns
        -------
        loc : Array, shape (D,)
        cov_diag : Array, shape (D,)
        cov_factor : Array, shape (D, r)
        """
        d = cls._dim_from_moment(jnp.size(moment))
        r = cls._effective_rank(d)
        loc = moment[:d]
        cov_diag = moment[d : 2 * d]
        cov_factor = moment[2 * d :].reshape(d, r)
        return loc, cov_diag, cov_factor

    @classmethod
    def _build_cov(cls, cov_diag: Array, cov_factor: Array) -> Array:
        """Materialize full covariance from structured parts."""
        return jnp.diag(cov_diag) + cov_factor @ cov_factor.T

    @classmethod
    def _pack_moment(cls, loc: Array, cov_diag: Array, cov_factor: Array) -> Array:
        """Concatenate structured moment."""
        return jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    # -- Approx interface ----------------------------------------------------

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        """See base class. Natural parameter size."""
        if cls._rank == 0:
            return 2 * state_dim
        return state_dim + state_dim * state_dim

    @classmethod
    def moment_size(cls, state_dim: int) -> int:
        """See base class. Moment parameter size: D(2 + r)."""
        r = cls._effective_rank(state_dim)
        return state_dim * (2 + r)

    # -- natural ↔ moment ---------------------------------------------------

    @classmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        """See base class."""
        if cls._rank == 0:
            eta1, eta2 = jnp.split(natural, 2)
            cov_diag = -0.5 / eta2
            loc = -0.5 * eta1 / eta2
            return jnp.concatenate((loc, cov_diag))

        d = cls._dim_from_natural(jnp.size(natural))
        r = cls._effective_rank(d)
        eta1, eta2_flat = jnp.split(natural, [d])
        P = -2.0 * jnp.reshape(eta2_flat, (d, d))
        loc = jnp.linalg.solve(P, eta1)
        sigma = damping_inv(P)
        cov_diag, cov_factor = cls._decompose_cov(sigma, d, r)
        return cls._pack_moment(loc, cov_diag, cov_factor)

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        """See base class.

        Notes
        -----
        A minimum floor of ``_EPS`` is applied to the covariance
        diagonal before inversion to prevent extreme natural
        parameters.
        """
        if cls._rank == 0:
            mean, cov_diag = jnp.split(moment, 2)
            cov_diag = jnp.maximum(cov_diag, _EPS)
            eta2 = -0.5 / cov_diag
            eta1 = mean / cov_diag
            return jnp.concatenate((eta1, eta2))

        loc, cov_diag, cov_factor = cls._split_moment(moment)
        sigma = cls._build_cov(jnp.maximum(cov_diag, _EPS), cov_factor)
        P = damping_inv(sigma)
        eta1 = P @ loc
        eta2 = -0.5 * P
        return jnp.concatenate((eta1, eta2.flatten()))

    # -- sampling -----------------------------------------------------------

    @classmethod
    def sample_by_moment(
        cls, key: Array, moment: Array, mc_size: int | None = None
    ) -> Array:
        """See base class."""
        if cls._rank == 0:
            mean, cov_diag = jnp.split(moment, 2)
            std = jnp.sqrt(jnp.maximum(cov_diag, _EPS))
            shape = mean.shape if mc_size is None else (mc_size,) + mean.shape
            return mean + std * jrnd.normal(key, shape)

        loc, cov_diag, cov_factor = cls._split_moment(moment)
        dist = tfd.MultivariateNormalDiagPlusLowRankCovariance(
            loc=loc,
            cov_diag_factor=jnp.maximum(cov_diag, _EPS),
            cov_perturb_factor=cov_factor,
        )
        return dist.sample(mc_size, seed=key)

    # -- KL divergence ------------------------------------------------------

    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Array:
        """See base class.

        Uses ``MultivariateNormalFullCovariance`` for all ranks to
        ensure KL is always registered.
        """
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        if cls._rank == 0:
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

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        """See base class.

        Returns
        -------
        mean : Array, shape (D,)
        cov : Array
            Diagonal vector (D,) for rank 0, full matrix (D, D) otherwise.
        """
        if cls._rank == 0:
            mean, cov_diag = jnp.split(moment, 2, -1)
            return mean, cov_diag

        loc, cov_diag, cov_factor = cls._split_moment(moment)
        cov = cls._build_cov(cov_diag, cov_factor)
        return loc, cov

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        """See base class.

        Parameters
        ----------
        mean : Array, shape (D,)
        cov : Array
            Diagonal vector (D,) for rank 0, full matrix (D, D) otherwise.
        """
        if cls._rank == 0:
            return jnp.concatenate((mean, cov))

        d = jnp.size(mean)
        r = cls._effective_rank(d)
        cov_diag, cov_factor = cls._decompose_cov(cov, d, r)
        return cls._pack_moment(mean, cov_diag, cov_factor)

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        """See base class."""
        if cls._rank == 0:
            return jnp.diag(cov)
        return cov

    # -- constrain / unconstrain --------------------------------------------

    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        """See base class.

        Applies ``softplus`` to the diagonal covariance entries;
        the low-rank factor (if any) is left unconstrained.
        """
        if cls._rank == 0:
            loc, v = jnp.split(unconstrained, 2)
            return jnp.concatenate((loc, constrain_positive(v)))

        d = cls._dim_from_moment(jnp.size(unconstrained))
        loc = unconstrained[:d]
        diag_unc = unconstrained[d : 2 * d]
        factor = unconstrained[2 * d :]
        return jnp.concatenate((loc, constrain_positive(diag_unc), factor))

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        """See base class."""
        if cls._rank == 0:
            n1, n2 = jnp.split(unconstrained, 2)
            return jnp.concatenate((n1, -constrain_positive(n2)))

        d = cls._dim_from_natural(jnp.size(unconstrained))
        nat1, flat_L = jnp.split(unconstrained, [d])
        L = jnp.tril(jnp.reshape(flat_L, (d, d)))
        return jnp.concatenate((nat1, -(L @ L.T).flatten()))

    @classmethod
    def unconstrain_natural(cls, natural: Array) -> Array:
        """See base class."""
        if cls._rank == 0:
            n1, n2 = jnp.split(natural, 2)
            return jnp.concatenate((n1, unconstrain_positive(-n2)))

        d = cls._dim_from_natural(jnp.size(natural))
        nat1, nat2_flat = jnp.split(natural, [d])
        neg_nat2 = jnp.reshape(-nat2_flat, (d, d))
        L = jnp.linalg.cholesky(neg_nat2 + _EPS * jnp.eye(d))
        return jnp.concatenate((nat1, L.flatten()))

    @classmethod
    def unconstrain_moment(cls, moment: Array) -> Array:
        """See base class.

        Applies ``softplus_inverse`` to the diagonal covariance entries;
        the low-rank factor is left as-is.
        """
        if cls._rank == 0:
            loc, v = jnp.split(moment, 2)
            return jnp.concatenate((loc, unconstrain_positive(v)))

        d = cls._dim_from_moment(jnp.size(moment))
        loc = moment[:d]
        cov_diag = moment[d : 2 * d]
        factor = moment[2 * d :]
        return jnp.concatenate((loc, unconstrain_positive(cov_diag), factor))

    # -- prior / noise ------------------------------------------------------

    @classmethod
    def prior_natural(cls, state_dim: int) -> Array:
        """See base class. Returns N(0, I) in natural form."""
        if cls._rank == 0:
            eta1 = jnp.zeros(state_dim)
            eta2 = jnp.full(state_dim, -0.5)
            return jnp.concatenate((eta1, eta2))

        eta1 = jnp.zeros(state_dim)
        eta2 = -0.5 * jnp.eye(state_dim)
        return jnp.concatenate((eta1, eta2.flatten()))

    @classmethod
    def init_noise(cls, scale: float, state_dim: int) -> Array:
        """See base class. Isotropic noise N(0, scale·I).

        All variance goes into ``cov_diag``; the low-rank factor
        is initialised to zero.
        """
        r = cls._effective_rank(state_dim)
        loc = jnp.zeros(state_dim)
        cov_diag = jnp.full(state_dim, scale)
        cov_factor = jnp.zeros(state_dim * r)
        moment = jnp.concatenate((loc, cov_diag, cov_factor))
        return cls.unconstrain_moment(moment)

    # -- predict_moment / mean_param_to_moment ------------------------------

    @classmethod
    def predict_moment(cls, loc: Array, noise_moment: Array) -> Array:
        """See base class.

        Returns the full mean parameter ``E[T(z)]``.

        * rank 0: ``[loc, loc² + Q_diag]`` (2D)
        * rank > 0: ``[loc, (loc locᵀ + Σ)_flat]`` (D + D²)
        """
        if cls._rank == 0:
            _, cov_diag = jnp.split(noise_moment, 2)
            return jnp.concatenate((loc, loc ** 2 + cov_diag))

        _, cov = cls.moment_to_canon(noise_moment)
        second = jnp.outer(loc, loc) + cov
        return jnp.concatenate((loc, second.flatten()))

    @classmethod
    def mean_param_to_moment(cls, mean_param: Array) -> Array:
        """See base class.

        Converts full mean parameter back to the structured moment.

        * rank 0: ``[m, s] → [m, s − m²]``
        * rank > 0: extract Σ, eigendecompose into diag + factor
        """
        if cls._rank == 0:
            mean, second = jnp.split(mean_param, 2)
            return jnp.concatenate((mean, second - mean ** 2))

        d = cls._dim_from_natural(jnp.size(mean_param))
        r = cls._effective_rank(d)
        mean = mean_param[:d]
        second = jnp.reshape(mean_param[d:], (d, d))
        cov = second - jnp.outer(mean, mean)
        cov_diag, cov_factor = cls._decompose_cov(cov, d, r)
        return cls._pack_moment(mean, cov_diag, cov_factor)


# ---------------------------------------------------------------------------
# Backward-compatible registered aliases
# ---------------------------------------------------------------------------


class DiagMVN(MVN):
    """Diagonal covariance MVN (rank 0). Parameter size: 2D."""

    _rank = 0


class FullMVN(MVN):
    """Full covariance MVN (rank = D). Parameter size: D + D²."""

    _rank = -1
