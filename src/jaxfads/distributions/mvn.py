"""
Exponential-family variational distributions for XFADS.

This module provides a unified multivariate normal (MVN) distribution with
natural and mean parameterizations for variational inference in XFADS.
The covariance structure is ``diag(d) + U U^T`` with configurable rank,
subsuming diagonal and low-rank cases.
"""

import jax
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

    Mean layout : ``[loc(D), cov_diag(D), cov_factor(D×r)]`` = D(2+r).
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

    def _split_mean(self, mean: Array) -> tuple[Array, Array, Array]:
        """Unpack structured mean ``[loc, cov_diag, cov_factor_flat]``.

        Returns
        -------
        loc : Array, shape (D,)
        cov_diag : Array, shape (D,)
        cov_factor : Array, shape (D, r)
        """
        d, r = self._dim, self._rank
        loc = mean[:d]
        cov_diag = mean[d : 2 * d]
        cov_factor = mean[2 * d :].reshape(d, r)
        return loc, cov_diag, cov_factor

    def _build_cov(self, cov_diag: Array, cov_factor: Array) -> Array:
        """Materialize full covariance from structured parts."""
        return jnp.diag(cov_diag) + cov_factor @ cov_factor.T

    def _pack_mean(self, loc: Array, cov_diag: Array, cov_factor: Array) -> Array:
        """Concatenate structured mean."""
        return jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    # -- Approx interface ----------------------------------------------------

    def param_size(self, state_dim: int) -> int:
        """See base class. Natural parameter size."""
        d = self._dim
        if self._rank == 0:
            return 2 * d
        return d + d * d

    def mean_size(self, state_dim: int) -> int:
        """See base class. Mean parameter size: D(2 + r)."""
        return self._dim * (2 + self._rank)

    # -- natural ↔ mean ------------------------------------------------------

    def natural_to_mean(self, natural: Array) -> Array:
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
        return self._pack_mean(loc, cov_diag, cov_factor)

    def mean_to_natural(self, mean: Array) -> Array:
        """See base class.

        Notes
        -----
        A minimum floor of ``_EPS`` is applied to the covariance
        diagonal before inversion to prevent extreme natural
        parameters.
        """
        if self._rank == 0:
            loc, cov_diag = jnp.split(mean, 2)
            cov_diag = jnp.maximum(cov_diag, _EPS)
            eta2 = -0.5 / cov_diag
            eta1 = loc / cov_diag
            return jnp.concatenate((eta1, eta2))

        loc, cov_diag, cov_factor = self._split_mean(mean)
        sigma = self._build_cov(jnp.maximum(cov_diag, _EPS), cov_factor)
        P = _damping_inv(sigma)
        eta1 = P @ loc
        eta2 = -0.5 * P
        return jnp.concatenate((eta1, eta2.flatten()))

    # -- sampling -----------------------------------------------------------

    def sample_by_mean(
        self, key: Array, mean: Array, mc_size: int | None = None
    ) -> Array:
        """See base class."""
        if self._rank == 0:
            loc, cov_diag = jnp.split(mean, 2)
            std = jnp.sqrt(jnp.maximum(cov_diag, _EPS))
            shape = loc.shape if mc_size is None else (mc_size,) + loc.shape
            return loc + std * jrnd.normal(key, shape)

        loc, cov_diag, cov_factor = self._split_mean(mean)
        dist = tfd.MultivariateNormalDiagPlusLowRankCovariance(
            loc=loc,
            cov_diag_factor=jnp.maximum(cov_diag, _EPS),
            cov_perturb_factor=cov_factor,
        )
        return dist.sample(mc_size, seed=key)

    # -- KL divergence ------------------------------------------------------

    def kl(self, mean1: Array, mean2: Array) -> Array:
        """See base class.

        Uses ``MultivariateNormalFullCovariance`` for all ranks to
        ensure KL is always registered.
        """
        m1, cov1 = self.mean_to_canon(mean1)
        m2, cov2 = self.mean_to_canon(mean2)
        if self._rank == 0:
            # mean_to_canon returns diagonal vectors
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

    def mean_to_canon(self, mean: Array) -> tuple[Array, Array]:
        """See base class.

        Returns
        -------
        loc : Array, shape (D,)
        cov : Array
            Diagonal vector (D,) for rank 0, full matrix (D, D) otherwise.
        """
        if self._rank == 0:
            loc, cov_diag = jnp.split(mean, 2, -1)
            return loc, cov_diag

        loc, cov_diag, cov_factor = self._split_mean(mean)
        cov = self._build_cov(cov_diag, cov_factor)
        return loc, cov

    def canon_to_mean(self, loc: Array, cov: Array) -> Array:
        """See base class.

        Parameters
        ----------
        loc : Array, shape (D,)
        cov : Array
            Diagonal vector (D,) for rank 0, full matrix (D, D) otherwise.
        """
        if self._rank == 0:
            return jnp.concatenate((loc, cov))

        cov_diag, cov_factor = self._decompose_cov(cov)
        return self._pack_mean(loc, cov_diag, cov_factor)

    def full_cov(self, cov: Array) -> Array:
        """See base class."""
        if self._rank == 0:
            return jnp.diag(cov)
        return cov

    # -- constrain / unconstrain --------------------------------------------

    def to_structured(self, unconstrained: Array) -> Array:
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

    def structured_to_natural(self, structured: Array) -> Array:
        """See base class.

        Delegates to :meth:`mean_to_natural` because for MVN the
        structured parameters coincide with the mean parameters.
        """
        return self.mean_to_natural(structured)

    def structured_to_mean(self, structured: Array) -> Array:
        """See base class.

        Identity for MVN — structured parameters are mean parameters.
        """
        return structured

    def to_free(self, mean: Array) -> Array:
        """See base class.

        Applies ``softplus_inverse`` to the diagonal covariance entries;
        the low-rank factor is left as-is.
        """
        d = self._dim
        if self._rank == 0:
            loc, v = jnp.split(mean, 2)
            return jnp.concatenate((loc, unconstrain_positive(v)))

        loc = mean[:d]
        cov_diag = mean[d : 2 * d]
        factor = mean[2 * d :]
        return jnp.concatenate((loc, unconstrain_positive(cov_diag), factor))

    # -- param_from_conf -----------------------------------------------------

    def param_from_conf(self, *, scale: float = 1.0) -> Array:
        """See base class.

        Creates free-form parameters for isotropic N(0, scale·I).

        Parameters
        ----------
        scale : float
            Diagonal covariance value (default 1.0).

        Returns
        -------
        Array
            Free-form parameter array.
        """
        d, r = self._dim, self._rank
        loc = jnp.zeros(d)
        cov_diag = jnp.full(d, scale)
        cov_factor = jnp.zeros(d * r)
        structured = jnp.concatenate((loc, cov_diag, cov_factor))
        return self.to_free(structured)

    # -- predict_mean --------------------------------------------------------

    def predict_mean(self, locs: Array, noise_mean: Array) -> Array:
        """See base class.

        Computes per-sample expanded mean parameters ``E[T(z)]``,
        averages in expanded space (where linearity holds), then
        converts back to the structured mean format.

        Expanded mean parameter layout:

        * rank 0: ``[loc, loc² + Q_diag]`` (2D per sample)
        * rank > 0: ``[loc, (loc locᵀ + Σ)_flat]`` (D + D² per sample)
        """
        if self._rank == 0:
            _, cov_diag = jnp.split(noise_mean, 2)
            expanded = jnp.concatenate(
                (locs, locs ** 2 + cov_diag), axis=-1
            )
        else:
            _, cov = self.mean_to_canon(noise_mean)
            outers = jax.vmap(lambda x: jnp.outer(x, x))(locs)
            seconds = (outers + cov).reshape(locs.shape[0], -1)
            expanded = jnp.concatenate((locs, seconds), axis=-1)

        # Non-finite safe averaging; return NaN when all samples are invalid
        valid = jnp.all(jnp.isfinite(expanded), axis=-1)  # (N,)
        safe = jnp.where(valid[:, None], expanded, 0.0)
        n_valid = jnp.sum(valid)
        avg = jnp.where(
            n_valid > 0,
            jnp.sum(safe, axis=0) / n_valid,
            jnp.full(expanded.shape[-1:], jnp.nan),
        )

        return self._expanded_to_mean(avg)

    def _expanded_to_mean(self, expanded: Array) -> Array:
        """Convert averaged expanded mean parameter to structured mean.

        * rank 0: ``[m, s] → [m, s − m²]``
        * rank > 0: extract Σ, eigendecompose into diag + factor
        """
        d = self._dim
        if self._rank == 0:
            loc, second = jnp.split(expanded, 2)
            return jnp.concatenate((loc, second - loc ** 2))

        loc = expanded[:d]
        second = jnp.reshape(expanded[d:], (d, d))
        cov = second - jnp.outer(loc, loc)
        cov_diag, cov_factor = self._decompose_cov(cov)
        return self._pack_mean(loc, cov_diag, cov_factor)
