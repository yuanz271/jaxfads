"""
Exponential-family variational distributions for XFADS.

This module provides a unified multivariate normal (MVN) distribution with
natural and mean parameterizations for variational inference in XFADS.
The covariance structure is ``diag(d) + U U^T`` with configurable rank,
subsuming diagonal and low-rank cases.
"""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax import random as jrnd
from tensorflow_probability.substrates.jax import distributions as tfd

from ..base import Approx
from ..constraints import _EPS, constrain_positive, unconstrain_positive


class MVNParam(NamedTuple):
    """Canon/free-form pytree for MVN parameters.

    Attributes
    ----------
    loc : Array, shape (D,)
        Location (or unconstrained location for free-form).
    cov_diag : Array, shape (D,)
        Diagonal covariance (or unconstrained for free-form).
    cov_factor : Array, shape (D, r)
        Low-rank factor (or unconstrained for free-form).
    """

    loc: Array
    cov_diag: Array
    cov_factor: Array


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
        """Unpack canon mean ``[loc, cov_diag, cov_factor_flat]``.

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
        """Materialize full covariance from canon parts."""
        return jnp.diag(cov_diag) + cov_factor @ cov_factor.T

    def _pack_mean(self, loc: Array, cov_diag: Array, cov_factor: Array) -> Array:
        """Concatenate canon mean."""
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

    # -- natural ↔ moment ------------------------------------------------------

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
        return self._pack_mean(loc, cov_diag, cov_factor)

    def moment_to_natural(self, moment: Array) -> Array:
        """See base class.

        Notes
        -----
        A minimum floor of ``_EPS`` is applied to the covariance
        diagonal before inversion to prevent extreme natural
        parameters.
        """
        if self._rank == 0:
            loc, cov_diag = jnp.split(moment, 2)
            cov_diag = jnp.maximum(cov_diag, _EPS)
            eta2 = -0.5 / cov_diag
            eta1 = loc / cov_diag
            return jnp.concatenate((eta1, eta2))

        loc, cov_diag, cov_factor = self._split_mean(moment)
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
            loc, cov_diag = jnp.split(moment, 2)
            std = jnp.sqrt(jnp.maximum(cov_diag, _EPS))
            shape = loc.shape if mc_size is None else (mc_size,) + loc.shape
            return loc + std * jrnd.normal(key, shape)

        loc, cov_diag, cov_factor = self._split_mean(moment)
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
        m1, cov1 = self.unpack(moment1)
        m2, cov2 = self.unpack(moment2)
        if self._rank == 0:
            # unpack returns diagonal vectors
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

    def unpack(self, mean: Array) -> tuple[Array, Array]:
        """Unpack flat mean params into (loc, full_cov) tuple.

        Parameters
        ----------
        mean : Array
            Flat mean parameter vector.

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

    def pack(self, loc: Array, cov: Array) -> Array:
        """Pack (loc, cov) into flat mean params.

        Convenience method — accepts the user-friendly ``(loc, cov)``
        form where ``cov`` is a diagonal vector (rank 0) or full
        matrix (rank > 0).

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

    def free_to_canon(self, free: MVNParam) -> MVNParam:
        """See base class.

        Applies ``softplus`` to the diagonal covariance entries;
        loc and low-rank factor are left as-is.
        """
        return MVNParam(
            loc=free.loc,
            cov_diag=constrain_positive(free.cov_diag),
            cov_factor=free.cov_factor,
        )

    def canon_to_natural(self, canon: MVNParam) -> Array:
        """MVN-specific (not on ABC).

        Delegates to :meth:`moment_to_natural` via :meth:`canon_to_moment`.
        """
        return self.moment_to_natural(self.canon_to_moment(canon))

    def canon_to_moment(self, canon: MVNParam) -> Array:
        """See base class.

        Packs the pytree into a flat mean array.
        """
        return self._pack_mean(
            canon.loc, canon.cov_diag, canon.cov_factor
        )

    def moment_to_canon(self, mean: Array) -> MVNParam:
        """See base class.

        Unpacks a flat mean array into an MVNParam pytree.
        """
        loc, cov_diag, cov_factor = self._split_mean(mean)
        return MVNParam(loc=loc, cov_diag=cov_diag, cov_factor=cov_factor)

    def canon_to_free(self, canon: MVNParam) -> MVNParam:
        """See base class.

        Applies ``softplus_inverse`` to the diagonal covariance entries;
        loc and low-rank factor are left as-is.
        """
        return MVNParam(
            loc=canon.loc,
            cov_diag=unconstrain_positive(canon.cov_diag),
            cov_factor=canon.cov_factor,
        )

    # -- free_from_kw -----------------------------------------------------

    def free_from_kw(
        self, *, loc: float | list[float] = 0.0, scale: float | list[float] = 1.0
    ) -> MVNParam:
        """See base class.

        Creates free-form parameters for N(loc, diag(scale)).

        Parameters
        ----------
        loc : float or list[float]
            Mean value(s). Scalar is broadcast to all dimensions.
            List must have length ``dim``.
        scale : float or list[float]
            Diagonal covariance value(s). Scalar is broadcast to all
            dimensions. List must have length ``dim``.

        Returns
        -------
        MVNParam
            Free-form parameter pytree.
        """
        d, r = self._dim, self._rank
        loc_arr = jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), (d,))
        cov_diag_arr = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), (d,))
        cov_factor_arr = jnp.zeros((d, r))
        canon = MVNParam(loc=loc_arr, cov_diag=cov_diag_arr, cov_factor=cov_factor_arr)
        return self.canon_to_free(canon)

    # -- predict_moment --------------------------------------------------------

    def predict_moment(self, z: Array, noise: Array) -> Array:
        """See base class.

        Returns moment parameters (expected sufficient statistics) for a single
        transition distribution.

        Under the XFADS paper convention, the Gaussian sufficient
        statistics are:

        * ``T₁(z_t) = z_t``
        * ``T₂(z_t) = -½ z_t z_tᵀ``

        With transition ``z_t ~ N(μ, Q)`` (here ``μ = z``), the conditional
        moments are:

        * ``E[T₁ | z] = μ``
        * ``E[T₂ | z] = -½ (Q + μ μᵀ)``

        Flat layout returned by this method (moment parameters):

        * rank 0: ``[μ, -½(μ² + Q_diag)]``
        * rank > 0: ``[μ, vec(-½(Q + μ μᵀ))]``
        """
        if self._rank == 0:
            _, q_diag = jnp.split(noise, 2)
            return jnp.concatenate((z, -0.5 * (z**2 + q_diag)))

        _, q = self.unpack(noise)
        second = (-0.5 * (jnp.outer(z, z) + q)).ravel()
        return jnp.concatenate((z, second))

    def from_sufficient_stats(self, stats: Array) -> Array:
        """See base class.

        Converts moment parameters ``[E[z], E[-½ zzᵀ]]`` into the storage mean
        format ``[loc, cov_diag, cov_factor]``.

        Internally recovers the second moment ``E[zzᵀ] = -2·E[-½ zzᵀ]`` and
        then computes covariance as ``Σ = E[zzᵀ] - E[z]E[z]ᵀ``.
        """
        d = self._dim
        if self._rank == 0:
            loc, t2 = jnp.split(stats, 2)
            second = -2.0 * t2
            return jnp.concatenate((loc, second - loc**2))

        loc = stats[:d]
        t2 = jnp.reshape(stats[d:], (d, d))
        second = -2.0 * t2
        cov = second - jnp.outer(loc, loc)
        cov_diag, cov_factor = self._decompose_cov(cov)
        return self._pack_mean(loc, cov_diag, cov_factor)

    def _expanded_to_mean(self, expanded: Array) -> Array:
        """Deprecated internal helper.

        Kept for backward-compat with older commits; prefer
        :meth:`from_sufficient_stats`.
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
