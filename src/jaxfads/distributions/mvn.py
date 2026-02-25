"""Multivariate normal (MVN) exponential-family approximation.

This implementation follows the XFADS paper convention for Gaussian sufficient
statistics:

- ``T₁(z) = z``
- ``T₂(z) = -½ z zᵀ``

So the *moment parameters* are the expected sufficient statistics
``μ = E[T(z)]``.

Notes
-----
This implementation currently uses a full-covariance MVN.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from ..base import Approx
from ..constraints import _EPS, constrain_positive, unconstrain_positive


class MVNParam(NamedTuple):
    """Canon/free-form pytree for MVN parameters.

    Attributes
    ----------
    loc : Array, shape (D,)
        Mean.
    chol : Array, shape (D, D)
        Lower-triangular Cholesky factor. For canon parameters, the diagonal is
        constrained to be positive.
    """

    loc: Array
    chol: Array


def _damping_inv(a: Array, damping: float = _EPS) -> Array:
    """Inverse with diagonal damping, via solve."""
    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    return jnp.linalg.solve(a + damping * eye, eye)


def _constrain_chol(chol_free: Array) -> Array:
    """Map unconstrained lower-triangular matrix to valid Cholesky factor."""
    tril = jnp.tril(chol_free)
    diag = jnp.diag(tril)
    diag_pos = constrain_positive(diag)
    return tril - jnp.diag(diag) + jnp.diag(diag_pos)


def _unconstrain_chol(chol: Array) -> Array:
    """Inverse of :func:`_constrain_chol` (up to numerical roundoff)."""
    tril = jnp.tril(chol)
    diag = jnp.diag(tril)
    diag_free = unconstrain_positive(diag)
    return tril - jnp.diag(diag) + jnp.diag(diag_free)


class MVN(Approx):
    """Full-covariance multivariate normal approximation.

    Parameters
    ----------
    dim : int
        State dimensionality.

    Natural layout
    --------------
    Flat natural parameters are stored as ``[h, J_flat]`` where:

    - ``h`` has shape ``(D,)``
    - ``J`` has shape ``(D, D)`` and represents the precision matrix

    This matches the paper's choice with ``T₂(z) = -½ zzᵀ`` where the second
    natural-parameter block is precision-like.

    Moment layout
    ------------
    Flat moment parameters are stored as ``[m, M2_flat]`` where:

    - ``m = E[z]`` has shape ``(D,)``
    - ``M2 = E[-½ zzᵀ]`` has shape ``(D, D)``

    Total moment size is ``D + D²``.
    """

    def __init__(self, dim: int):
        self._dim = int(dim)

    # ---------------------------------------------------------------------
    # sizes
    # ---------------------------------------------------------------------

    def param_size(self, state_dim: int) -> int:
        """See base class."""
        d = self._dim
        return d + d * d

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _split_moment(self, moment: Array) -> tuple[Array, Array]:
        d = self._dim
        m = moment[:d]
        t2 = jnp.reshape(moment[d:], (d, d))
        return m, t2

    def unpack(self, moment: Array) -> tuple[Array, Array]:
        """Extract (mean, covariance) from moment parameters.

        Parameters
        ----------
        moment : Array
            Flat moment vector ``[E[z], E[-½ zzᵀ]_flat]``.

        Returns
        -------
        mean : Array, shape (D,)
            ``E[z]``.
        cov : Array, shape (D, D)
            ``Cov(z)``.
        """
        m, t2 = self._split_moment(moment)
        second = -2.0 * t2
        cov = second - jnp.outer(m, m)
        cov = 0.5 * (cov + cov.T)
        return m, cov

    def pack(self, mean: Array, cov: Array) -> Array:
        """Pack (mean, covariance) into moment parameters."""
        cov = 0.5 * (cov + cov.T)
        second = cov + jnp.outer(mean, mean)
        t2 = -0.5 * second
        return jnp.concatenate((mean, t2.ravel()))

    # ---------------------------------------------------------------------
    # natural ↔ moment
    # ---------------------------------------------------------------------

    def natural_to_moment(self, natural: Array) -> Array:
        """See base class."""
        d = self._dim
        h, j_flat = jnp.split(natural, [d])
        J = jnp.reshape(j_flat, (d, d))
        # Symmetrize for numerical stability
        J = 0.5 * (J + J.T)
        mean = jnp.linalg.solve(J, h)
        cov = _damping_inv(J)
        return self.pack(mean, cov)

    def moment_to_natural(self, moment: Array) -> Array:
        """See base class."""
        mean, cov = self.unpack(moment)
        # Ensure PD-ish
        cov = cov + _EPS * jnp.eye(self._dim, dtype=cov.dtype)
        J = _damping_inv(cov)
        h = J @ mean
        return jnp.concatenate((h, J.ravel()))

    # ---------------------------------------------------------------------
    # sampling / KL
    # ---------------------------------------------------------------------

    def sample_by_moment(self, key: Array, moment: Array, mc_size: int) -> Array:
        """See base class."""
        mean, cov = self.unpack(moment)
        dist = tfd.MultivariateNormalFullCovariance(mean, cov)
        return dist.sample(mc_size, seed=key)

    def kl(self, moment1: Array, moment2: Array) -> Array:
        """See base class."""
        m1, cov1 = self.unpack(moment1)
        m2, cov2 = self.unpack(moment2)
        return tfd.kl_divergence(
            tfd.MultivariateNormalFullCovariance(m1, cov1),
            tfd.MultivariateNormalFullCovariance(m2, cov2),
            allow_nan_stats=False,
        )

    # ---------------------------------------------------------------------
    # free ↔ canon
    # ---------------------------------------------------------------------

    def free_to_canon(self, free: MVNParam) -> MVNParam:
        """See base class."""
        return MVNParam(loc=free.loc, chol=_constrain_chol(free.chol))

    def canon_to_free(self, canon: MVNParam) -> MVNParam:
        """See base class."""
        return MVNParam(loc=canon.loc, chol=_unconstrain_chol(canon.chol))

    # ---------------------------------------------------------------------
    # canon ↔ moment
    # ---------------------------------------------------------------------

    def canon_to_moment(self, canon: MVNParam) -> Array:
        """See base class."""
        cov = canon.chol @ canon.chol.T
        return self.pack(canon.loc, cov)

    def moment_to_canon(self, moment: Array) -> MVNParam:
        """See base class."""
        mean, cov = self.unpack(moment)
        cov = cov + _EPS * jnp.eye(self._dim, dtype=cov.dtype)
        chol = jnp.linalg.cholesky(cov)
        return MVNParam(loc=mean, chol=chol)

    # ---------------------------------------------------------------------
    # initialization
    # ---------------------------------------------------------------------

    def free_from_kw(
        self, *, loc: float | list[float] = 0.0, scale: float | list[float] = 1.0
    ) -> MVNParam:
        """See base class.

        Creates free-form parameters for ``N(loc, diag(scale))``.
        """
        d = self._dim
        loc_arr = jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), (d,))
        diag = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), (d,))
        cov = jnp.diag(diag)
        chol = jnp.linalg.cholesky(cov + _EPS * jnp.eye(d, dtype=cov.dtype))
        canon = MVNParam(loc=loc_arr, chol=chol)
        return self.canon_to_free(canon)

    # ---------------------------------------------------------------------
    # predictive moments
    # ---------------------------------------------------------------------

    def predictive_moment(self, z: Array, noise: Array) -> Array:
        """See base class.

        The transition is interpreted as:

        ``z_t | z_{t-1} ~ N(z, Q)``

        where ``Q`` is recovered from the provided noise moment parameters.
        """
        _, Q = self.unpack(noise)
        return self.pack(z, Q)
