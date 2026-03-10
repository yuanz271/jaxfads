"""Multivariate normal (MVN) exponential-family approximation.

This implementation follows the XFADS paper convention for Gaussian sufficient
statistics in the **full-covariance** case:

- ``T₁(z) = z``
- ``T₂(z) = -½ z zᵀ``

So the *moment parameters* are the expected sufficient statistics
``μ = E[T(z)]``.

In addition to the full-covariance Gaussian exponential family, this module
also supports a **diagonal** Gaussian exponential family variant with
sufficient statistics:

- ``T₁(z) = z``
- ``T₂(z) = -½ (z ⊙ z)``

This yields a 2D moment/natural representation while keeping the same public
`Approx` interface. Callers stay agnostic by using `param_size()` and the
(flat) moment/natural conversions.

Notes
-----
`MVN.unpack(moment)` always returns a **full covariance matrix** (shape
``(D, D)``) for both variants, so likelihood code can remain uniform.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from jax import Array
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from ..base import Approx
from ..constraints import _EPS, constrain_positive, unconstrain_positive


class _Layout(NamedTuple):
    """Internal MVN layout helper.

    Centralizes the structure-dependent parsing/sizing logic for the flat
    natural/moment/free vectors.
    """

    dim: int
    structure: Literal["full", "diag"]

    @property
    def is_full(self) -> bool:
        return self.structure == "full"

    @property
    def is_diag(self) -> bool:
        return self.structure == "diag"

    def param_size(self) -> int:
        d = self.dim
        return (d + d * d) if self.is_full else (2 * d)

    def split_moment(self, moment: Array) -> tuple[Array, Array]:
        d = self.dim
        m = moment[:d]
        if self.is_full:
            t2 = jnp.reshape(moment[d:], (d, d))
        else:
            t2 = moment[d:]
        return m, t2

    def split_natural(self, natural: Array) -> tuple[Array, Array]:
        d = self.dim
        h = natural[:d]
        if self.is_full:
            J = jnp.reshape(natural[d:], (d, d))
            J = 0.5 * (J + J.T)
            return h, J
        j = natural[d:]
        return h, j

    def split_free(self, free: Array) -> tuple[Array, Array]:
        d = self.dim
        loc = free[:d]
        if self.is_full:
            chol_free = jnp.reshape(free[d:], (d, d))
        else:
            chol_free = free[d:]
        return loc, chol_free


class MVNParam(NamedTuple):
    """Canon/free-form pytree for MVN parameters.

    Attributes
    ----------
    loc : Array, shape (D,)
        Mean.
    chol : Array, shape (D, D)
        Lower-triangular Cholesky factor. For canon parameters, the diagonal is
        constrained to be positive.

    Notes
    -----
    Even in the diagonal MVN variant, we store `chol` as a dense ``(D, D)``
    matrix (with zero off-diagonals). This keeps the canon pytree uniform.
    """

    loc: Array
    chol: Array


def _damping_inv(a: Array, damping: float = _EPS) -> Array:
    """Inverse with diagonal damping, via solve."""
    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    return jnp.linalg.solve(a + damping * eye, eye)


def _constrain_chol_full(chol_free: Array) -> Array:
    """Map unconstrained lower-triangular matrix to valid Cholesky factor."""
    tril = jnp.tril(chol_free)
    diag = jnp.diag(tril)
    diag_pos = constrain_positive(diag)
    return tril - jnp.diag(diag) + jnp.diag(diag_pos)


def _unconstrain_chol_full(chol: Array) -> Array:
    """Inverse of :func:`_constrain_chol_full` (up to numerical roundoff)."""
    tril = jnp.tril(chol)
    diag = jnp.diag(tril)
    diag_free = unconstrain_positive(diag)
    return tril - jnp.diag(diag) + jnp.diag(diag_free)


def _constrain_chol_diag(diag_free: Array) -> Array:
    """Map unconstrained diagonal vector to a valid diagonal Cholesky factor."""
    diag_pos = constrain_positive(diag_free)
    return jnp.diag(diag_pos)


def _unconstrain_chol_diag(chol: Array) -> Array:
    """Inverse of :func:`_constrain_chol_diag` (up to numerical roundoff)."""
    diag = jnp.diag(chol)
    diag_free = unconstrain_positive(diag)
    return diag_free


class MVN(Approx):
    """Multivariate normal approximation.

    A single ``rank`` parameter controls both the encoder precision
    parameterization and the internal exponential-family layout:

    - ``rank=0``: diagonal EF layout — ``param_size = 2D``,
      ``free_size = 2D`` (fast path).
    - ``rank>0``: full EF layout — ``param_size = D + D²``,
      ``free_size = 2D + D·rank``.

    Encoder precision parameterization
    -----------------------------------
    The encoder's ``free_to_natural`` maps free parameters to an additive
    natural update whose precision part is::

        J = diag(softplus(d_free)) + L @ Lᵀ

    where ``d_free`` is a ``(D,)`` baseline precision vector and ``L`` is a
    ``(D, rank)`` low-rank factor.

    Parameters
    ----------
    dim : int
        State dimensionality.
    rank : int
        Rank of the off-diagonal precision factor ``L``.
        ``rank=0``: diagonal-only precision (efficient ``2D`` param layout).
        ``rank=D``: full-rank precision.

    Notes
    -----
    `unpack(moment)` always returns a full covariance matrix, even for
    ``rank=0``.
    """

    def __init__(self, dim: int, rank: int):
        dim = int(dim)
        rank = int(rank)

        structure: Literal["full", "diag"] = "diag" if rank == 0 else "full"
        self._layout = _Layout(dim=dim, structure=structure)
        self._rank = rank

    # ---------------------------------------------------------------------
    # sizes
    # ---------------------------------------------------------------------

    def param_size(self) -> int:
        """See base class."""
        return self._layout.param_size()

    def free_size(self) -> int:
        """Return encoder free-form output size.

        For ``structure='diag'`` (rank=0): ``2D`` (h + diag precision free).
        For ``structure='full'``: ``2D + D·rank`` (h + diag precision free + L).
        """
        d = self._layout.dim
        return 2 * d + d * self._rank

    def free_to_natural(self, free: Array) -> Array:
        """Convert encoder free-form output to additive natural update.

        Encoder outputs are mapped to precision via::

            J = diag(softplus(d_free)) + L @ Lᵀ

        where ``d_free`` parameterizes the diagonal baseline and ``L`` is
        a ``(D, rank)`` low-rank factor.

        For ``structure='diag'``, returns ``[h, softplus(d_free)]`` (size 2D).
        For ``structure='full'``, returns ``[h, J_flat]`` (size D + D²).
        """
        d = self._layout.dim
        h = free[:d]
        d_free = free[d : 2 * d]
        diag_prec = constrain_positive(d_free)

        if self._layout.is_diag:
            return jnp.concatenate((h, diag_prec))

        # full structure: J = diag(prec) + L @ L^T
        J = jnp.diag(diag_prec)
        if self._rank > 0:
            L = jnp.reshape(free[2 * d :], (d, self._rank))
            J = J + L @ L.T
            J = 0.5 * (J + J.T)
        return jnp.concatenate((h, J.ravel()))

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def unpack(self, moment: Array) -> tuple[Array, Array]:
        """Extract (mean, covariance) from moment parameters."""
        m, t2 = self._layout.split_moment(moment)

        if self._layout.is_full:
            second = -2.0 * t2  # E[zz^T]
            cov = second - jnp.outer(m, m)
            cov = 0.5 * (cov + cov.T)
            cov = cov + _EPS * jnp.eye(self._layout.dim, dtype=cov.dtype)
            return m, cov

        # diag structure: t2 = E[-1/2 z^2]
        ez2 = -2.0 * t2  # E[z^2]
        var = ez2 - m * m
        var = jnp.maximum(var, _EPS)
        return m, jnp.diag(var)

    def pack(self, mean: Array, cov: Array) -> Array:
        """Pack (mean, covariance) into moment parameters."""
        if self._layout.is_full:
            cov = 0.5 * (cov + cov.T)
            second = cov + jnp.outer(mean, mean)
            t2 = -0.5 * second
            return jnp.concatenate((mean, t2.ravel()))

        # diag structure: keep only diagonal variance
        if cov.ndim == 2:
            var = jnp.diag(cov)
        else:
            var = cov
        ez2 = var + mean * mean
        t2 = -0.5 * ez2
        return jnp.concatenate((mean, t2))

    # ---------------------------------------------------------------------
    # natural ↔ moment
    # ---------------------------------------------------------------------

    def natural_to_moment(self, natural: Array) -> Array:
        """See base class."""
        h, J_or_j = self._layout.split_natural(natural)

        if self._layout.is_full:
            J = J_or_j
            mean = jnp.linalg.solve(J, h)
            cov = _damping_inv(J)
            return self.pack(mean, cov)

        j = J_or_j
        j = jnp.maximum(j, _EPS)
        mean = h / j
        var = 1.0 / j
        return self.pack(mean, jnp.diag(var))

    def moment_to_natural(self, moment: Array) -> Array:
        """See base class."""
        mean, cov = self.unpack(moment)

        if self._layout.is_full:
            cov = cov + _EPS * jnp.eye(self._layout.dim, dtype=cov.dtype)
            J = _damping_inv(cov)
            h = J @ mean
            return jnp.concatenate((h, J.ravel()))

        var = jnp.diag(cov)
        var = jnp.maximum(var, _EPS)
        j = 1.0 / var
        h = j * mean
        return jnp.concatenate((h, j))

    # ---------------------------------------------------------------------
    # sampling / KL
    # ---------------------------------------------------------------------

    def _tfd_dist_from_moment(self, moment: Array):
        mean, cov = self.unpack(moment)

        if self._layout.is_diag:
            scale = jnp.sqrt(jnp.diag(cov))
            return tfd.MultivariateNormalDiag(mean, scale_diag=scale)

        return tfd.MultivariateNormalFullCovariance(mean, cov)

    def sample_by_moment(self, key: Array, moment: Array, mc_size: int) -> Array:
        """See base class."""
        return self._tfd_dist_from_moment(moment).sample(mc_size, seed=key)

    def kl(self, moment1: Array, moment2: Array) -> Array:
        """See base class."""
        p = self._tfd_dist_from_moment(moment1)
        q = self._tfd_dist_from_moment(moment2)
        return tfd.kl_divergence(p, q, allow_nan_stats=False)

    # ---------------------------------------------------------------------
    # free ↔ canon
    # ---------------------------------------------------------------------

    def free_to_canon(self, free: Array) -> MVNParam:
        """See base class.

        The free-form vector is interpreted as ``[loc, prec_chol_free]`` where
        ``prec_chol_free`` parameterizes the **precision** Cholesky factor via
        ``constrain_positive`` on the diagonal.  The covariance Cholesky is
        obtained by inverting.

        Parameters
        ----------
        free : Array
            Flat unconstrained parameters.

            - ``structure='full'``: ``[loc, prec_chol_free_flat]`` with
              ``prec_chol_free`` reshaped to ``(D, D)``.
            - ``structure='diag'``: ``[loc, prec_diag_free]`` with
              ``prec_diag_free`` shape ``(D,)``.
        """
        loc, chol_free = self._layout.split_free(free)
        d = self._layout.dim

        if self._layout.is_full:
            # free -> precision Cholesky -> invert -> covariance Cholesky
            prec_chol = _constrain_chol_full(chol_free)
            prec = prec_chol @ prec_chol.T
            cov = _damping_inv(prec)
            cov = 0.5 * (cov + cov.T)
            chol = jnp.linalg.cholesky(cov + _EPS * jnp.eye(d, dtype=cov.dtype))
        else:
            # free -> precision diagonal -> invert -> covariance Cholesky
            prec_diag = constrain_positive(chol_free)
            var = 1.0 / jnp.maximum(prec_diag, _EPS)
            chol = jnp.diag(jnp.sqrt(var))

        return MVNParam(loc=loc, chol=chol)

    def canon_to_free(self, canon: MVNParam) -> Array:
        """See base class."""
        d = self._layout.dim

        if self._layout.is_full:
            # covariance Cholesky -> precision -> precision Cholesky -> unconstrain
            cov = canon.chol @ canon.chol.T
            prec = _damping_inv(cov)
            prec = 0.5 * (prec + prec.T)
            prec_chol = jnp.linalg.cholesky(prec + _EPS * jnp.eye(d, dtype=prec.dtype))
            chol_free = _unconstrain_chol_full(prec_chol)
            return jnp.concatenate((canon.loc, chol_free.ravel()))

        # diagonal: covariance Cholesky -> precision diagonal -> unconstrain
        var = jnp.diag(canon.chol) ** 2
        prec_diag = 1.0 / jnp.maximum(var, _EPS)
        prec_free = unconstrain_positive(prec_diag)
        return jnp.concatenate((canon.loc, prec_free))

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

        if self._layout.is_full:
            cov = cov + _EPS * jnp.eye(self._layout.dim, dtype=cov.dtype)
            chol = jnp.linalg.cholesky(cov)
            return MVNParam(loc=mean, chol=chol)

        var = jnp.diag(cov)
        var = jnp.maximum(var, _EPS)
        chol_diag = jnp.sqrt(var)
        return MVNParam(loc=mean, chol=jnp.diag(chol_diag))

    # ---------------------------------------------------------------------
    # initialization
    # ---------------------------------------------------------------------

    def free_from_kw(
        self, *, loc: float | list[float] = 0.0, scale: float | list[float] = 1.0
    ) -> Array:
        """See base class.

        Creates free-form parameters for ``N(loc, diag(scale))``.
        """
        d = self._layout.dim
        loc_arr = jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), (d,))
        diag = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), (d,))
        diag = jnp.maximum(diag, _EPS)

        if self._layout.is_full:
            cov = jnp.diag(diag)
            chol = jnp.linalg.cholesky(cov + _EPS * jnp.eye(d, dtype=cov.dtype))
            canon = MVNParam(loc=loc_arr, chol=chol)
            return self.canon_to_free(canon)

        chol_diag = jnp.sqrt(diag)
        canon = MVNParam(loc=loc_arr, chol=jnp.diag(chol_diag))
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
