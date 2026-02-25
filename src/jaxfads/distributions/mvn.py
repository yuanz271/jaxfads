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
`Approx` interface. Callers stay agnostic by using `param_size(...)` and the
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

    This class supports both:

    - ``structure='full'``: full-covariance Gaussian exponential family
      (moment size ``D + D²``)
    - ``structure='diag'``: diagonal Gaussian exponential family
      (moment size ``2D``)

    Parameters
    ----------
    dim : int
        State dimensionality.
    structure : {'full', 'diag'}
        Which Gaussian exponential-family sufficient-stat layout to use.

    Natural layout
    --------------
    ``structure='full'`` stores flat natural parameters as ``[h, J_flat]``:

    - ``h`` has shape ``(D,)``
    - ``J`` has shape ``(D, D)`` and is precision-like

    ``structure='diag'`` stores flat natural parameters as ``[h, j]``:

    - ``h`` has shape ``(D,)``
    - ``j`` has shape ``(D,)`` and is the diagonal precision-like term

    Moment layout
    ------------
    ``structure='full'`` stores flat moment parameters as ``[m, M2_flat]``:

    - ``m = E[z]`` has shape ``(D,)``
    - ``M2 = E[-½ zzᵀ]`` has shape ``(D, D)``

    ``structure='diag'`` stores flat moment parameters as ``[m, t2]``:

    - ``m = E[z]`` has shape ``(D,)``
    - ``t2 = E[-½ (z ⊙ z)]`` has shape ``(D,)``

    Notes
    -----
    `unpack(moment)` always returns a full covariance matrix, even for
    ``structure='diag'``.
    """

    def __init__(self, dim: int, *, structure: Literal["full", "diag"] = "full"):
        self._dim = int(dim)
        if structure not in ("full", "diag"):
            raise ValueError("structure must be one of {'full', 'diag'}")
        self._structure: Literal["full", "diag"] = structure

    # ---------------------------------------------------------------------
    # sizes
    # ---------------------------------------------------------------------

    def param_size(self, state_dim: int) -> int:
        """See base class."""
        d = self._dim
        if self._structure == "full":
            return d + d * d
        return 2 * d

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _split_moment(self, moment: Array) -> tuple[Array, Array]:
        d = self._dim
        m = moment[:d]
        if self._structure == "full":
            t2 = jnp.reshape(moment[d:], (d, d))
        else:
            t2 = moment[d:]
        return m, t2

    def _split_natural(self, natural: Array) -> tuple[Array, Array]:
        d = self._dim
        h = natural[:d]
        if self._structure == "full":
            J = jnp.reshape(natural[d:], (d, d))
            J = 0.5 * (J + J.T)
            return h, J
        j = natural[d:]
        return h, j

    def unpack(self, moment: Array) -> tuple[Array, Array]:
        """Extract (mean, covariance) from moment parameters."""
        m, t2 = self._split_moment(moment)

        if self._structure == "full":
            second = -2.0 * t2  # E[zz^T]
            cov = second - jnp.outer(m, m)
            cov = 0.5 * (cov + cov.T)
            cov = cov + _EPS * jnp.eye(self._dim, dtype=cov.dtype)
            return m, cov

        # diag structure: t2 = E[-1/2 z^2]
        ez2 = -2.0 * t2  # E[z^2]
        var = ez2 - m * m
        var = jnp.maximum(var, _EPS)
        return m, jnp.diag(var)

    def pack(self, mean: Array, cov: Array) -> Array:
        """Pack (mean, covariance) into moment parameters."""
        if self._structure == "full":
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
        h, J_or_j = self._split_natural(natural)

        if self._structure == "full":
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

        if self._structure == "full":
            cov = cov + _EPS * jnp.eye(self._dim, dtype=cov.dtype)
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

    def sample_by_moment(self, key: Array, moment: Array, mc_size: int) -> Array:
        """See base class."""
        mean, cov = self.unpack(moment)

        if self._structure == "diag":
            var = jnp.diag(cov)
            scale = jnp.sqrt(var)
            dist = tfd.MultivariateNormalDiag(mean, scale_diag=scale)
        else:
            dist = tfd.MultivariateNormalFullCovariance(mean, cov)

        return dist.sample(mc_size, seed=key)

    def kl(self, moment1: Array, moment2: Array) -> Array:
        """See base class."""
        m1, cov1 = self.unpack(moment1)
        m2, cov2 = self.unpack(moment2)

        if self._structure == "diag":
            s1 = jnp.sqrt(jnp.diag(cov1))
            s2 = jnp.sqrt(jnp.diag(cov2))
            p = tfd.MultivariateNormalDiag(m1, scale_diag=s1)
            q = tfd.MultivariateNormalDiag(m2, scale_diag=s2)
        else:
            p = tfd.MultivariateNormalFullCovariance(m1, cov1)
            q = tfd.MultivariateNormalFullCovariance(m2, cov2)

        return tfd.kl_divergence(p, q, allow_nan_stats=False)

    # ---------------------------------------------------------------------
    # free ↔ canon
    # ---------------------------------------------------------------------

    def _split_free(self, free: Array) -> tuple[Array, Array]:
        d = self._dim
        loc = free[:d]
        if self._structure == "full":
            chol_free = jnp.reshape(free[d:], (d, d))
        else:
            chol_free = free[d:]
        return loc, chol_free

    def free_to_canon(self, free: Array) -> MVNParam:
        """See base class.

        Parameters
        ----------
        free : Array
            Flat unconstrained parameters.

            - ``structure='full'``: ``[loc, chol_free_flat]`` with ``chol_free``
              reshaped to ``(D, D)``.
            - ``structure='diag'``: ``[loc, chol_diag_free]`` with
              ``chol_diag_free`` shape ``(D,)``.
        """
        loc, chol_free = self._split_free(free)

        if self._structure == "full":
            chol = _constrain_chol_full(chol_free)
        else:
            chol = _constrain_chol_diag(chol_free)

        return MVNParam(loc=loc, chol=chol)

    def canon_to_free(self, canon: MVNParam) -> Array:
        """See base class."""
        if self._structure == "full":
            chol_free = _unconstrain_chol_full(canon.chol)
            return jnp.concatenate((canon.loc, chol_free.ravel()))

        chol_diag_free = _unconstrain_chol_diag(canon.chol)
        return jnp.concatenate((canon.loc, chol_diag_free))

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

        if self._structure == "full":
            cov = cov + _EPS * jnp.eye(self._dim, dtype=cov.dtype)
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
        d = self._dim
        loc_arr = jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), (d,))
        diag = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), (d,))
        diag = jnp.maximum(diag, _EPS)

        if self._structure == "full":
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
