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

import jax
from jax import Array
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from ..base import Approx
from ..constraints import _EPS, constrain_positive, unconstrain_positive
from ..noise import Noise


def _weighted_moments(zs: Array, weights: Array) -> tuple[Array, Array]:
    """Non-finite-safe weighted mean and covariance of a raw point set.

    ``MVN``'s own reduction of a generic ``(zs, weights)`` point set (as
    produced by ``core.propagate_transition_points``) into the sufficient
    statistic this Gaussian family needs -- called from ``MVN.
    transition_stat``, which ``core.py``'s recursions call polymorphically
    to build each pair's ``transition_stat`` (see ``core._site_filter``/
    ``nofilt``). Not shared with ``core.py``: reducing a point set to a
    mean/covariance pair is a Gaussian-specific choice, not something the
    subclass-agnostic recursions in ``core.py`` may assume.

    ``zs`` : shape ``(n_points, dim)``, ``weights`` : shape ``(n_points,)``,
    summing to 1 by convention (not required to be nonnegative, so this
    also works for signed unscented-transform weights).

    Any point containing NaN/Inf is masked out (its weight zeroed) before
    the weighted reduction. If every point is non-finite, both outputs are
    themselves non-finite.

    Returns
    -------
    mean : Array, shape (dim,)
    cov : Array, shape (dim, dim)
        The weighted covariance about ``mean`` (not about any other
        reference point).
    """
    dim = zs.shape[-1]
    valid = jnp.all(jnp.isfinite(zs), axis=-1)
    safe = jnp.where(valid[:, None], zs, 0.0)
    w_valid = jnp.where(valid, weights, 0.0)
    w_sum = jnp.sum(w_valid)
    mean = jnp.where(
        w_sum > 0,
        jnp.sum(w_valid[:, None] * safe, axis=0) / w_sum,
        jnp.full((dim,), jnp.nan, dtype=zs.dtype),
    )
    centered = safe - mean
    cov = jnp.where(
        w_sum > 0,
        jnp.einsum("i,ij,ik->jk", w_valid, centered, centered) / w_sum,
        jnp.full((dim, dim), jnp.nan, dtype=zs.dtype),
    )
    return mean, cov


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

    def __init__(
        self,
        dim: int,
        rank: int,
        *,
        use_sigma_points: bool = True,
        ut_alpha: float = 1.0,
        ut_kappa: float = 0.0,
    ):
        dim = int(dim)
        rank = int(rank)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not 0 <= rank <= dim:
            raise ValueError(
                f"rank must satisfy 0 <= rank <= dim, got rank={rank}, dim={dim}"
            )

        structure: Literal["full", "diag"] = "diag" if rank == 0 else "full"
        self._layout = _Layout(dim=dim, structure=structure)
        self._rank = rank
        self._use_sigma_points = use_sigma_points
        self._ut_alpha = ut_alpha
        self._ut_kappa = ut_kappa

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
        # h is emitted independently of J.  Unlike the old LoRaMVN
        # parameterization (h = K^T b, J = K^T K) which coupled the
        # linear and quadratic terms, the unified design keeps h free.
        # The diagonal baseline ``diag(softplus(d_free))`` guarantees
        # J is always positive-definite (even when L ≈ 0), so a large
        # h cannot produce an unbounded posterior mean — the baseline
        # precision prevents ``J^{-1} h`` from blowing up.
        h = free[:d]
        d_free = free[d : 2 * d]
        diag_prec = constrain_positive(d_free)

        if self._layout.is_diag:
            return jnp.concatenate((h, diag_prec))

        # full: J = diag(prec) + L @ L^T
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

    def transition_points(
        self, key: Array, moment: Array, mc_size: int
    ) -> tuple[Array, Array]:
        """See base class.

        Deterministic unscented-transform sigma points when
        ``use_sigma_points=True`` (constructor kwarg); otherwise falls back
        to the base class's plain Monte Carlo default via ``super()``, so
        this class stays in sync with any future change to that default
        instead of duplicating it.
        """
        if self._use_sigma_points:
            return _unscented_transition_points(
                self, key, moment, mc_size, alpha=self._ut_alpha, kappa=self._ut_kappa
            )
        return super().transition_points(key, moment, mc_size)

    def transition_stat(self, zs: Array, weights: Array) -> tuple[Array, Array]:
        """See base class.

        Reduces the raw propagated point set to its weighted
        mean/covariance pair via :func:`_weighted_moments` -- the
        sufficient statistic this Gaussian family's own :meth:`shrink`
        needs, and asymptotically smaller than the raw point set whenever
        the point count exceeds ``state_dim``.
        """
        return _weighted_moments(zs, weights)

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


class _MVNNoiseMstep:
    """Exact-MVN Noise M-step strategy, delegating algebra to ``noise.approx``."""

    @staticmethod
    def collect_minibatch_stat(noise, moment: Array, transition_stat):
        approx = noise.approx
        moment_t = moment[:, 1:, :]
        mean_f, cov_f = transition_stat

        def one(moment_t_i, mean_f_i, cov_f_i):
            mean_t, cov_t = approx.unpack(moment_t_i)
            residual = mean_t - mean_f_i
            return jnp.outer(residual, residual) + cov_t + cov_f_i

        raw = jax.vmap(jax.vmap(one))(moment_t, mean_f, cov_f)
        return jnp.sum(raw, axis=(0, 1)), jnp.asarray(
            raw.shape[0] * raw.shape[1], dtype=raw.dtype
        )

    @staticmethod
    def mstep(noise, epoch_stat, *, prior):
        approx = noise.approx
        sums, count = epoch_stat
        value, prior_count = prior
        d = approx._layout.dim
        value = jnp.asarray(value, dtype=sums.dtype)
        if value.ndim == 0:
            value = value * jnp.eye(d, dtype=sums.dtype)
        elif value.ndim == 1:
            value = jnp.diag(value)

        cov = (sums + prior_count * value) / (count + prior_count)
        cov = 0.5 * (cov + cov.T)
        if approx._layout.is_diag:
            cov = jnp.diag(jnp.diagonal(cov))
        chol = jnp.linalg.cholesky(cov + _EPS * jnp.eye(d, dtype=cov.dtype))
        canon = MVNParam(loc=jnp.zeros(d, dtype=cov.dtype), chol=chol)
        return approx.canon_to_free(canon)


Noise.register_mstep(MVN, _MVNNoiseMstep())



# ---------------------------------------------------------------------------
# unscented-transform transition points
# ---------------------------------------------------------------------------


def _unscented_transition_points(
    approx: MVN,
    key: Array,
    moment: Array,
    mc_size: int,
    *,
    alpha: float,
    kappa: float,
) -> tuple[Array, Array]:
    """Deterministic ``2*dim + 1`` unscented-transform sigma points and
    weights for ``approx``'s distribution.

    Standalone, composable, independently testable -- ``MVN.
    transition_points`` merely delegates to it when ``use_sigma_points=True``.
    Ignores ``key`` and ``mc_size`` (point count is always ``2*dim + 1``, not
    ``mc_size``-controlled). See ``docs/transition_points.md`` for the
    derivation and the reasoning behind using a single weight set (no
    separate covariance/``beta`` weights) and the ``alpha=1.0, kappa=0.0``
    default (chosen for float32 numerical stability, not the smaller
    ``alpha`` common in the UKF literature).
    """
    del key, mc_size
    mean, cov = approx.unpack(moment)
    dim = mean.shape[-1]

    lam = alpha**2 * (dim + kappa) - dim
    c = dim + lam

    if approx._layout.is_diag:
        sqrt_c_col = jnp.sqrt(c * jnp.diag(cov))  # (dim,)
        spread = jnp.diag(
            sqrt_c_col
        )  # (dim, dim), columns = sqrt(c) * e_i * sqrt(var_i)
    else:
        chol = jnp.linalg.cholesky(cov + _EPS * jnp.eye(dim, dtype=cov.dtype))
        spread = jnp.sqrt(c) * chol  # (dim, dim), columns are the spread directions

    points = jnp.concatenate(
        (mean[None, :], mean[None, :] + spread.T, mean[None, :] - spread.T),
        axis=0,
    )  # (2*dim + 1, dim)

    w0 = jnp.asarray(lam / c, dtype=mean.dtype)
    wi = jnp.full((2 * dim,), 1.0 / (2.0 * c), dtype=mean.dtype)
    weights = jnp.concatenate((w0[None], wi), axis=0)

    return points, weights
