"""
Observation/emission models for XFADS.

This module implements concrete observation components that define the
relationship between latent states and observed data. Abstract interfaces
are defined in ``jaxfads.base``.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jrnd
from jax.scipy.linalg import cho_solve

from .base import Observation
from .constraints import _EPS, _MIN_VARIANCE, constrain_positive, unconstrain_positive
from .base import Approx
from .nn import StationaryLinear, VariantBiasLinear

_MAX_LOGRATE = 7.0


def _quadratic_diag(C: Array, cov_z: Array) -> Array:
    """Compute diag(C @ Σ_z @ C.T) without materialising (D_obs, D_obs)."""
    if cov_z.ndim == 1:
        return (C**2) @ cov_z
    return jnp.sum((C @ cov_z) * C, axis=-1)


# ---------------------------------------------------------------------------
# Readout initializer registry
# ---------------------------------------------------------------------------

_READOUT_INIT: dict[str, Callable[[Array, Any], tuple[Array | None, Array]]] = {}


def register_readout_init(name: str) -> Callable:
    """
    Register a readout initializer by name.

    The decorated function must have signature
    ``(y: Array, conf) -> (C | None, bias)`` where *C* is the weight
    matrix (or ``None`` to leave weights unchanged) and *bias* is the
    readout bias in observation space.  ``GLM.initialize`` applies the
    likelihood's link function (e.g. ``log`` for Poisson) to *bias*
    before setting it on the readout.

    Parameters
    ----------
    name : str
        Key used to look up the initializer from configuration
        (``obs_conf.readout_init``).

    Returns
    -------
    Callable
        Decorator that registers the function.

    Examples
    --------
    >>> @register_readout_init("pca")
    ... def _pca_init(y, conf):
    ...     ...
    ...     return C_pca, mean_y
    """

    def decorator(fn: Callable) -> Callable:
        _READOUT_INIT[name] = fn
        return fn

    return decorator


def _init_bias(y: Array, n_steps: int) -> Array:
    """Compute mean bias from observations.

    Parameters
    ----------
    y : Array, shape (N, T, obs_dim)
        Observations.
    n_steps : int
        Number of time-variant bias steps.  When ``> 0`` a per-step
        mean is returned (for :class:`VariantBiasLinear`); otherwise
        the grand mean is returned (for :class:`StationaryLinear`).
        When ``> 0``, ``y.shape[1]`` must equal ``n_steps``.

    Returns
    -------
    Array
        Bias of shape ``(obs_dim,)`` (stationary) or
        ``(T, obs_dim)`` (time-variant).
    """
    if n_steps > 0:
        return jnp.mean(y, axis=0)  # (T, obs_dim)
    return jnp.mean(y.reshape(-1, y.shape[-1]), axis=0)  # (obs_dim,)


def _fa_weight(y_centered: Array, state_dim: int, obs_noise_var: float) -> Array:
    """Estimate Factor Analysis loading matrix from centred data.

    Parameters
    ----------
    y_centered : Array, shape (N, T, obs_dim) or (N*T, obs_dim)
        Observations with bias removed.
    state_dim : int
        Number of latent dimensions (columns of the loading matrix).
    obs_noise_var : float
        Isotropic observation noise variance to subtract.  Use ``0.0``
        to degrade gracefully to PCA.

    Returns
    -------
    Array, shape (obs_dim, state_dim)
        Loading matrix with sign disambiguation via skewness.
    """
    y_flat = y_centered.reshape(-1, y_centered.shape[-1])

    n = y_flat.shape[0]
    cov_y = (y_flat.T @ y_flat) / jnp.maximum(n - 1, 1)
    cov_signal = cov_y - obs_noise_var * jnp.eye(cov_y.shape[0])

    eigvals, eigvecs = jnp.linalg.eigh(cov_signal)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    top_vals = jnp.maximum(eigvals[:state_dim], 0.0)
    C_fa = eigvecs[:, :state_dim] * jnp.sqrt(top_vals)  # (obs_dim, state_dim)

    # Sign disambiguation via skewness
    z_proj = y_flat @ C_fa
    signs = jnp.sign(jnp.sum(z_proj**3, axis=0))
    signs = jnp.where(signs == 0, 1.0, signs)
    return C_fa * signs


def _resolve_obs_noise_var(conf: Any) -> float:
    """Resolve observation noise variance from config.

    Resolution chain:
    ``readout_init_conf.obs_noise_var`` → ``mean(conf.cov)`` → ``0.0``.

    Parameters
    ----------
    conf : DictConfig
        Observation configuration.

    Returns
    -------
    float
        Scalar observation noise variance.
    """
    init_conf = conf.get("readout_init_conf", None)
    if init_conf is not None and init_conf.get("obs_noise_var", None) is not None:
        return float(init_conf.obs_noise_var)
    if conf.get("cov", None) is not None:
        return float(jnp.mean(jnp.asarray(conf.cov)))
    return 0.0


@register_readout_init("fa")
def _fa_init(y: Array, conf: Any) -> tuple[Array, Array]:
    """Factor Analysis readout initializer.

    Estimates the loading matrix *C* and bias *b* in two stages:

    1. **Bias** — per-step or grand mean of the observations.
    2. **Weight** — eigendecomposition of the signal covariance after
       subtracting isotropic observation noise.

    The ``obs_noise_var`` is resolved via :func:`_resolve_obs_noise_var`.

    Parameters
    ----------
    y : Array, shape (N, T, obs_dim)
        Observations.
    conf : DictConfig
        Observation configuration (``obs_conf``).

    Returns
    -------
    C : Array, shape (obs_dim, state_dim)
        Factor Analysis loading matrix.
    bias : Array
        Mean observations — shape ``(obs_dim,)`` or ``(T, obs_dim)``.
    """
    bias = _init_bias(y, conf.get("n_steps", 0))
    y_centered = y - bias
    C = _fa_weight(y_centered, conf.state_dim, _resolve_obs_noise_var(conf))
    return C, bias


def make_readout(conf, key: Array) -> StationaryLinear | VariantBiasLinear:
    """
    Construct a readout module from observation configuration.

    Parameters
    ----------
    conf : DictConfig
        Observation configuration containing readout settings.
    key : Array
        PRNG key for parameter initialization.

    Returns
    -------
    StationaryLinear or VariantBiasLinear
        Readout module configured for stationary or time-varying biases.
    """
    n_steps = conf.get("n_steps", 0)
    if n_steps > 0:
        return VariantBiasLinear(
            conf.state_dim,
            conf.observation_dim,
            n_steps,
            key=key,
            norm_readout=conf.norm_readout,
        )
    return StationaryLinear(
        conf.state_dim,
        conf.observation_dim,
        key=key,
        norm_readout=conf.norm_readout,
    )


# ---------------------------------------------------------------------------
# Likelihood protocol and GLM orchestrator
# ---------------------------------------------------------------------------


class Likelihood(Protocol):
    """
    Protocol for observation/emission likelihoods in XFADS.

    Defines the interface for computing expected log-likelihoods of
    observations given latent state distributions.
    """

    @staticmethod
    def link(y: Array) -> Array:
        """
        Link function mapping observations to natural parameter space.

        Parameters
        ----------
        y : Array
            Values in observation space.

        Returns
        -------
        Array
            Transformed values in natural parameter space.
        """
        ...

    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "Likelihood":
        """
        Initialize likelihood parameters from data statistics.

        Returns
        -------
        Likelihood
            Updated likelihood instance. Default implementation is a no-op.
        """
        ...

    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx: Approx,
        mc_size: int,
        readout: Callable[..., Array],
    ) -> Array:
        """
        Compute expected log-likelihood of observations.
        """
        ...

    def mstep(
        self,
        t: Array,
        moment: Array,
        y: Array,
        approx: Approx,
        readout: Callable[..., Array],
    ) -> "Likelihood":
        """
        Closed-form, non-SGD parameter update from a full forward pass.

        Optional: not every ``Likelihood`` needs a closed-form update (e.g.
        ``Poisson`` has no free dispersion parameter to estimate this way).
        ``GLM`` dispatches to this via ``hasattr`` rather than requiring
        every ``Likelihood`` to implement it, since ``Likelihood`` is a
        structural ``Protocol``, not an inheritable base with a runtime
        default. Implementations are not required to be gradient-free on
        their own terms -- the guarantee against interfering with SGD
        lives at the call site, which only invokes this after the current
        step's gradient has already been computed and applied.
        """
        ...

    def mstep_frozen_paths(self) -> list[str]:
        """
        Attribute paths (relative to this Likelihood) that must be excluded
        from gradient updates whenever :meth:`mstep`-driven updates are
        active.

        Optional, same caveat as :meth:`mstep`: not every ``Likelihood``
        implements this; ``GLM`` dispatches via ``hasattr``.
        """
        ...


class GLM(Observation):
    """GLM observation model composed of a likelihood and readout.

    This wrapper owns the readout module and the likelihood, delegating
    expected log-likelihood computations and coordinating initialization.

    Notes
    -----
    The current analytic implementations of :meth:`Poisson.eloglik` and
    :meth:`Gaussian.eloglik` assume that the latent approximation can be
    unpacked into a mean and covariance via ``approx.unpack(moment)``.

    This is MVN-specific. The expected Approx name is injected by `XFADS` into
    the observation config as `obs_conf._approx_name`, and validated in
    `GLM.__init__` for fail-fast errors.
    """

    readout: StationaryLinear | VariantBiasLinear
    likelihood: Any

    def __init__(self, conf, key: Array):
        self.conf = conf
        self._validate_conf()
        key, readout_key = jrnd.split(key)
        self.readout = make_readout(conf, readout_key)
        key, likelihood_key = jrnd.split(key)
        likelihood_name = conf.likelihood
        if likelihood_name == "Poisson":
            self.likelihood = Poisson(
                conf,
                key=likelihood_key,
            )
        elif likelihood_name == "Gaussian":
            self.likelihood = Gaussian(
                conf,
                key=likelihood_key,
            )
        else:
            raise ValueError(f"Unsupported observation likelihood '{likelihood_name}'.")

    def _validate_conf(self) -> None:
        """Fail-fast validation of Approx compatibility from config.

        GLM likelihood code currently relies on MVN-only helpers (notably
        ``approx.unpack(moment)``), which are not part of the `Approx` ABC.
        """
        approx_name = self.conf.get("_approx_name", None)
        if approx_name is None:
            raise NotImplementedError(
                "GLM requires `obs_conf._approx_name` to be injected by XFADS for "
                "fail-fast Approx validation."
            )
        if str(approx_name) != "MVN":
            raise NotImplementedError(
                "GLM analytic eloglik currently supports only MVN approximations "
                "(requires approx.unpack(moment) -> (mean, cov))."
            )

    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx: Approx,
        mc_size: int,
    ) -> Array:
        """
        Compute expected log-likelihood using the bound likelihood.

        Parameters
        ----------
        key : Array
            JAX PRNG key for stochastic computation (if needed).
        t : Array
            Time index for time-varying parameters.
        moment : Array
            Moment parameters of the latent state distribution q(z_t).
        y : Array
            Observed data at time t.
        approx : Approx
            Exponential family approximation instance defining q(z).
        mc_size : int
            Number of Monte Carlo samples (for stochastic approximations).

        Returns
        -------
        Array
            Expected log-likelihood E_{q(z_t)}[log p(y_t | z_t)].
        """
        return self.likelihood.eloglik(key, t, moment, y, approx, mc_size, self.readout)

    def mstep(self, t: Array, moment: Array, y: Array, approx: Approx) -> "GLM":
        """See :meth:`Observation.mstep`.

        Dispatches to ``self.likelihood.mstep`` via ``hasattr`` rather than
        requiring every ``Likelihood`` to implement it, since ``Likelihood``
        is a structural ``Protocol`` (not an inheritable base with a
        runtime default) -- see ``Likelihood.mstep``'s docstring. Falls
        back to a no-op (``return self``) for likelihoods that don't
        implement it (e.g. ``Poisson``).
        """
        if not hasattr(self.likelihood, "mstep"):
            return self
        new_likelihood = self.likelihood.mstep(t, moment, y, approx, self.readout)
        return eqx.tree_at(lambda m: m.likelihood, self, new_likelihood)

    def mstep_frozen_paths(self) -> list[str]:
        """See :meth:`Observation.mstep_frozen_paths`.

        Dispatches to ``self.likelihood.mstep_frozen_paths`` via ``hasattr``
        (same reasoning as :meth:`mstep`), prepending ``"likelihood."`` to
        each returned path since ``self.likelihood`` is nested one level
        inside ``GLM``. Falls back to ``[]`` for likelihoods that don't
        implement it.
        """
        if not hasattr(self.likelihood, "mstep_frozen_paths"):
            return []
        return ["likelihood." + p for p in self.likelihood.mstep_frozen_paths()]

    def set_readout(
        self, weight: Array | None = None, bias: Array | None = None
    ) -> "GLM":
        """
        Manually set readout weight and/or bias.

        Parameters
        ----------
        weight : Array or None, optional
            Weight matrix of shape (observation_dim, state_dim).
            If ``None``, the weight is left unchanged.
        bias : Array or None, optional
            Bias vector of shape (observation_dim,) for
            :class:`StationaryLinear`, or (n_biases, observation_dim) for
            :class:`VariantBiasLinear`.  If ``None``, the bias is left
            unchanged.

        Returns
        -------
        GLM
            Updated observation model.
        """
        readout = self.readout
        if weight is not None:
            readout = readout.set_weight(weight)
        if bias is not None:
            readout = readout.set_bias(bias)
        return eqx.tree_at(lambda m: m.readout, self, readout)

    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "GLM":
        """
        Initialize likelihood and readout parameters from data statistics.

        The readout initialization strategy is controlled by the
        ``readout_init`` key in the observation configuration:

        - ``"fa"`` (default): Factor Analysis — estimates weight *C* and
          bias *b* from data covariance.
        - ``None``: skip readout initialization entirely.

        Custom initializers can be added via
        :func:`register_readout_init`.

        Parameters
        ----------
        t : Array
            Time steps for the sequences.
        y : Array
            Observation sequences, shape (N, T, obs_dim).
        u : Array
            Control input sequences.
        c : Array
            Covariate sequences.

        Returns
        -------
        GLM
            Updated observation model with initialized components.
        """
        likelihood = self.likelihood
        readout = self.readout

        # --- likelihood init (existing) ---------------------------------
        likelihood_initializer = getattr(likelihood, "initialize", None)
        if likelihood_initializer is not None:
            likelihood = likelihood_initializer(t, y, u, c)

        # --- readout init -----------------------------------------------
        init_name = self.conf.get("readout_init", "fa")

        if init_name is not None:
            if init_name not in _READOUT_INIT:
                raise ValueError(
                    f"Unknown readout_init '{init_name}'. "
                    f"Registered: {list(_READOUT_INIT)}."
                )
            init_fn = _READOUT_INIT[init_name]
            weight, bias = init_fn(y, self.conf)

            if weight is not None:
                readout = readout.set_weight(weight)
            readout = readout.set_bias(likelihood.link(bias))

        return eqx.tree_at(
            lambda model: (model.readout, model.likelihood),
            self,
            (readout, likelihood),
        )


__all__ = [
    "GLM",
    "Observation",
    "Poisson",
    "Gaussian",
    "make_readout",
    "Likelihood",
    "register_readout_init",
]


# ---------------------------------------------------------------------------
# Concrete likelihoods
# ---------------------------------------------------------------------------


class Poisson(eqx.Module, strict=True):
    """
    Poisson observation model for count data in XFADS.

    Implements Poisson likelihood for discrete count observations with
    log-linear dependence on latent states. Suitable for neural spike
    counts, word counts, or other non-negative integer data.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - observation_dim: Number of observed count variables
    key : Array
        Random key for parameter initialization (unused).

    Notes
    -----
    The Poisson model assumes:
    y_t | z_t ~ Poisson(λ_t)
    log(λ_t) = C z_t + b_t + δ_t

    where C is the readout matrix, b_t are (optional) time-varying biases,
    and δ_t accounts for uncertainty propagation from the latent state.
    """

    conf: Any = eqx.field(static=True)

    def __init__(self, conf, key):  # pyright: ignore[reportMissingSuperCall]
        self.conf = conf

    @staticmethod
    def link(y: Array) -> Array:
        """
        Log link function for Poisson observations.

        Parameters
        ----------
        y : Array
            Values in observation space (rates / counts).

        Returns
        -------
        Array
            Log-transformed values in natural parameter space.
        """
        return jnp.log(jnp.maximum(y, _EPS))

    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx,
        mc_size: int,
        readout,
    ) -> Array:
        """
        Compute expected log-likelihood for Poisson observations.

        Parameters
        ----------
        key : Array
            Random key (unused in this implementation).
        t : Array
            Time index for time-varying parameters.
        moment : Array
            Moment parameters of latent state distribution q(z_t).
        y : Array, shape (observation_dim,)
            Observed count data.
        approx : Approx
            Exponential family approximation instance.
        mc_size : int
            Number of Monte Carlo samples (unused for analytic computation).
        readout : callable
            Readout module mapping latent states to log-rates.

        Returns
        -------
        Array
            Expected log-likelihood E_{q(z_t)}[log p(y_t | z_t)].

        Notes
        -----
        Computes the expectation analytically using the log-sum-exp identity:
        E[log p(y|z)] = Σ_i (y_i * η_i - λ_i)

        where η_i = E[C_i z] and λ_i = E[exp(C_i z + b_i)] with uncertainty
        correction for the exponential nonlinearity.
        """
        mean_z, cov_z = approx.unpack(moment)
        eta = readout(t, mean_z)
        C = readout.weight
        cvc = _quadratic_diag(C, cov_z)
        loglam = eta + 0.5 * cvc
        # loglam = jnp.where(loglam < _MAX_LOGRATE, loglam, jnp.log(loglam))
        loglam = jnp.minimum(loglam, _MAX_LOGRATE)
        lam = jnp.exp(loglam)
        ll = jnp.sum(y * eta - lam)
        return ll


class Gaussian(eqx.Module, strict=True):
    """
    Diagonal Gaussian observation model for continuous data in XFADS.

    Implements Gaussian likelihood with diagonal observation noise for
    continuous-valued observations. Assumes independence between observation
    dimensions but allows uncertainty propagation from latent states.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - observation_dim: Number of observed continuous variables
        - cov: Initial observation noise variance (scalar or vector)
    key : Array
        Random key for parameter initialization (unused).

    Attributes
    ----------
    unconstrained_cov : Array, shape (observation_dim,)
        Unconstrained observation noise parameters.

    Notes
    -----
    The Gaussian model assumes:
    y_t | z_t ~ N(μ_t, Σ_obs)
    μ_t = C z_t + b_t
    Σ_obs = diag(σ²_1, ..., σ²_d)

    where C is the readout matrix, b_t are (optional) time-varying biases,
    and σ²_i are observation noise variances.
    """

    unconstrained_cov: Array
    conf: Any = eqx.field(static=True)

    def __init__(self, conf, key):  # pyright: ignore[reportMissingSuperCall]
        self.conf = conf
        cov = jnp.array(conf.get("cov", jnp.ones(conf.observation_dim)))
        self.unconstrained_cov = unconstrain_positive(jnp.maximum(cov, _EPS))

    @staticmethod
    def link(y: Array) -> Array:
        """
        Identity link function for Gaussian observations.

        Parameters
        ----------
        y : Array
            Values in observation space.

        Returns
        -------
        Array
            Same values (identity transform).
        """
        return y

    def cov(self):
        """
        Get the observation noise covariance.

        Returns
        -------
        Array, shape (observation_dim,)
            Diagonal observation noise variances.

        Notes
        -----
        Applies positive constraint, then adds ``_MIN_VARIANCE`` (always, a
        private float32-safety constant -- not a modeling choice) so the
        variance stays bounded away from zero for any ``unconstrained_cov``,
        including values that would otherwise underflow ``constrain_positive``
        to an exact ``0.0``.
        """
        return _MIN_VARIANCE + constrain_positive(self.unconstrained_cov)

    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx: Approx,
        mc_size: int,
        readout,
    ) -> Array:
        """
        Compute expected log-likelihood for Gaussian observations.

        Parameters
        ----------
        key : Array
            Random key (unused in this implementation).
        t : Array
            Time index for time-varying parameters.
        moment : Array
            Moment parameters of latent state distribution q(z_t).
        y : Array, shape (observation_dim,)
            Observed continuous data.
        approx : Approx
            Exponential family approximation instance.
        mc_size : int
            Number of Monte Carlo samples (unused for analytic computation).
        readout : callable
            Readout module mapping latent states to observation means.

        Returns
        -------
        Array
            Expected log-likelihood E_{q(z_t)}[log p(y_t | z_t)].

        Notes
        -----
        Computes the expectation analytically by propagating uncertainty
        from the latent state through the linear readout:

        E[log p(y|z)] = log N(y; E[Cz], C*Cov(z)*C^T + Σ_obs)

        where the observation covariance includes both state uncertainty
        and observation noise.

        Uses the matrix determinant lemma and Woodbury identity to avoid ever
        forming the dense ``(observation_dim, observation_dim)`` covariance
        matrix ``Sigma_y = R + C @ Cov(z) @ C.T``. This matters when
        ``observation_dim >> state_dim`` (e.g. a linear readout onto raw,
        high-dimensional observations): the naive dense form costs
        ``O(observation_dim**2)`` memory and ``O(observation_dim**3)`` time
        (a Cholesky factorization) per time step per trial in a batch; the
        Woodbury form below costs ``O(observation_dim * state_dim)`` memory
        and ``O(observation_dim * state_dim**2 + state_dim**3)`` time
        instead.

        The one ``O(observation_dim * state_dim**2)`` contraction
        (``C.T @ diag(1/r) @ C``) depends only on the (trainable) readout
        weight ``C`` and observation noise ``r`` -- not on ``moment``, ``t``,
        or the batch/time index -- so under ``jax.vmap`` over batch and time
        it carries no mapped axis and is effectively computed once rather
        than once per (batch, time) instance.
        """
        mean_z, cov_z = approx.unpack(moment)
        mean_y = readout(t, mean_z)
        C = readout.weight  # (d, m): d = observation_dim, m = state_dim
        r = self.cov()  # (d,)

        d = r.shape[-1]
        m = cov_z.shape[-1]
        eye_m = jnp.eye(m, dtype=cov_z.dtype)
        delta = y - mean_y  # (d,)
        r_inv = 1.0 / r  # (d,)

        # Hoistable: depends only on C, r (see Notes above).
        Ct_Rinv = C.T * r_inv[None, :]  # (m, d)
        K = Ct_Rinv @ C  # (m, m) = C^T R^-1 C

        # Per-(batch,time): Sigma_z^-1 via a damped solve, matching this
        # module's MVN.moment_to_natural convention for covariance inversion.
        cov_z_damped = cov_z + _EPS * eye_m
        cov_z_inv = jnp.linalg.solve(cov_z_damped, eye_m)
        M = cov_z_inv + K  # (m, m), guaranteed PD (cov_z_inv is PD, K is PSD)
        L_M = jnp.linalg.cholesky(M)

        v = Ct_Rinv @ delta  # (m,)
        Minv_v = cho_solve((L_M, True), v)
        quad = jnp.sum(delta**2 * r_inv) - v @ Minv_v

        logdet_R = jnp.sum(jnp.log(r))
        L_cov_z = jnp.linalg.cholesky(cov_z_damped)
        logdet_cov_z = 2.0 * jnp.sum(jnp.log(jnp.diag(L_cov_z)))
        logdet_M = 2.0 * jnp.sum(jnp.log(jnp.diag(L_M)))
        logdet = logdet_R + logdet_cov_z + logdet_M

        ll = -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet + quad)
        return ll

    def mstep_stat(self, t: Array, moment: Array, y: Array, approx: Approx, readout) -> Array:
        """Per-(batch,time) sufficient statistic for the closed-form EM M-step
        update of this likelihood's observation covariance.

        Returns ``(y - E[C z])**2 + diag(C Cov(z) C^T)``, shape ``(observation_dim,)``.
        Averaging this quantity over batch and time (see :func:`mstep_gaussian_cov`)
        gives the covariance value that maximizes the expected log-likelihood given
        the current posterior -- the standard EM M-step for a Gaussian observation
        model with a (possibly time-varying) linear readout, holding the
        dynamics/readout/encoder fixed (E-step already performed by the caller via
        ``model(...)``).

        Why this exists (Heywood-case immunity)
        ----------------------------------------
        ``eloglik`` above computes the *joint* likelihood
        ``log N(y; E[Cz], C Cov(z) C^T + R)`` with a low-rank (``C Cov(z) C^T``,
        rank <= state_dim) plus diagonal (``R``) covariance structure -- the same
        structure as a factor-analysis model, with ``R`` playing the role of the
        factor model's "uniquenesses". This is well known in the factor-analysis
        literature to admit a degenerate MLE mode (a "Heywood case"): joint
        gradient-based optimization of ``R`` can drive one or more of its
        components toward the numerical floor while the corresponding
        dimension's residual stays large, because the low-rank correction term
        can make the *joint* density favor this even though it does not reflect a
        genuine improvement in fit for that dimension. This was observed directly
        in practice: a component's fitted covariance reached ``_MIN_VARIANCE``
        while its actual residual variance, measured independently, was ~10^6
        times larger.

        Estimating ``R`` via this M-step instead of joint gradient descent avoids
        the exploit entirely: the estimate *is* the expected squared residual
        (including the propagated posterior uncertainty ``diag(C Cov(z) C^T)``),
        so it cannot decouple from the actual reconstruction quality the way a
        freely-optimized parameter can.

        Parameters
        ----------
        t : Array
            Time index for the (possibly time-varying) readout.
        moment : Array
            Moment parameters of the posterior q(z_t) (from the E-step).
        y : Array, shape (observation_dim,)
            Observed data at this (batch, time) instance.
        approx : Approx
            Exponential-family approximation instance (``approx.unpack`` must
            return ``(mean, cov)`` with a dense ``(state_dim, state_dim)``
            covariance -- MVN-only, matching this class's ``eloglik``).
        readout : callable
            Readout module mapping latent states to observation means; must
            expose a ``.weight`` attribute (the linear map ``C``), matching
            ``eloglik``'s usage.

        Returns
        -------
        Array, shape (observation_dim,)
            ``(y - E[Cz])**2 + diag(C Cov(z) C^T)`` at this (batch, time) instance.
        """
        mean_z, cov_z = approx.unpack(moment)
        mean_y = readout(t, mean_z)
        C = readout.weight  # (d, m)
        residual_sq = (y - mean_y) ** 2  # (d,)
        propagated_var = jnp.einsum("dj,jk,dk->d", C, cov_z, C)  # (d,) = diag(C cov_z C^T)
        return residual_sq + propagated_var

    def mstep(self, t: Array, moment: Array, y: Array, approx: Approx, readout) -> "Gaussian":
        """Closed-form EM M-step update over a full forward pass's worth of
        ``(t, moment, y)`` (all (batch, time) instances at once), reusing
        :meth:`mstep_stat`'s per-instance math (see its docstring for the
        Heywood-case rationale). Same result as :func:`mstep_gaussian_cov`
        computing over a single batch, just packaged as a per-Likelihood
        method rather than a standalone full-dataset driver.

        Not required to be gradient-free on its own terms: callers (e.g.
        ``train()``'s own ``train_step``/``apply_mstep``) are responsible
        for invoking this only after any differentiated computation for
        the current step has already completed, so nothing here ever
        feeds back into an active gradient tape. That call-site ordering
        is the actual guarantee, not an internal ``stop_gradient``.
        """
        stat_fn = jax.vmap(
            jax.vmap(partial(self.mstep_stat, approx=approx, readout=readout))
        )
        stat = stat_fn(t, moment, y)  # (batch, time, observation_dim)
        r_new = jnp.mean(stat, axis=(0, 1))
        # Invert cov() = _MIN_VARIANCE + constrain_positive(unconstrained_cov):
        # unconstrained_cov = unconstrain_positive(max(r_new - _MIN_VARIANCE, _EPS)),
        # matching mstep_gaussian_cov's own convention.
        new_unconstrained_cov = unconstrain_positive(
            jnp.maximum(r_new - _MIN_VARIANCE, _EPS)
        )
        return eqx.tree_at(lambda m: m.unconstrained_cov, self, new_unconstrained_cov)

    def mstep_frozen_paths(self) -> list[str]:
        """See :meth:`Observation.mstep_frozen_paths`. Relative to
        ``self`` (i.e. relative to ``self.likelihood`` from ``GLM``'s
        perspective).
        """
        return ["unconstrained_cov"]

    # def set_static(self, static=True) -> None:
    #     """
    #     Set observation noise parameters as static (non-trainable).

    #     Parameters
    #     ----------
    #     static : bool, default=True
    #         Whether to make parameters static.
    #     """
    #     self.__dataclass_fields__["unconstrained_cov"].metadata = {"static": static}


def mstep_gaussian_cov(model, data, *, key, batch_size: int | None = None):
    """Closed-form EM M-step update for a Gaussian-likelihood XFADS model's
    observation covariance, given the model's *current* dynamics, readout, and
    encoder (all held fixed) and a dataset. Returns a new model with
    ``observation.likelihood.unconstrained_cov`` replaced by the M-step-optimal
    value; every other attribute (including ``observation.likelihood.cov_floor``
    -- there is none; only the always-on ``_MIN_VARIANCE``, see
    ``constraints.py``) is unchanged.

    Rationale: see :meth:`Gaussian.mstep_stat`'s docstring for why joint
    gradient-based MLE of a Gaussian observation model's per-dimension noise
    covariance is prone to a Heywood-case degeneracy, and why computing it
    directly from the current posterior's residuals instead is immune to that
    failure mode. This function performs the E-step (running the model forward
    to get the posterior over data) and the M-step (aggregating
    :meth:`Gaussian.mstep_stat` over the dataset) together.

    Usage
    -----
    Intended for classic EM-style alternation with gradient-based optimization
    of the remaining parameters: freeze ``observation.likelihood.unconstrained_cov``
    via ``conf.freeze_paths`` so the optimizer's own gradient updates do not fight
    this direct estimate, run a round of gradient-based training (Adam, L-BFGS,
    ...), call this function to update the covariance from the resulting
    posterior, and repeat::

        conf.freeze_paths = ["observation.likelihood.unconstrained_cov"]
        for _ in range(n_rounds):
            model = train(model, data, conf=conf)
            model = mstep_gaussian_cov(model, data, key=key)

    Parameters
    ----------
    model : XFADS
        Model whose ``observation`` exposes ``.likelihood`` (with an
        ``mstep_stat(t, moment, y, approx, readout)`` method -- ``Gaussian``
        provides one; a custom likelihood can too, without needing to
        subclass ``Gaussian``, since dispatch here is duck-typed on the method,
        not on ``Gaussian`` specifically) and ``.readout``.
    data : tuple of Array
        ``(t, y, u, c)``, as accepted by ``model(...)``.
    key : Array
        JAX PRNG key (passed through to the model's forward pass; ``Gaussian``'s
        ``eloglik``/``mstep_stat`` are analytic and do not use it, but the
        encoder or other model components may).
    batch_size : int or None, optional
        Process the dataset in chunks of this many trials at a time (default:
        all trials in a single batch). Use this if the full dataset does not
        fit in memory for a single forward pass.

    Returns
    -------
    XFADS
        A new model with ``observation.likelihood.unconstrained_cov`` set to
        the M-step-optimal value; all other attributes unchanged.

    Raises
    ------
    NotImplementedError
        If ``model.observation.likelihood`` has no ``mstep_stat`` method (e.g.
        ``Poisson``, which has no free variance parameter to estimate this way).
    """
    observation = model.observation
    likelihood = getattr(observation, "likelihood", None)
    readout = getattr(observation, "readout", None)
    if likelihood is None or readout is None or not hasattr(likelihood, "mstep_stat"):
        raise NotImplementedError(
            "mstep_gaussian_cov requires model.observation.likelihood to implement "
            f"mstep_stat(...); got observation.likelihood={type(likelihood).__name__}"
        )

    t, y, u, c = data
    n_trials = t.shape[0]
    if batch_size is None:
        batch_size = n_trials
    approx = model.approx

    stat_fn = jax.vmap(
        jax.vmap(partial(likelihood.mstep_stat, approx=approx, readout=readout))
    )  # (batch, time) -> (batch, time, observation_dim)

    total_stat = jnp.zeros(y.shape[-1], dtype=y.dtype)
    total_n = 0
    for start in range(0, n_trials, batch_size):
        stop = min(start + batch_size, n_trials)
        tb, yb, ub, cb = t[start:stop], y[start:stop], u[start:stop], c[start:stop]
        _natural, moment, _predicted, _transition_stat = model(tb, yb, ub, cb, key=key)
        stat = stat_fn(tb, moment, yb)  # (nb, T, d)
        total_stat = total_stat + jnp.sum(stat, axis=(0, 1))
        total_n += stat.shape[0] * stat.shape[1]

    r_new = total_stat / total_n
    # Invert cov() = _MIN_VARIANCE + constrain_positive(unconstrained_cov):
    # unconstrained_cov = unconstrain_positive(max(r_new - _MIN_VARIANCE, _EPS)),
    # matching Gaussian.__init__'s own convention for turning a target
    # covariance into unconstrained_cov.
    new_unconstrained_cov = unconstrain_positive(jnp.maximum(r_new - _MIN_VARIANCE, _EPS))
    new_likelihood = eqx.tree_at(lambda lik: lik.unconstrained_cov, likelihood, new_unconstrained_cov)
    new_observation = eqx.tree_at(lambda obs: obs.likelihood, observation, new_likelihood)
    return eqx.tree_at(lambda m: m.observation, model, new_observation)


def mstep_observation_cov(model, data, *, key):
    """Closed-form, non-SGD parameter update for ``model.observation`` via a
    single full-dataset forward pass, dispatching through the
    :class:`~jaxfads.base.Observation` ABC's :meth:`~jaxfads.base.
    Observation.mstep` rather than duck-typing on a specific likelihood.

    This is the family-neutral counterpart to :func:`mstep_gaussian_cov`:
    where that function is Gaussian-specific (duck-typed on
    ``mstep_stat``, supports ``batch_size``-chunked scanning for datasets
    too large for one forward pass) and is left untouched, this function
    works for *any* :class:`~jaxfads.base.Observation` implementation that
    overrides :meth:`~jaxfads.base.Observation.mstep` (returning the model
    unchanged, a no-op, for those that don't -- e.g. ``GLM`` wrapping
    ``Poisson``), at the cost of no chunking support. Use
    :func:`mstep_gaussian_cov` directly if a dataset doesn't fit in a
    single forward pass.

    Parameters
    ----------
    model : XFADS
        Model whose ``observation`` may implement ``mstep``.
    data : tuple of Array
        ``(t, y, u, c)``, as accepted by ``model(...)``.
    key : Array
        JAX PRNG key, passed through to the model's forward pass.

    Returns
    -------
    XFADS
        A new model with ``observation`` replaced by the result of
        ``model.observation.mstep(t, moment, y, model.approx)``; all other
        attributes unchanged. Identical to ``model`` if ``observation``
        does not override ``mstep``.
    """
    t, y, u, c = data
    _natural, moment, _predicted, _transition_stat = model(t, y, u, c, key=key)
    new_observation = model.observation.mstep(t, moment, y, model.approx)
    return eqx.tree_at(lambda m: m.observation, model, new_observation)
