"""
Observation/emission models for XFADS.

This module implements concrete observation components that define the
relationship between latent states and observed data. Abstract interfaces
are defined in ``jaxfads.base``.
"""

from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfp
from jax import Array
from jax import numpy as jnp
from jax import random as jrnd

from .base import Observation
from .constraints import _EPS, constrain_positive, unconstrain_positive
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
        self.unconstrained_cov = unconstrain_positive(cov)

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
        Applies positive constraint to ensure valid variance values.
        """
        return constrain_positive(self.unconstrained_cov)

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
        """
        mean_z, cov_z = approx.unpack(moment)
        mean_y = readout(t, mean_z)
        C = readout.weight  # left matrix
        cov_y = C @ cov_z @ C.T + jnp.diag(self.cov())
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll

    # def set_static(self, static=True) -> None:
    #     """
    #     Set observation noise parameters as static (non-trainable).

    #     Parameters
    #     ----------
    #     static : bool, default=True
    #         Whether to make parameters static.
    #     """
    #     self.__dataclass_fields__["unconstrained_cov"].metadata = {"static": static}
