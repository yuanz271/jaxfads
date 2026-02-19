"""
Observation/emission models for XFADS.

This module implements various observation models that define the relationship
between latent states and observed data. It provides likelihood functions for
different data types including count data (Poisson) and continuous data
(Gaussian) with support for time-varying parameters.
"""

from abc import abstractmethod

import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfp
from jax import Array
from jax import numpy as jnp

from gearax.mixin import SubclassRegistryMixin
from gearax.modules import ConfModule

from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx
from .nn import StationaryLinear, VariantBiasLinear, WeightNorm

MAX_LOGRATE = 7.0


def _set_stationary_bias(readout: StationaryLinear, bias: Array) -> StationaryLinear:
    layer = readout.layer
    if isinstance(layer, WeightNorm):
        layer = eqx.tree_at(lambda l: l.layer.bias, layer, bias)
    else:
        layer = eqx.tree_at(lambda l: l.bias, layer, bias)
    return eqx.tree_at(lambda r: r.layer, readout, layer)


class Likelihood(SubclassRegistryMixin, ConfModule):
    """
    Abstract base class for observation/emission models in XFADS.

    Defines the interface for computing expected log-likelihoods of
    observations given latent state distributions. Subclasses implement
    specific observation models (e.g., Poisson, Gaussian).

    Methods
    -------
    eloglik(key, t, moment, y, approx, mc_size)
        Compute expected log-likelihood E_{q(z)}[log p(y|z)].
    initialize(t, y, u, c)
        Initialize likelihood parameters from data statistics.

    Notes
    -----
    The expected log-likelihood is a key component of the ELBO objective:

    .. math::

        \\mathcal{L} = \\sum_t E_{q(z_t)}[\\log p(y_t | z_t)] - KL(q || p)

    Implementations should handle uncertainty propagation from the
    approximate posterior q(z) through the observation model.
    """

    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "Likelihood":
        """
        Initialize likelihood parameters from data statistics.

        Parameters
        ----------
        t : Array
            Time steps for the sequences.
        y : Array
            Observation sequences.
        u : Array
            Control input sequences.
        c : Array
            Covariate sequences.

        Returns
        -------
        Likelihood
            Updated likelihood instance. Default implementation is a no-op.
        """
        return self

    @abstractmethod
    def eloglik(
        self, key: Array, t: Array, moment: Array, y: Array, approx, mc_size: int
    ) -> Array:
        """
        Compute expected log-likelihood of observations.

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
        approx : type[Approx]
            Exponential family approximation class defining q(z).
        mc_size : int
            Number of Monte Carlo samples (for stochastic approximations).

        Returns
        -------
        Array
            Expected log-likelihood E_{q(z_t)}[log p(y_t | z_t)].
        """
        ...


class Poisson(Likelihood):
    """
    Poisson observation model for count data in XFADS.

    Implements Poisson likelihood for discrete count observations with
    log-linear dependence on latent states. Suitable for neural spike
    counts, word counts, or other non-negative integer data.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - state_dim: Dimensionality of latent states
        - observation_dim: Number of observed count variables
        - n_steps: Number of time steps (>0 for time-varying biases)
        - norm_readout: Whether to use weight normalization
    key : Array
        Random key for parameter initialization.

    Attributes
    ----------
    readout : StationaryLinear or VariantBiasLinear
        Linear readout layer mapping states to log-rates.

    Notes
    -----
    The Poisson model assumes:
    y_t | z_t ~ Poisson(λ_t)
    log(λ_t) = C z_t + b_t + δ_t

    where C is the readout matrix, b_t are (optional) time-varying biases,
    and δ_t accounts for uncertainty propagation from the latent state.
    """

    readout: StationaryLinear | VariantBiasLinear

    def __init__(self, conf, key):
        self.conf = conf
        n_steps = conf.get("n_steps", 0)

        if n_steps > 0:
            self.readout = VariantBiasLinear(
                conf.state_dim,
                conf.observation_dim,
                n_steps,
                key=key,
                norm_readout=conf.norm_readout,
            )
        else:
            self.readout = StationaryLinear(
                conf.state_dim,
                conf.observation_dim,
                key=key,
                norm_readout=conf.norm_readout,
            )

    def set_static(self, static=True) -> None:
        """
        Set readout parameters as static (non-trainable).

        Parameters
        ----------
        static : bool, default=True
            Whether to make parameters static.
        """
        self.readout.set_static(static)  # type: ignore

    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "Poisson":
        """
        Initialize readout biases from observation means.

        Parameters
        ----------
        t : Array
            Time steps for the sequences.
        y : Array
            Observation sequences.
        u : Array
            Control input sequences.
        c : Array
            Covariate sequences.

        Returns
        -------
        Poisson
            Updated Poisson likelihood with initialized biases.
        """
        mean_y = jnp.mean(y, axis=0)
        if isinstance(self.readout, VariantBiasLinear):
            biases = jnp.log(jnp.maximum(mean_y, 1e-6))
            readout = eqx.tree_at(lambda r: r.biases, self.readout, biases)
        else:
            mean_y = jnp.mean(mean_y, axis=0)
            bias = jnp.log(jnp.maximum(mean_y, 1e-6))
            readout = _set_stationary_bias(self.readout, bias)
        return eqx.tree_at(lambda m: m.readout, self, readout)

    def eloglik(
        self, key: Array, t: Array, moment: Array, y: Array, approx, mc_size: int
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
        approx : type[Approx]
            Exponential family approximation class.
        mc_size : int
            Number of Monte Carlo samples (unused for analytic computation).

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
        mean_z, cov_z = approx.moment_to_canon(moment)
        eta = self.readout(t, mean_z)
        V = jnp.diag(cov_z)
        C = self.readout.weight
        cvc = jnp.diag(C @ V @ C.T)
        loglam = eta + 0.5 * cvc
        # loglam = jnp.where(loglam < MAX_LOGRATE, loglam, jnp.log(loglam))
        loglam = jnp.minimum(loglam, MAX_LOGRATE)
        lam = jnp.exp(loglam)
        ll = jnp.sum(y * eta - lam)
        return ll


class DiagGaussian(Likelihood):
    """
    Diagonal Gaussian observation model for continuous data in XFADS.

    Implements Gaussian likelihood with diagonal observation noise for
    continuous-valued observations. Assumes independence between observation
    dimensions but allows uncertainty propagation from latent states.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - state_dim: Dimensionality of latent states
        - observation_dim: Number of observed continuous variables
        - cov: Initial observation noise variance (scalar or vector)
        - n_steps: Number of time steps (>0 for time-varying biases)
        - norm_readout: Whether to use weight normalization
    key : Array
        Random key for parameter initialization.

    Attributes
    ----------
    unconstrained_cov : Array, shape (observation_dim,)
        Unconstrained observation noise parameters.
    readout : StationaryLinear or VariantBiasLinear
        Linear readout layer mapping states to observation means.

    Notes
    -----
    The Gaussian model assumes:
    y_t | z_t ~ N(μ_t, Σ_obs)
    μ_t = C z_t + b_t
    Σ_obs = diag(σ²_1, ..., σ²_d)

    where C is the readout matrix, b_t are (optional) time-varying biases,
    and σ²_i are observation noise variances.
    """

    unconstrained_cov: Array = eqx.field(static=False)
    readout: StationaryLinear | VariantBiasLinear

    def __init__(self, conf, key):
        self.conf = conf
        cov = jnp.array(conf.get("cov", jnp.ones(conf.observation_dim)))
        self.unconstrained_cov = unconstrain_positive(cov)

        n_steps = conf.get("n_steps", 0)

        if n_steps > 0:
            self.readout = VariantBiasLinear(
                conf.state_dim,
                conf.observation_dim,
                n_steps,
                key=key,
                norm_readout=conf.norm_readout,
            )
        else:
            self.readout = StationaryLinear(
                conf.state_dim,
                conf.observation_dim,
                key=key,
                norm_readout=conf.norm_readout,
            )

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

    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "DiagGaussian":
        """
        Initialize readout biases from observation means.

        Parameters
        ----------
        t : Array
            Time steps for the sequences.
        y : Array
            Observation sequences.
        u : Array
            Control input sequences.
        c : Array
            Covariate sequences.

        Returns
        -------
        DiagGaussian
            Updated Gaussian likelihood with initialized biases.
        """
        mean_y = jnp.mean(y, axis=0)
        if isinstance(self.readout, VariantBiasLinear):
            readout = eqx.tree_at(lambda r: r.biases, self.readout, mean_y)
        else:
            bias = jnp.mean(mean_y, axis=0)
            readout = _set_stationary_bias(self.readout, bias)
        return eqx.tree_at(lambda m: m.readout, self, readout)

    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx: type[Approx],
        mc_size: int,
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
        approx : type[Approx]
            Exponential family approximation class.
        mc_size : int
            Number of Monte Carlo samples (unused for analytic computation).

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
        mean_z, cov_z = approx.moment_to_canon(moment)
        mean_y = self.readout(t, mean_z)
        C = self.readout.weight  # left matrix
        cov_y = C @ approx.full_cov(cov_z) @ C.T + jnp.diag(self.cov())
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll

    def set_static(self, static=True) -> None:
        """
        Set observation noise parameters as static (non-trainable).

        Parameters
        ----------
        static : bool, default=True
            Whether to make parameters static.
        """
        self.__dataclass_fields__["unconstrained_cov"].metadata = {"static": static}
