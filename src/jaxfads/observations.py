"""
Observation/emission models for XFADS.

This module implements concrete observation components that define the
relationship between latent states and observed data. Abstract interfaces
are defined in ``jaxfads.base``.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfp
from jax import Array
from jax import numpy as jnp
from jax import random as jrnd

from gearax.mixin import SubclassRegistryMixin
from gearax.modules import ConfModule

from .base import ObservationModel
from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx
from .nn import StationaryLinear, VariantBiasLinear

MAX_LOGRATE = 7.0


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


class Likelihood(Protocol):
    """
    Protocol for observation/emission likelihoods in XFADS.

    Defines the interface for computing expected log-likelihoods of
    observations given latent state distributions.
    """

    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "Likelihood":
        """
        Initialize likelihood parameters from data statistics.

        Returns
        -------
        Likelihood
            Updated likelihood instance. Default implementation is a no-op.
        """
        ...

    def readout_init_target(self, mean_y: Array) -> Array:
        """
        Transform mean observations into readout initialization targets.

        Returns
        -------
        Array
            Initialization targets for the readout biases.
        """
        ...

    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx: type[Approx],
        mc_size: int,
        readout: Callable[..., Array],
    ) -> Array:
        """
        Compute expected log-likelihood of observations.
        """
        ...


class LikelihoodRegistry(SubclassRegistryMixin, ConfModule):
    """
    Registry base class for likelihood implementations.
    """

    def initialize(
        self, t: Array, y: Array, u: Array, c: Array
    ) -> "LikelihoodRegistry":
        """
        Initialize likelihood parameters from data statistics.
        """
        return self

    def readout_init_target(self, mean_y: Array) -> Array:
        """
        Transform mean observations into readout initialization targets.
        """
        return mean_y

    @abstractmethod
    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx: type[Approx],
        mc_size: int,
        readout: Callable[..., Array],
    ) -> Array:
        """
        Compute expected log-likelihood of observations.
        """
        ...


class GLM(ObservationModel):
    """
    GLM observation model composed of a likelihood and readout.

    This wrapper owns the readout module and the likelihood, delegating
    expected log-likelihood computations and coordinating initialization.
    """

    readout: StationaryLinear | VariantBiasLinear
    likelihood: Likelihood

    def __init__(self, conf, key: Array):
        self.conf = conf
        key, readout_key = jrnd.split(key)
        self.readout = make_readout(conf, readout_key)
        key, likelihood_key = jrnd.split(key)
        likelihood_name = conf.get("observation", "Poisson")
        self.likelihood = LikelihoodRegistry.get_subclass(likelihood_name)(
            conf,
            key=likelihood_key,
        )

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
        approx : type[Approx]
            Exponential family approximation class defining q(z).
        mc_size : int
            Number of Monte Carlo samples (for stochastic approximations).

        Returns
        -------
        Array
            Expected log-likelihood E_{q(z_t)}[log p(y_t | z_t)].
        """
        return self.likelihood.eloglik(key, t, moment, y, approx, mc_size, self.readout)

    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "GLM":
        """
        Initialize likelihood and readout parameters from data statistics.

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
        GLM
            Updated observation model with initialized components.
        """
        likelihood = self.likelihood.initialize(t, y, u, c)
        readout = self.readout
        initializer = getattr(readout, "initialize", None)
        if initializer is not None:
            mean_y = jnp.mean(y, axis=0)
            target = likelihood.readout_init_target(mean_y)
            if isinstance(readout, VariantBiasLinear):
                readout = initializer(target)
            else:
                if target.ndim > 1:
                    target = jnp.mean(target, axis=0)
                readout = initializer(target)
        return eqx.tree_at(
            lambda model: (model.readout, model.likelihood),
            self,
            (readout, likelihood),
        )


__all__ = [
    "GLM",
    "ObservationModel",
    "Poisson",
    "DiagGaussian",
    "make_readout",
    "Likelihood",
    "LikelihoodRegistry",
]


class Poisson(LikelihoodRegistry):
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

    def __init__(self, conf, key):
        self.conf = conf

    def readout_init_target(self, mean_y: Array) -> Array:
        """
        Transform mean observations into log-rate initialization targets.

        Parameters
        ----------
        mean_y : Array
            Mean observations over the batch dimension.

        Returns
        -------
        Array
            Log-mean targets for initializing the readout biases.
        """
        return jnp.log(jnp.maximum(mean_y, 1e-6))

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
        approx : type[Approx]
            Exponential family approximation class.
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
        mean_z, cov_z = approx.moment_to_canon(moment)
        eta = readout(t, mean_z)
        V = jnp.diag(cov_z)
        C = readout.weight
        cvc = jnp.diag(C @ V @ C.T)
        loglam = eta + 0.5 * cvc
        # loglam = jnp.where(loglam < MAX_LOGRATE, loglam, jnp.log(loglam))
        loglam = jnp.minimum(loglam, MAX_LOGRATE)
        lam = jnp.exp(loglam)
        ll = jnp.sum(y * eta - lam)
        return ll


class DiagGaussian(LikelihoodRegistry):
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

    unconstrained_cov: Array = eqx.field(static=False)

    def __init__(self, conf, key):
        self.conf = conf
        cov = jnp.array(conf.get("cov", jnp.ones(conf.observation_dim)))
        self.unconstrained_cov = unconstrain_positive(cov)

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
        approx: type[Approx],
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
        approx : type[Approx]
            Exponential family approximation class.
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
        mean_z, cov_z = approx.moment_to_canon(moment)
        mean_y = readout(t, mean_z)
        C = readout.weight  # left matrix
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
