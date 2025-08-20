"""
Observation/emission models for XFADS.

This module implements various observation models that define the relationship
between latent states and observed data. It provides likelihood functions for
different data types including count data (Poisson) and continuous data
(Gaussian) with support for time-varying parameters.

Classes
-------
Likelihood
    Abstract base class for observation models.
Poisson
    Poisson observation model for count data in XFADS.
DiagGaussian
    Diagonal Gaussian observation model for continuous data in XFADS.
"""

from abc import abstractmethod

from jax import Array, numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfp
import equinox as eqx

from gearax.modules import ConfModule
from gearax.mixin import SubclassRegistryMixin

from .nn import VariantBiasLinear, StationaryLinear
from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx


MAX_LOGRATE = 7.0


class Likelihood(SubclassRegistryMixin, ConfModule):
    @abstractmethod
    def eloglik(self, key: Array, t: Array, moment: Array, y: Array, approx, mc_size: int) -> Array: ...


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
        cov = conf.get("cov", jnp.ones(conf.observation_dim))
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
