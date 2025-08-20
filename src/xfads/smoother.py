"""
XFADS smoother module.

This module implements the main XFADS class that orchestrates the complete
variational inference pipeline for Bayesian state-space modeling. It combines
neural encoders, dynamics models, observation models, and filtering/smoothing
algorithms.

Classes
-------
XFADS
    Main class for Bayesian state-space modeling with variational inference.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Self

from jax import Array, numpy as jnp, random as jrnd, vmap
import equinox as eqx
from gearax.modules import ConfModule, load_model, save_model

from . import core, encoders
from .distributions import Approx
from .dynamics import Dynamics
from .observations import Likelihood
from .nn import DataMasker
from .core import Hyperparam, Mode


class XFADS(ConfModule):
    """
    XFADS for Bayesian state-space modeling.

    XFADS implements variational inference for nonlinear dynamical systems using
    neural networks to parameterize variational distributions. It supports both
    forward filtering and bidirectional smoothing with various exponential family
    approximations.

    Parameters
    ----------
    conf : DictConfig
        Configuration object containing all model hyperparameters including:
        - state_dim: Dimensionality of latent state
        - observation_dim: Dimensionality of observations
        - mc_size: Number of Monte Carlo samples
        - approx: Exponential family approximation type
        - forward: Forward dynamics model type
        - observation: Observation model type
        - mode: Inference mode ('pseudo' or 'bifilter')

    Attributes
    ----------
    hyperparam : Hyperparam
        Compiled hyperparameters for the model.
    forward : Dynamics
        Forward dynamics model for state transitions.
    likelihood : Likelihood
        Observation/emission model.
    alpha_encoder : AlphaEncoder
        Neural encoder for observation information updates.
    beta_encoder : BetaEncoder
        Neural encoder for temporal dependencies.
    masker : DataMasker
        Dropout masker for pseudo-observations during training.
    unconstrained_prior_natural : Array
        Unconstrained prior natural parameters.

    Notes
    -----
    The model follows the state-space formulation:

    z_t = f(z_{t-1}, u_t, c_t) + ε_t    (dynamics)
    y_t = g(z_t, u_t, c_t) + δ_t        (observations)

    where z_t is the latent state, y_t are observations, u_t are controls,
    c_t are covariates, and ε_t, δ_t are noise terms.

    Examples
    --------
    >>> import jax.random as jrnd
    >>> from omegaconf import DictConfig
    >>>
    >>> conf = DictConfig({
    ...     'state_dim': 10,
    ...     'observation_dim': 50,
    ...     'mc_size': 100,
    ...     'approx': 'DiagMVN',
    ...     'forward': 'Linear',
    ...     'observation': 'Poisson'
    ... })
    >>>
    >>> key = jrnd.key(42)
    >>> model = XFADS(conf, key)
    >>>
    >>> # Run inference
    >>> t = jnp.arange(100)
    >>> y = jrnd.normal(key, (32, 100, 50))  # batch x time x obs
    >>> u = jnp.zeros((32, 100, 1))         # controls
    >>> c = jnp.zeros((32, 100, 1))         # covariates
    >>>
    >>> natural, moment, prediction = model(t, y, u, c, key=key)
    """
    hyperparam: Hyperparam = eqx.field(init=False, static=True)
    forward: eqx.Module
    # backward: eqx.Module | None
    likelihood: eqx.Module
    alpha_encoder: eqx.Module
    beta_encoder: eqx.Module
    masker: DataMasker
    unconstrained_prior_natural: Array

    def __init__(self, conf, key):
        """
        Initialize XFADS model components.

        Parameters
        ----------
        conf : DictConfig
            Configuration object containing model hyperparameters.
        key : Array
            JAX random key for parameter initialization.

        Notes
        -----
        Initializes all neural networks, dynamics models, and observation models
        based on the provided configuration. This method is automatically called
        by the ConfModule framework.
        """
        self.conf = conf

        state_dim = self.conf.state_dim
        # observation_dim = self.conf.observation_dim
        mc_size = self.conf.mc_size
        seed = self.conf.seed
        dropout = self.conf.dropout
        mode = self.conf.mode
        forward = self.conf.forward
        observation = self.conf.observation
        fb_penalty = self.conf.fb_penalty
        noise_penalty = self.conf.noise_penalty

        key = jrnd.key(seed)

        self.masker: DataMasker = DataMasker(dropout)

        approx: type[Approx] = Approx.get_subclass(self.conf.approx)

        self.hyperparam = Hyperparam(
            approx=approx,
            state_dim=state_dim,
            # iu_dim=iu_dim,
            # eu_dim=eu_dim,
            # observation_dim=observation_dim,
            mc_size=mc_size,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            mode=mode,
        )

        key, ky = jrnd.split(key)
        self.forward = Dynamics.get_subclass(forward)(
            self.conf.dyn_conf,
            key=ky,
            # state_dim=state_dim,
            # iu_dim=iu_dim,
            # eu_dim=eu_dim,
            # cov=state_noise,
            # **dyn_kwargs,
        )

        # if backward is not None:
        #     key, ky = jrnd.split(key)
        #     self.backward = AbstractDynamics.get_subclass(backward)(
        #         key=ky,
        #         state_dim=state_dim,
        #         iu_dim=iu_dim,
        #         eu_dim=eu_dim,
        #         cov=state_noise,
        #         **dyn_kwargs,
        #     )
        # else:
        #     self.backward = None

        key, ky = jrnd.split(key)
        self.likelihood = Likelihood.get_subclass(observation)(
            self.conf.obs_conf,
            key=ky,
            # state_dim,  # type: ignore
            # observation_dim,
            # n_steps=n_steps,
            # **obs_kwargs,
        )

        key, ky = jrnd.split(key)
        self.alpha_encoder = encoders.AlphaEncoder(
            self.conf.enc_conf,
            ky,
            # state_dim, observation_dim, approx, key=ky, **enc_kwargs
        )

        key, ky = jrnd.split(key)
        self.beta_encoder = encoders.BetaEncoder(
            self.conf.enc_conf,
            ky,
            # state_dim, approx, key=ky, **enc_kwargs
        )

        # if "l" in static_params:
        #     self.likelihood.set_static()
        # if "s" in static_params:
        #     self.forward.set_static()

        self.unconstrained_prior_natural = approx.unconstrain_natural(
            approx.prior_natural(state_dim)
        )

    def initialize(self, t, y, u, c):
        """
        Initialize model parameters based on data statistics.

        Parameters
        ----------
        t : Array, shape (T,)
            Time steps for the sequences.
        y : Array, shape (N, T, D_obs)
            Observation sequences.
        u : Array, shape (N, T, D_u)
            Control input sequences.
        c : Array, shape (N, T, D_c)
            Covariate sequences.

        Returns
        -------
        XFADS
            Model instance with initialized parameters.

        Notes
        -----
        Initializes observation model biases based on empirical mean of
        observations. Particularly useful for Poisson observations where
        biases are set to log-mean rates.
        """
        mean_y = jnp.mean(y, axis=0)
        biases = jnp.log(jnp.maximum(mean_y, 1e-6))
        return eqx.tree_at(lambda model: model.likelihood.readout.biases, self, biases)

    @classmethod
    def load(cls, path: str | Path):
        """
        Load a trained XFADS model from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.

        Returns
        -------
        XFADS
            Loaded model instance.
        """
        return load_model(path, cls)

    @classmethod
    def save(cls, model: Self, path: str | Path):
        """
        Save a trained XFADS model to disk.

        Parameters
        ----------
        model : XFADS
            Model instance to save.
        path : str or Path
            Path where to save the model.
        """
        save_model(path, model)

    def prior_natural(self) -> Array:
        """
        Get the prior distribution in natural parameter form.

        Returns
        -------
        Array
            Prior natural parameters for the initial state distribution.

        Notes
        -----
        Applies constraints to ensure parameters are in valid range
        for the chosen exponential family approximation.
        """
        return self.hyperparam.approx.constrain_natural(
            self.unconstrained_prior_natural
        )

    def __call__(self, t, y, u, c, *, key) -> tuple[Array, Array, Array]:
        """
        Perform variational inference for state-space model.

        This is the main inference method that processes observation sequences
        through neural encoders and applies filtering/smoothing algorithms to
        estimate posterior distributions over latent states.

        Parameters
        ----------
        t : Array, shape (T,)
            Time steps for the sequence.
        y : Array, shape (N, T, D_obs)
            Observation sequences where N is batch size, T is sequence length,
            and D_obs is observation dimensionality.
        u : Array, shape (N, T, D_u)
            Control/input sequences.
        c : Array, shape (N, T, D_c)
            Covariate sequences.
        key : Array
            JAX random key for stochastic operations.

        Returns
        -------
        natural_params : Array, shape (N, T, param_dim)
            Natural parameters of posterior distributions over states.
        moment_params : Array, shape (N, T, param_dim)
            Moment parameters of posterior distributions over states.
        predictions : Array, shape (N, T, param_dim)
            Predicted moment parameters from dynamics model.

        Notes
        -----
        The inference pipeline consists of:

        1. **Encoding**: Neural networks convert observations to natural parameter
           updates (alpha encoder) and temporal dependencies (beta encoder).

        2. **Missing Value Handling**: Non-finite observations are treated as
           missing and their updates are zeroed out.

        3. **Pseudo-Observations**: During training, the masker applies dropout
           to create pseudo-missing observations for regularization.

        4. **Filtering/Smoothing**: Applies either forward filtering (PSEUDO mode)
           or bidirectional smoothing (BIFILTER mode) to estimate posterior states.

        The method handles batched sequences efficiently using JAX transformations
        and supports both training and inference modes.

        Examples
        --------
        >>> # Single sequence inference
        >>> t = jnp.arange(100)
        >>> y = jrnd.normal(key, (1, 100, 50))
        >>> u = jnp.zeros((1, 100, 5))
        >>> c = jnp.zeros((1, 100, 3))
        >>>
        >>> natural, moment, pred = model(t, y, u, c, key=key)
        >>>
        >>> # Batch inference
        >>> y_batch = jrnd.normal(key, (32, 100, 50))
        >>> u_batch = jnp.zeros((32, 100, 5))
        >>> c_batch = jnp.zeros((32, 100, 3))
        >>>
        >>> natural, moment, pred = model(t, y_batch, u_batch, c_batch, key=key)
        """
        batch_alpha_encode: Callable = vmap(vmap(self.alpha_encoder))  # type: ignore
        batch_constrain_natural: Callable = vmap(
            vmap(self.hyperparam.approx.constrain_natural)
        )
        batch_beta_encode: Callable = vmap(self.beta_encoder)  # type: ignore

        match self.hyperparam.mode:
            case Mode.BIFILTER:
                raise NotImplementedError("BIFILTER mode is not implemented.")
                # batch_smooth = vmap(partial(core.bismooth, model=self))

                # def batch_encode(y: Array, key) -> Array:
                #     mask_y = jnp.all(
                #         jnp.isfinite(y), axis=2, keepdims=True
                #     )  # nonfinite are missing values
                #     # chex.assert_equal_shape((y, valid_mask), dims=(0,1))
                #     y = jnp.where(mask_y, y, 0)

                #     key, ky = jrnd.split(key)
                #     a = batch_constrain_natural(
                #         batch_alpha_encode(y, key=jrnd.split(ky, y.shape[:2]))
                #     )
                #     a: Array = jnp.where(mask_y, a, 0)

                #     key, mask_a = self.masker(a, key=key)
                #     a = jnp.where(mask_a, a, 0)  # type: ignore

                #     return a
            case _:
                batch_smooth = vmap(partial(core.filter, model=self))

                def batch_encode(y: Array, key) -> Array:
                    # handling missing values
                    mask_y = jnp.all(
                        jnp.isfinite(y), axis=2, keepdims=True
                    )  # nonfinite are missing values
                    # chex.assert_equal_shape((y, mask_y), dims=(0, 1))
                    y = jnp.where(mask_y, y, 0)

                    key, ky = jrnd.split(key)
                    a = batch_constrain_natural(
                        batch_alpha_encode(y, key=jrnd.split(ky, y.shape[:2]))
                    )
                    a = jnp.where(mask_y, a, 0)  # miss_values have no updates to state

                    key, mask_a = self.masker(y, key=key)
                    a = jnp.where(mask_a, a, 0)  # type: ignore # pseudo missing values

                    b = batch_constrain_natural(
                        batch_beta_encode(a, key=jrnd.split(key, a.shape[0]))  # type: ignore
                    )
                    key, mask_b = self.masker(y, key=key)
                    b = jnp.where(mask_b, b, 0)  # filter only steps

                    ab = a + b

                    # key, mask_ab = self.masker(y, key=key)
                    # ab = jnp.where(mask_ab, ab, 0)

                    return ab

        key, ky = jrnd.split(key)
        alpha = batch_encode(y, ky)

        return batch_smooth(jrnd.split(key, len(t)), t, alpha, u, c)
