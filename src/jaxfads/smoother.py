"""
XFADS smoother module.

This module implements the main XFADS class that orchestrates the complete
variational inference pipeline for Bayesian state-space modeling. It combines
neural encoders, dynamics models, observation models, and filtering/smoothing
algorithms.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp
from jax import random as jrnd
from gearax.modules import ConfModule, load_model, save_model
from omegaconf import OmegaConf

from . import core, distributions, encoders  # noqa: F401 — distributions registers Approx subclasses
from .core import Mode
from .base import Approx
from .base import Dynamics
from .nn import DataMasker
from .base import Observation
from .util import vmap_with_key
from .logging import get_logger


logger = get_logger(__name__)


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
        - approx: Exponential family approximation name (e.g. 'MVN')
        - approx_kwargs: Keyword arguments for approx instantiation
        - forward: Forward dynamics model type
        - obs_conf: Observation model config
        - mode: Inference mode ('pseudo', 'bifilter' not tested)

    Attributes
    ----------
    forward : Dynamics
        Forward dynamics model for state transitions.
    observation : ObservationModel
        Observation model with likelihood and readout.
    alpha_encoder : AlphaEncoder
        Neural encoder for observation information updates.
    beta_encoder : BetaEncoder
        Neural encoder for temporal dependencies.
    masker : DataMasker
        Dropout masker for pseudo-observations during training.
    unconstrained_prior_natural : Array
        Free-form prior parameters (constrained to natural at inference).
    noise_free : Array
        Free-form noise parameters (constrained to canon/mean at inference).

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
    ...     'approx': 'MVN',
    ...     'approx_kwargs': {},
    ...     'forward': 'Linear',
    ...     'obs_conf': {
    ...         'model': 'GLM',
    ...         'observation_dim': 50,
    ...         'state_dim': 10,
    ...         'likelihood': 'Poisson',
    ...         'cov': [1.0] * 50,
    ...         'norm_readout': False,
    ...     },
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
    >>> natural, mean, prediction = model(t, y, u, c, key=key)
    """

    forward: Dynamics
    # backward: Dynamics | None
    observation: Observation
    alpha_encoder: Callable
    beta_encoder: Callable
    masker: DataMasker
    unconstrained_prior_natural: Any
    noise_free: Any

    def __init__(self, conf, key=None):  # key unused; seed from conf for serializable reproducibility
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

        seed = self.conf.seed
        dropout = self.conf.dropout
        forward = self.conf.forward

        key = jrnd.key(seed)

        logger.info(
            "XFADS init: mode=%s approx=%s approx_kwargs=%s forward=%s observation_model=%s state_dim=%s obs_dim=%s mc_size=%s dropout=%s seed=%s",
            str(self.conf.mode),
            str(self.conf.approx),
            str(dict(self.conf.approx_kwargs)),
            str(forward),
            str(self.conf.obs_conf.model),
            str(self.conf.state_dim),
            str(self.conf.observation_dim),
            str(self.conf.mc_size),
            str(dropout),
            str(seed),
        )

        self.masker = DataMasker(dropout)

        key, ky = jrnd.split(key)
        self.forward = Dynamics.get_subclass(forward)(
            self.conf.dyn_conf,
            key=ky,
        )

        self.noise_free = self.approx.free_from_kw(
            scale=self.conf.dyn_conf.state_noise
        )

        key, ky = jrnd.split(key)
        observation_model_cls = Observation.get_subclass(self.conf.obs_conf.model)
        self.observation = observation_model_cls(self.conf.obs_conf, key=ky)

        # Encoders are approx-agnostic; inject the flat parameter size derived
        # from the configured approximation.
        param_size = int(self.approx.param_size(self.conf.state_dim))
        enc_conf = OmegaConf.merge(self.conf.enc_conf, {"param_size": param_size})

        key, ky = jrnd.split(key)
        self.alpha_encoder = encoders.AlphaEncoder(enc_conf, ky)

        key, ky = jrnd.split(key)
        self.beta_encoder = encoders.BetaEncoder(enc_conf, ky)

        # TODO: add hooks to freeze observation parameters if needed.
        # if "s" in static_params:
        #     self.forward.set_static()

        self.unconstrained_prior_natural = self.approx.free_from_kw(scale=1.0)

    def initialize(self, t, y, u, c):
        """
        Initialize model parameters based on data statistics.

        Parameters
        ----------
        t : Array, shape (N, T)
            Time steps for each sequence in the batch.
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
        Delegates initialization to the observation model.
        """
        observation = self.observation.initialize(t, y, u, c)
        return eqx.tree_at(lambda model: model.observation, self, observation)

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
        logger.info("XFADS load: path=%s", str(path))
        return load_model(path, cls)

    @classmethod
    def save(cls, model, path: str | Path):
        """
        Save a trained XFADS model to disk.

        Parameters
        ----------
        model : XFADS
            Model instance to save.
        path : str or Path
            Path where to save the model.
        """
        logger.info("XFADS save: path=%s", str(path))
        save_model(path, model)

    @property
    def approx(self):
        """
        Exponential-family approximation instance.

        Returns
        -------
        Approx
            An approximation instance configured from ``approx`` and
            ``approx_kwargs``.
        """
        cls = Approx.get_subclass(self.conf.approx)
        return cls(dim=self.conf.state_dim, **self.conf.approx_kwargs)

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
        return self.approx.moment_to_natural(
            self.approx.canon_to_moment(
                self.approx.free_to_canon(self.unconstrained_prior_natural)
            )
        )


    def __call__(self, t, y, u, c, *, key) -> tuple[Array, Array, Array]:
        """
        Perform variational inference for state-space model.

        This is the main inference method that processes observation sequences
        through neural encoders and applies filtering/smoothing algorithms to
        estimate posterior distributions over latent states.

        Parameters
        ----------
        t : Array, shape (N, T)
            Time steps for each sequence in the batch. Each row is passed
            to the per-sequence filter/smoother after vmapping.
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
        approx = self.approx

        def _free_to_natural(free_flat):
            canon = approx.free_to_canon(free_flat)
            moment = approx.canon_to_moment(canon)
            return approx.moment_to_natural(moment)

        batch_free_to_natural = vmap(vmap(_free_to_natural))
        batch_alpha_encode = vmap_with_key(vmap_with_key(self.alpha_encoder))
        batch_beta_encode = vmap_with_key(self.beta_encoder)

        match self.conf.mode:
            case Mode.BIFILTER:
                raise NotImplementedError("BIFILTER mode is not implemented.")
            case _:
                smooth_batch = vmap(partial(core.filter, self))

                def batch_encode(y: Array, key) -> Array:
                    # handling missing values
                    mask_y = jnp.all(
                        jnp.isfinite(y), axis=2, keepdims=True
                    )  # nonfinite are missing values
                    # chex.assert_equal_shape((y, mask_y), dims=(0, 1))
                    y = jnp.where(mask_y, y, 0)

                    key, alpha_key = jrnd.split(key)
                    a = batch_free_to_natural(batch_alpha_encode(y, key=alpha_key))
                    a = jnp.where(mask_y, a, 0)  # miss_values have no updates to state

                    key, mask_key = jrnd.split(key)
                    mask_a = self.masker(y, key=mask_key)
                    a = jnp.where(mask_a, a, 0)  # pseudo missing values

                    key, beta_key = jrnd.split(key)
                    b = batch_free_to_natural(batch_beta_encode(a, key=beta_key))

                    key, mask_key = jrnd.split(key)
                    mask_b = self.masker(y, key=mask_key)
                    b = jnp.where(mask_b, b, 0)  # filter only steps

                    a_plus_b = a + b

                    # key, mask_ab = self.masker(y, key=key)
                    # ab = jnp.where(mask_ab, ab, 0)

                    return a_plus_b

        key, encode_key = jrnd.split(key)
        alpha = batch_encode(y, encode_key)

        return smooth_batch(jrnd.split(key, jnp.size(t, 0)), t, alpha, u, c)
