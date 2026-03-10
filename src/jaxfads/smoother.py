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

from . import core, distributions, encoders, observations, state_maps, steppers  # noqa: F401 — side-effect registers subclasses
from .core import Mode
from .base import Approx
from .base import StateMap, Stepper
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
        - state_map: State-map model type
        - stepper: Stepper type
        - obs_conf: Observation model config
        - mode: Inference mode ('filter', 'smooth', 'causal')

    Attributes
    ----------
    state_map : StateMap
        State-map model for latent dynamics.
    stepper : Stepper
        Numerical stepper used to evolve latent state.
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
    ...     'state_map': 'OUStateMap',
    ...     'stepper': 'EulerStepper',
    ...     'obs_conf': {
    ...         'model': 'GLM',
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

    state_map: StateMap
    stepper: Stepper
    # backward: StateMap | None
    observation: Observation
    alpha_encoder: Callable
    beta_encoder: Callable
    masker: DataMasker
    unconstrained_prior_natural: Any
    noise_free: Any

    def __init__(
        self, conf, key=None
    ):  # key unused; seed from conf for serializable reproducibility
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
        state_map_name = self.conf.state_map
        stepper_name = self.conf.stepper

        key = jrnd.key(seed)

        logger.info(
            "XFADS init: mode=%s approx=%s approx_kwargs=%s state_map=%s stepper=%s observation_model=%s state_dim=%s obs_dim=%s mc_size=%s dropout=%s seed=%s",
            str(self.conf.mode),
            str(self.conf.approx),
            str(dict(self.conf.approx_kwargs)),
            str(state_map_name),
            str(stepper_name),
            str(self.conf.obs_conf.model),
            str(self.conf.state_dim),
            str(self.conf.observation_dim),
            str(self.conf.mc_size),
            str(dropout),
            str(seed),
        )

        self.masker = DataMasker(dropout)

        # Sub-configs may omit duplicated dimension fields; inject the global
        # dimensions so submodules have a consistent view.
        dyn_conf = OmegaConf.merge(
            self.conf.dyn_conf,
            {
                "state_dim": self.conf.state_dim,
                "observation_dim": self.conf.observation_dim,
            },
        )

        key, ky = jrnd.split(key)
        self.state_map = StateMap.get_subclass(state_map_name)(dyn_conf, key=ky)
        self.stepper = Stepper.get_subclass(stepper_name)(dyn_conf)

        self.noise_free = self.approx.free_from_kw(scale=dyn_conf.state_noise)

        obs_conf = OmegaConf.merge(
            self.conf.obs_conf,
            {
                "state_dim": self.conf.state_dim,
                "observation_dim": self.conf.observation_dim,
                # Fail-fast validation for observation models that rely on
                # Approx-specific helpers.
                "_approx_name": self.conf.approx,
            },
        )

        key, ky = jrnd.split(key)
        observation_model_cls = Observation.get_subclass(obs_conf.model)
        self.observation = observation_model_cls(obs_conf, key=ky)

        # Encoders are approx-agnostic; inject representation-level sizes.
        free_size = int(self.approx.free_size())
        param_size = int(self.approx.param_size())

        enc_conf = OmegaConf.merge(
            self.conf.enc_conf,
            {
                "observation_dim": self.conf.observation_dim,
                "param_size": param_size,
                "free_size": free_size,
            },
        )

        key, ky = jrnd.split(key)
        self.alpha_encoder = encoders.AlphaEncoder(enc_conf, ky)

        key, ky = jrnd.split(key)
        self.beta_encoder = encoders.BetaEncoder(enc_conf, ky)

        # if "s" in static_params:
        #     self.state_map.set_static()

        self.unconstrained_prior_natural = self.approx.free_from_kw(scale=1.0)

    def transition(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        """One-step latent transition composed from state map and stepper."""
        return self.stepper.step(z, u, c, self.state_map, key=key)

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
        kwargs = dict(self.conf.approx_kwargs)
        # Default rank to state_dim (full rank) when not specified.
        kwargs.setdefault("rank", self.conf.state_dim)
        return cls(dim=self.conf.state_dim, **kwargs)

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

        4. **Inference Mode**:
           - `FILTER`: forward filtering on alpha-only sites (`alpha`)
           - `SMOOTH`: forward filtering on additive sites (`alpha + beta`)
           - `CAUSAL`: alpha-only filtering then beta reconstruction

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

        def _free_to_natural(free_flat: Array) -> Array:
            return approx.free_to_natural(free_flat)

        batch_free_to_natural = vmap(vmap(_free_to_natural))
        batch_alpha_encode = vmap_with_key(vmap_with_key(self.alpha_encoder))
        batch_beta_encode = vmap_with_key(self.beta_encoder)

        def batch_encode_alpha(y: Array, key) -> Array:
            # handling missing values
            mask_y = jnp.all(
                jnp.isfinite(y), axis=2, keepdims=True
            )  # nonfinite are missing values
            y_clean = jnp.where(mask_y, y, 0)

            key, alpha_key = jrnd.split(key)
            a_free = batch_alpha_encode(y_clean, key=alpha_key)
            a = batch_free_to_natural(a_free)
            a = jnp.where(mask_y, a, 0)  # missing values: no update

            key, mask_key = jrnd.split(key)
            mask_a = self.masker(y_clean, key=mask_key)
            a = jnp.where(mask_a, a, 0)  # pseudo missing values
            return a

        def batch_encode_beta(a: Array, key) -> Array:
            key, beta_key = jrnd.split(key)
            b_free = batch_beta_encode(a, key=beta_key)
            b = batch_free_to_natural(b_free)

            key, mask_key = jrnd.split(key)
            # DataMasker only uses shape[:2], so use alpha as mask reference.
            mask_b = self.masker(a, key=mask_key)
            b = jnp.where(mask_b, b, 0)
            return b

        alpha_key, beta_key, rest_key = jrnd.split(key, 3)
        a = batch_encode_alpha(y, alpha_key)
        keys = jrnd.split(rest_key, jnp.size(t, 0))

        match self.conf.mode:
            case Mode.FILTER:
                smooth_batch = vmap(partial(core.filter, self))
                return smooth_batch(keys, t, a, u, c)
            case Mode.CAUSAL:
                b = batch_encode_beta(a, beta_key)
                smooth_batch = vmap(partial(core.causal, self))
                return smooth_batch(keys, t, a, b, u, c)
            case Mode.SMOOTH:
                b = batch_encode_beta(a, beta_key)
                smooth_batch = vmap(partial(core.smooth, self))
                return smooth_batch(keys, t, a, b, u, c)
            case _:
                raise ValueError(
                    f"Unsupported mode: {self.conf.mode}. Expected one of "
                    f"{Mode.FILTER!s}, {Mode.SMOOTH!s}, {Mode.CAUSAL!s}."
                )
