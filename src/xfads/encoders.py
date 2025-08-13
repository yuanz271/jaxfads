"""
Neural encoders for variational inference in XFADS.

This module implements neural network encoders that convert observations and
temporal information into natural parameter updates for variational inference
in XFADS. The encoders learn to map raw observations to
structured updates for the posterior distributions over latent states.
"""

from collections.abc import Callable
import math
from jax import lax, random as jrnd, vmap
from jaxtyping import Array
import equinox as eqx
from omegaconf import DictConfig
from gearax.modules import ConfModule

from .distributions import Approx
from .nn import make_mlp


class AlphaEncoder(ConfModule):
    """
    Alpha encoder for observation-driven information updates in XFADS.

    The alpha encoder is a feedforward neural network that converts raw
    observations into natural parameter updates for the variational
    posterior. These updates represent the instantaneous information
    gained from each observation about the latent state.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - observation_dim: Dimensionality of input observations
        - state_dim: Dimensionality of latent state
        - approx: Exponential family approximation type
        - width: Hidden layer width
        - depth: Number of hidden layers
        - dropout: Dropout probability (optional)
    key : Array
        JAX random key for parameter initialization.

    Attributes
    ----------
    layer : Callable
        Multi-layer perceptron mapping observations to parameter updates.

    Notes
    -----
    The alpha encoder implements the mapping:
    α_t = AlphaEncoder(y_t)

    where α_t are natural parameter updates that capture the information
    content of observation y_t about the latent state z_t. These updates
    are combined with prior information in the filtering recursion.

    The output dimension matches the parameter size of the chosen
    exponential family approximation (e.g., 2D for diagonal MVN,
    D+D² for full covariance MVN).
    """
    layer: Callable

    def __init__(self, conf: DictConfig, key: Array):
        self.conf = conf
        approx = Approx.get_subclass(conf.approx)
        self.layer = make_mlp(
            conf.observation_dim,
            approx.param_size(conf.state_dim),
            conf.width,
            conf.depth,
            key=key,
            dropout=conf.dropout,
        )

    def __call__(self, y: Array, *, key=None) -> Array:
        """
        Encode observations into natural parameter updates.

        Parameters
        ----------
        y : Array, shape (observation_dim,)
            Input observation vector.
        key : Array, optional
            Random key for dropout during training.

        Returns
        -------
        Array, shape (param_dim,)
            Natural parameter update vector α_t.

        Notes
        -----
        The output represents instantaneous information updates that
        will be added to the prior natural parameters in the filtering
        step to form the posterior.
        """
        return self.layer(y, key=key)


class BetaEncoder(ConfModule):
    """
    Beta encoder for temporal dependency modeling in XFADS.

    The beta encoder is a recurrent neural network (GRU) that processes
    sequences of alpha updates in reverse time order to capture temporal
    dependencies and smooth state transitions. It learns to predict
    additional natural parameter updates that encourage temporal coherence.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - state_dim: Dimensionality of latent state
        - approx: Exponential family approximation type
        - width: Hidden state dimension for the RNN
        - dropout: Dropout probability (optional)
    key : Array
        JAX random key for parameter initialization.

    Attributes
    ----------
    h0 : Array, shape (width,)
        Initial hidden state for the GRU cell.
    cell : Callable
        GRU cell for processing sequences.
    output : Callable
        Linear layer mapping hidden states to parameter updates.
    dropout : eqx.nn.Dropout, optional
        Dropout layer for regularization.

    Notes
    -----
    The beta encoder implements temporal smoothing by processing
    alpha updates in reverse chronological order:

    β_{1:T} = BetaEncoder(α_{T:1})

    where β_t are additional natural parameter updates that encourage
    temporal consistency. The reverse processing allows each β_t to
    incorporate information from future observations, leading to
    smoother state trajectories.

    The final posterior update at time t is:
    η_t = η_prior + α_t + β_t

    This architecture is inspired by bidirectional RNNs but operates
    on parameter space rather than raw observations.
    """
    h0: Array
    cell: Callable
    output: Callable
    dropout: eqx.nn.Dropout | None = None

    def __init__(self, conf: DictConfig, key: Array):
        self.conf = conf
        approx = Approx.get_subclass(conf.approx)

        param_size = approx.param_size(conf.state_dim)

        key, ky = jrnd.split(key)
        lim = 1 / math.sqrt(conf.width)
        self.h0 = jrnd.uniform(ky, (conf.width,), minval=-lim, maxval=lim)

        key, ky = jrnd.split(key)
        self.cell = eqx.nn.GRUCell(param_size, conf.width, key=ky)

        key, ky = jrnd.split(key)
        self.output = eqx.nn.Linear(conf.width, param_size, key=ky)

        if conf.dropout is not None:
            self.dropout = eqx.nn.Dropout(conf.dropout)

    def __call__(self, a: Array, *, key: Array) -> Array:
        """
        Encode temporal dependencies from alpha updates.

        Parameters
        ----------
        a : Array, shape (T, param_dim)
            Sequence of alpha updates (natural parameter updates from observations).
        key : Array
            Random key for dropout during training.

        Returns
        -------
        Array, shape (T, param_dim)
            Sequence of beta updates (temporal smoothing terms).

        Notes
        -----
        Processes the alpha sequence in reverse time order to allow each
        beta update to incorporate information from future time points.
        This creates a form of temporal smoothing that encourages coherent
        state trajectories.

        The processing steps are:
        1. Initialize hidden state h_0
        2. For t = T, T-1, ..., 1: h_t = GRU(α_t, h_{t+1})
        3. For t = 1, ..., T: β_t = Linear(h_t)
        """

        def step(h, inp):
            h = self.cell(inp, h)
            return h, h

        _, hs = lax.scan(step, init=self.h0, xs=a, reverse=True)

        if self.dropout is not None:
            hs = self.dropout(hs, key=key)

        b = vmap(self.output)(hs)

        return b
