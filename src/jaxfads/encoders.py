"""Neural encoders for variational inference in XFADS.

This module implements neural network encoders that convert observations and
temporal information into natural parameter updates for variational inference in
XFADS.

Encoders are **Approx-agnostic**: they only need flat vector sizes injected by
`XFADS`:

- `param_size`: natural-parameter vector length
- `free_size`: encoder free-form vector length

`XFADS` injects these sizes into the encoder config at construction time based
on the configured `Approx` instance.
"""

import math
from collections.abc import Callable

import equinox as eqx
from jax import Array, lax, vmap
from jax import random as jrnd
from omegaconf import DictConfig

from .base import Encoder
from .nn import make_mlp


class AlphaEncoder(Encoder):
    """Alpha encoder for observation-driven information updates in XFADS.

    The alpha encoder is a feedforward neural network that converts raw
    observations into flat free-form updates for the variational
    posterior.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - observation_dim: Dimensionality of input observations
        - free_size: Length of the flat encoder free-form vector
        - width: Hidden layer width
        - depth: Number of hidden layers
        - dropout: Dropout probability (optional)
    key : Array
        JAX random key for parameter initialization.

    Notes
    -----
    The mapping is:

    ``α_t = AlphaEncoder(y_t)``

    where ``α_t`` is a flat vector in the encoder free-form layout of the
    chosen `Approx`.
    """

    layer: Callable

    def __init__(self, conf: DictConfig, key: Array):
        self.conf = conf
        free_size = int(conf.free_size)

        self.layer = make_mlp(
            conf.observation_dim,
            free_size,
            conf.width,
            conf.depth,
            key=key,
            dropout=conf.dropout,
        )

    def __call__(self, y: Array, *, key=None) -> Array:
        """Encode a single observation vector into a free-form update."""
        return self.layer(y, key=key)


class BetaEncoder(Encoder):
    """Beta encoder for temporal dependency modeling in XFADS.

    The beta encoder is a GRU that processes sequences of alpha updates in
    reverse time order to capture temporal dependencies.

    Parameters
    ----------
    conf : DictConfig
        Configuration containing:
        - param_size: Length of the flat natural parameter vector (input)
        - free_size: Length of the flat encoder free-form vector (output)
        - width: Hidden state dimension for the GRU
        - dropout: Dropout probability (optional)
    key : Array
        JAX random key for parameter initialization.
    """

    h0: Array
    cell: Callable
    output: Callable
    dropout: eqx.nn.Dropout | None = None

    def __init__(self, conf: DictConfig, key: Array):
        self.conf = conf
        input_size = int(conf.param_size)
        output_size = int(conf.free_size)

        key, ky = jrnd.split(key)
        lim = 1 / math.sqrt(conf.width)
        self.h0 = jrnd.uniform(ky, (conf.width,), minval=-lim, maxval=lim)

        key, ky = jrnd.split(key)
        self.cell = eqx.nn.GRUCell(input_size, conf.width, key=ky)

        key, ky = jrnd.split(key)
        self.output = eqx.nn.Linear(conf.width, output_size, key=ky)

        if conf.dropout is not None:
            self.dropout = eqx.nn.Dropout(conf.dropout)

    def __call__(self, a: Array, *, key=None) -> Array:
        """Encode a sequence of alpha updates into beta updates."""

        def step(h, inp):
            h = self.cell(inp, h)
            return h, h

        _, hs = lax.scan(step, init=self.h0, xs=a, reverse=True)

        if self.dropout is not None:
            hs = self.dropout(hs, key=key)

        return vmap(self.output)(hs)


