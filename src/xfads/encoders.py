import math
import jax
from jax import random as jrnd
from jaxtyping import Array, Float
import equinox as eqx

from .nn import make_mlp


class AlphaEncoder(eqx.Module):
    # approx: Type = eqx.field(static=True)
    layer: eqx.Module

    def __init__(
        self, state_dim, observation_dim, depth, width, approx, *, key, dropout=None
    ):
        # self.approx = approx

        self.layer = make_mlp(
            observation_dim,
            approx.param_size(state_dim),
            width,
            depth,
            key=key,
            dropout=dropout,
        )

    def __call__(self, y, *, key=None):
        return self.layer(y, key=key)


class BetaEncoder(eqx.Module):
    # approx: Type = eqx.field(static=True)
    h0: Array
    cell: eqx.Module
    output: eqx.Module
    dropout: eqx.nn.Dropout | None

    def __init__(self, state_dim, depth, width, approx, *, key, dropout=None):
        # self.approx = approx

        param_size = approx.param_size(state_dim)

        key, subkey = jrnd.split(key)
        lim = 1 / math.sqrt(width)
        self.h0 = jrnd.uniform(subkey, (width,), minval=-lim, maxval=lim)

        key, subkey = jrnd.split(key)
        self.cell = eqx.nn.GRUCell(param_size, width, key=subkey)

        key, subkey = jrnd.split(key)
        self.output = eqx.nn.Linear(width, param_size, key=subkey)

        if dropout is not None:
            self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, a: Float[Array, "t h"], *, key):
        """
        :param a: natural form observation information
        """

        def step(h, inp):
            h = self.cell(inp, h)
            return h, h

        _, hs = jax.lax.scan(step, init=self.h0, xs=a, reverse=True)

        if self.dropout is not None:
            hs = self.dropout(hs, key=key)

        ab = jax.vmap(self.output)(hs)

        return ab
