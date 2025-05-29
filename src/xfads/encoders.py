import math
from jax import lax, random as jrnd, vmap
from jaxtyping import Array
import equinox as eqx
from omegaconf import DictConfig
from gearax.modules import ConfModule

from .distributions import Approx
from .nn import make_mlp


class AlphaEncoder(ConfModule):
    layer: eqx.Module

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
        return self.layer(y, key=key)


class BetaEncoder(ConfModule):
    h0: Array
    cell: eqx.Module
    output: eqx.Module
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
        :param a: natural form observation information
        """

        def step(h, inp):
            h = self.cell(inp, h)
            return h, h

        _, hs = lax.scan(step, init=self.h0, xs=a, reverse=True)

        if self.dropout is not None:
            hs = self.dropout(hs, key=key)

        b = vmap(self.output)(hs)

        return b
