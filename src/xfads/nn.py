from collections.abc import Callable
from functools import partial
import math
from typing import Literal

import jax
from jax import nn as jnn, random as jrnd, numpy as jnp
from jaxtyping import Array, Scalar
import equinox as eqx
from equinox import nn as enn, Module

from . import constraints


_MIN_NORM: float = 1e-6
MAX_EXP: float = 5.0
EPS: float = jnp.finfo(jnp.float32).eps


def make_mlp(
    in_size: int | Literal["scalar"],
    out_size: int | Literal["scalar"],
    width: int,
    depth: int,
    *,
    key: Array,
    activation: Callable = jnn.swish,
    final_bias: bool = True,
    final_activation: Callable | None = None,
    dropout: float | None = None,
) -> enn.Sequential:
    key, layer_key = jrnd.split(key)
    layers = [enn.Linear(in_size, width, key=layer_key), enn.Lambda(activation)]
    for i in range(depth - 1):
        key, layer_key = jrnd.split(key)
        layers.append(enn.Linear(width, width, key=layer_key))
        layers.append(enn.Lambda(activation))
        if dropout is not None:
            layers.append(enn.Dropout(dropout))
    key, layer_key = jrnd.split(key)
    layers.append(enn.Linear(width, out_size, key=layer_key, use_bias=final_bias))
    if final_activation is not None:
        layers.append(enn.Lambda(activation))
    if dropout is not None:
        layers.append(enn.Dropout(dropout))

    return enn.Sequential(layers)


def _norm_except_axis(
    v: Array, norm: Callable[[Array], Scalar], axis: int | None
) -> Array:
    if axis is None:
        return norm(v)
    else:
        return jax.vmap(norm, in_axes=axis, out_axes=axis)(v)


class WeightNorm(Module):
    layer: enn.Linear
    weight_name: str = eqx.field(static=True)
    axis: int | None = eqx.field(static=True)

    def __init__(
        self,
        layer: enn.Linear,
        weight_name: str = "weight",
        axis: int | None = None,
    ):
        """
        :param layer: The layer to wrap. equinox.nn.Linear
        :param weight_name: The name of the layer's parameter (a JAX array) to apply weight normalisation to.
        :param axis: The norm is computed across every axis except this one. If `None`, compute across every axis.
        """
        self.layer = layer
        self.weight_name = weight_name
        self.axis = axis

    def _norm(self, w):
        return _norm_except_axis(
            w, norm=partial(jnp.linalg.norm, keepdims=True), axis=self.axis
        )

    @property
    def weight(self) -> Array:
        w = getattr(self.layer, self.weight_name)
        w = w / (self._norm(w) + _MIN_NORM)

        return w

    @property
    def bias(self) -> Array | None:
        return self.layer.bias

    @jax.named_scope("xfads.nn.WeightNorm")
    def __call__(self, x: Array) -> Array:
        """
        :param x: JAX Array
        """
        weight: Array = self.weight
        layer: Callable = eqx.tree_at(
            lambda layer: getattr(layer, self.weight_name), self.layer, weight
        )
        return layer(x)


class StationaryLinear(Module):
    layer: enn.Linear | WeightNorm

    def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False):
        self.layer = enn.Linear(state_dim, observation_dim, key=key, use_bias=True)

        if norm_readout:
            self.layer = WeightNorm(self.layer)

    def __call__(self, idx, x) -> Array:
        return self.layer(x)

    @property
    def weight(self) -> Array:
        return self.layer.weight


class VariantBiasLinear(Module):
    biases: Array
    layer: enn.Linear | WeightNorm

    def __init__(
        self, state_dim, observation_dim, n_biases, *, key, norm_readout: bool = False
    ):
        wkey, bkey = jrnd.split(key, 2)

        self.layer = enn.Linear(state_dim, observation_dim, key=wkey, use_bias=False)
        lim = 1 / math.sqrt(state_dim)
        self.biases = jrnd.uniform(
            bkey,
            (n_biases, observation_dim),
            dtype=self.layer.weight.dtype,
            minval=-lim,
            maxval=lim,
        )

        if norm_readout:
            self.layer = WeightNorm(self.layer)

    def __call__(self, idx, x) -> Array:
        x = self.layer(x)
        return x + self.biases[idx]

    @property
    def weight(self) -> Array:
        return self.layer.weight

    def set_static(self, static=True) -> None:
        pass


def gauss_rbf(x, c, s):
    return jnp.exp(-jnp.sum(jnp.square((x - c)) * s))


class RBFN(Module):
    centers: Array = eqx.field(static=True)
    scale: Array
    readout: Module

    def __init__(
        self, input_size, output_size, network_size, *, key, normalized: bool = False
    ):
        key, ckey = jrnd.split(key)
        self.centers = jrnd.uniform(
            ckey, shape=(network_size, input_size), minval=-1, maxval=1
        )
        self.scale = jnp.ones(input_size)
        self.readout = enn.Linear(network_size, output_size, key=key)

    def __call__(self, x):
        kernels = jax.vmap(gauss_rbf, in_axes=(None, 0, None))(
            x, self.centers, constraints.constrain_positive(self.scale)
        )
        return self.readout(kernels)  # type: ignore


# class Param(Module):
#     unconstrained: Array
#     constraint: constraints.AbstractConstraint

#     def __init__(self, value: ArrayLike, constraint: constraints.AbstractConstraint):
#         self.constraint = constraint
#         self.unconstrained = constraint.unconstrain(value)

#     def __call__(self) -> Array:
#         return self.constraint(self.unconstrained)


class DataMasker(eqx.Module, strict=True):
    p: float
    inference: bool

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
    ):
        self.p = p
        self.inference = inference

    @jax.named_scope("xfads.DataMasker")
    def __call__(
        self,
        x: Array,
        *,
        key: Array | None = None,
        inference: bool | None = None,
    ) -> tuple[Array | None, Array]:
        if inference is None:
            inference = self.inference

        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True

        shape = x.shape[:2] + (1,)  # broadcast to the last dimension
        if inference:
            return key, jnp.ones(shape)
        elif key is None:
            raise RuntimeError(
                f"{DataMasker.__name__} requires a key when running in non-deterministic mode."
            )
        else:
            key, subkey = jrnd.split(key)
            q = 1 - jax.lax.stop_gradient(self.p)
            mask = jrnd.bernoulli(key, q, shape)  # type: ignore
            return subkey, mask
