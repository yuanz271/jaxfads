from collections.abc import Callable
from functools import partial
from typing import ClassVar, Protocol

import jax
from jax import numpy as jnp, random as jrnd
from jaxtyping import Array, PRNGKeyArray, ScalarLike
import equinox as eqx

from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx


class Noise(Protocol):
    def cov(self) -> Array: ...


def predict_moment(
    z: Array, iu: Array, eu: Array, f, noise: Noise, approx: type[Approx], *, key=None
) -> Array:
    """mu[t](z[t-1])"""
    ztp1 = f(z, iu, eu, key=key)
    M2 = approx.noise_moment(noise.cov())
    moment = approx.canon_to_moment(ztp1, M2)
    return moment


def sample_expected_moment(
    key: PRNGKeyArray,
    moment: Array,
    iu: Array,
    eu: Array,
    f: Callable,
    noise: Noise,
    approx: type[Approx],
    mc_size: int,
) -> Array:
    """E[mu[t](z[t-1])]"""
    key, subkey = jrnd.split(key)
    z = approx.sample_by_moment(subkey, moment, mc_size)
    iu = jnp.broadcast_to(iu, shape=(mc_size,) + iu.shape)
    eu = jnp.broadcast_to(eu, shape=(mc_size,) + eu.shape)
    f_vmap_sample_axis = jax.vmap(
        partial(predict_moment, f=f, noise=noise, approx=approx, key=key),
        in_axes=(0, 0, 0),
    )
    moment = jnp.mean(f_vmap_sample_axis(z, iu, eu), axis=0)
    return moment


class DiagGaussian(eqx.Module, strict=True):
    unconstrained_cov: Array

    def __init__(self, cov: ScalarLike, size: int):
        self.unconstrained_cov = jnp.full(size, fill_value=unconstrain_positive(cov))

    def cov(self) -> Array:
        return constrain_positive(self.unconstrained_cov)

    # def set_static(self, static=True) -> None:
    #     self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


class AbstractDynamics(eqx.Module):
    registry: ClassVar[dict] = dict()
    noise: eqx.AbstractVar[Noise]

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        AbstractDynamics.registry[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, name: str) -> type:
        if name not in AbstractDynamics.registry:
            raise ValueError(f"Dynamics {name} is not registered.")
        return AbstractDynamics.registry[name]

    def cov(self) -> Array:
        return self.noise.cov()  # type: ignore

    def loss(self) -> ScalarLike:
        return 0.0
