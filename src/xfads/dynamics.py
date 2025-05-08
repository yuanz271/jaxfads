from functools import partial
from typing import ClassVar, Protocol, Type

import jax
from jax import numpy as jnp, random as jrnd
from jaxtyping import Array, PRNGKeyArray, Scalar
import equinox as eqx

from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx


class Noise(Protocol):
    def cov(self): ...


def predict_moment(
    z: Array, u: Array, c: Array, f, noise: Noise, approx: Type[Approx], *, key=None
) -> Array:
    """mu[t](z[t-1])"""
    ztp1 = f(z, u, c, key=key)
    M2 = approx.noise_moment(noise.cov())
    moment = approx.canon_to_moment(ztp1, M2)
    return moment


def sample_expected_moment(
    key: PRNGKeyArray,
    moment: Array,
    u: Array,
    c: Array,
    f,
    noise: Noise,
    approx: Type[Approx],
    mc_size: int,
) -> Array:
    """E[mu[t](z[t-1])]"""
    key, subkey = jrnd.split(key)
    z = approx.sample_by_moment(subkey, moment, mc_size)
    u_shape = (mc_size,) + u.shape
    c_shape = (mc_size,) + c.shape
    u = jnp.broadcast_to(u, shape=u_shape)
    c = jnp.broadcast_to(c, shape=c_shape)
    f_vmap_sample_axis = jax.vmap(
        partial(predict_moment, f=f, noise=noise, approx=approx, key=key), in_axes=(0, 0, 0)
    )
    moment = jnp.mean(f_vmap_sample_axis(z, u, c), axis=0)
    return moment


class DiagGaussian(eqx.Module, strict=True):
    unconstrained_cov: Array
    
    def __init__(self, cov, size):
        self.unconstrained_cov = jnp.full(size, fill_value=unconstrain_positive(cov))

    def cov(self) -> Array:
        return constrain_positive(self.unconstrained_cov)

    # def set_static(self, static=True) -> None:
    #     self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


class AbstractDynamics(eqx.Module):
    registry: ClassVar[dict] = dict()
    noise: eqx.AbstractVar[eqx.Module]

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        AbstractDynamics.registry[cls.__name__] = cls

    def cov(self) -> Array:
        return self.noise.cov()
    
    def loss(self) -> Scalar:
        return 0.


def get_class(name) -> Type:
    return AbstractDynamics.registry[name]
