from functools import partial
from typing import Type, Callable

from jaxtyping import Array, PRNGKeyArray
import jax
from jax import numpy as jnp, random as jrnd
import equinox as eqx
from gearax.models import Model

from . import core, distributions, dynamics, observations, encoders
from .core import Hyperparam, Mode


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
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Array:
        
        if inference is None:
            inference = self.inference

        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        
        shape = x.shape[:2] + (1,)  # broadcast to the last dimension
        if inference:
            return key, jnp.ones(shape)
        elif key is None:
            raise RuntimeError(
                f"{self.__name__} requires a key when running in non-deterministic mode."
            )
        else:
            key, subkey = jrnd.split(key)
            q  = 1 - jax.lax.stop_gradient(self.p)
            mask = jrnd.bernoulli(key, q, shape) 
            return subkey, mask


class XFADS(Model):
    hyperparam: Hyperparam = eqx.field(init=False, static=True)
    forward: eqx.Module = eqx.field(init=False)
    backward: eqx.Module = eqx.field(init=False)
    likelihood: eqx.Module = eqx.field(init=False)
    alpha_encoder: eqx.Module = eqx.field(init=False)
    beta_encoder: eqx.Module = eqx.field(init=False)
    masker: eqx.Module = eqx.field(init=False)
    unconstrained_prior_natural: Array = eqx.field(init=False)

    def __post_init__(self):
        state_dim = self.conf.state_dim
        input_dim = self.conf.input_dim
        observation_dim = self.conf.observation_dim
        approx: str = self.conf.approx
        mc_size = self.conf.mc_size
        seed = self.conf.seed
        dropout = self.conf.dropout
        mode = self.conf.mode
        forward = self.conf.forward
        backward = self.conf.backward
        state_noise = self.conf.state_noise
        dyn_kwargs = self.conf.dyn_kwargs
        observation = self.conf.observation
        n_steps = self.conf.n_steps
        obs_kwargs = self.conf.obs_kwargs
        enc_kwargs = self.conf.enc_kwargs
        fb_penalty = self.conf.fb_penalty
        noise_penalty = self.conf.noise_penalty

        key = jrnd.key(seed) 

        self.masker: DataMasker = DataMasker(dropout)

        approx: Type[distributions.Approx] = distributions.get_class(approx)

        self.hyperparam = Hyperparam(
            approx=approx,
            state_dim=state_dim,
            input_dim=input_dim,
            observation_dim=observation_dim,
            mc_size=mc_size,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            mode=mode,
        )

        dynamics_class = dynamics.get_class(forward)
        key, subkey = jrnd.split(key)
        self.forward = dynamics_class(
            key=subkey,
            state_dim=state_dim,
            input_dim=input_dim,
            cov=state_noise,
            **dyn_kwargs,
        )
        
        if backward is not None:
            dynamics_class = dynamics.get_class(backward)
            key, subkey = jrnd.split(key)
            self.backward = dynamics_class(
                key=subkey,
                state_dim=state_dim,
                input_dim=input_dim,
                cov=state_noise,
                **dyn_kwargs,
            )
        else:
            self.backward = None

        observation_class = observations.get_class(observation)
        key, subkey = jrnd.split(key)
        self.likelihood = observation_class(
            state_dim,
            observation_dim,
            key=subkey,
            n_steps=n_steps,
            **obs_kwargs,
        )
        
        #####
        key, subkey = jrnd.split(key)
        self.alpha_encoder = encoders.AlphaEncoder(state_dim, observation_dim, approx=approx, key=subkey, **enc_kwargs)
        
        key, subkey = jrnd.split(key)
        self.beta_encoder = encoders.BetaEncoder(state_dim, approx=approx, key=subkey, **enc_kwargs)
        #####

        # if "l" in static_params:
        #     self.likelihood.set_static()
        # if "s" in static_params:
        #     self.forward.set_static()

        self.unconstrained_prior_natural = approx.unconstrain_natural(approx.prior_natural(state_dim))
    
    def init(self, t, y, u):
        mean_y = jnp.mean(y, axis=0)
        biases = jnp.log(jnp.maximum(mean_y, 1e-6))
        return eqx.tree_at(lambda model: model.likelihood.readout.biases, self, biases)
    
    def prior_natural(self) -> Array:
        return self.hyperparam.approx.constrain_natural(self.unconstrained_prior_natural)

    def __call__(self, t, y, u, *, key) -> tuple[Array, Array, Array]:
        batch_alpha_encode: Callable = jax.vmap(jax.vmap(self.alpha_encoder))
        batch_constrain_natural: Callable = jax.vmap(jax.vmap(self.hyperparam.approx.constrain_natural))
        batch_beta_encode: Callable = jax.vmap(self.beta_encoder)

        match self.hyperparam.mode:
            case Mode.BIFILTER:
                batch_smooth = jax.vmap(partial(core.bismooth, model=self))
                def batch_encode(y, key) -> Array:
                    mask_y = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
                    # chex.assert_equal_shape((y, valid_mask), dims=(0,1))
                    y = jnp.where(mask_y, y, 0)

                    key, subkey = jrnd.split(key)
                    a = batch_constrain_natural(batch_alpha_encode(y, key=jrnd.split(subkey, y.shape[:2])))
                    a = jnp.where(mask_y, a, 0)

                    key, mask_a= self.masker(a, key=key)
                    a = jnp.where(mask_a, a, 0)

                    return a
            case _:
                batch_smooth = jax.vmap(partial(core.filter, model=self))
                def batch_encode(y, key) -> Array:
                    # handling missing values
                    mask_y = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
                    # chex.assert_equal_shape((y, mask_y), dims=(0, 1))
                    y = jnp.where(mask_y, y, 0)
                    
                    key, subkey = jrnd.split(key)
                    a = batch_constrain_natural(batch_alpha_encode(y, key=jrnd.split(subkey, y.shape[:2])))
                    a = jnp.where(mask_y, a, 0)  # miss_values have no updates to state

                    key, mask_a = self.masker(y, key=key)
                    a = jnp.where(mask_a, a, 0)  # pseudo missing values
        
                    b = batch_constrain_natural(batch_beta_encode(a, key=jrnd.split(key, a.shape[0])))
                    key, mask_b = self.masker(y, key=key)
                    b = jnp.where(mask_b, b, 0)  # filter only steps

                    ab = a + b

                    # key, mask_ab = self.masker(y, key=key)
                    # ab = jnp.where(mask_ab, ab, 0)

                    return ab
                
        key, subkey = jrnd.split(key)
        alpha = batch_encode(y, subkey)

        return batch_smooth(jrnd.split(key, len(t)), t, alpha, u)
