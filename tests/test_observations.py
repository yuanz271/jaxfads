import chex
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

from jaxfads.distributions import DiagMVN
from jaxfads.observations import DiagGaussian, Poisson


def _poisson_conf(state_dim: int, observation_dim: int):
    return OmegaConf.create(
        dict(
            state_dim=state_dim,
            observation_dim=observation_dim,
            n_steps=0,
            norm_readout=False,
        )
    )


def _gaussian_conf(state_dim: int, observation_dim: int):
    return OmegaConf.create(
        dict(
            state_dim=state_dim,
            observation_dim=observation_dim,
            cov=[1.0] * observation_dim,
            n_steps=0,
            norm_readout=False,
        )
    )


def test_poisson_eloglik_shape_and_finite():
    key = jrnd.key(0)
    state_dim = 2
    observation_dim = 3
    conf = _poisson_conf(state_dim, observation_dim)
    model = Poisson(conf, key)

    mean = jnp.zeros(state_dim)
    cov = jnp.ones(state_dim)
    moment = DiagMVN.canon_to_moment(mean, cov)
    y = jnp.ones((observation_dim,))

    ll = model.eloglik(key, jnp.array(0), moment, y, DiagMVN, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_diag_gaussian_eloglik_shape_and_finite():
    key = jrnd.key(1)
    state_dim = 2
    observation_dim = 3
    conf = _gaussian_conf(state_dim, observation_dim)
    model = DiagGaussian(conf, key)

    mean = jnp.zeros(state_dim)
    cov = jnp.ones(state_dim)
    moment = DiagMVN.canon_to_moment(mean, cov)
    y = jnp.zeros((observation_dim,))

    ll = model.eloglik(key, jnp.array(0), moment, y, DiagMVN, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)
