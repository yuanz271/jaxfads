import chex
import pytest
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

from jaxfads.distributions import MVN
from jaxfads.observations import GLM


def _poisson_conf(state_dim: int, observation_dim: int, *, n_steps: int = 0):
    return OmegaConf.create(
        dict(
            model="GLM",
            state_dim=state_dim,
            observation_dim=observation_dim,
            n_steps=n_steps,
            norm_readout=False,
            likelihood="Poisson",
            # Default readout initializer is "fa".
        )
    )


def _gaussian_conf(state_dim: int, observation_dim: int, *, n_steps: int = 0):
    return OmegaConf.create(
        dict(
            model="GLM",
            state_dim=state_dim,
            observation_dim=observation_dim,
            cov=[1.0] * observation_dim,
            n_steps=n_steps,
            norm_readout=False,
            likelihood="Gaussian",
            # Default readout initializer is "fa".
        )
    )


def test_poisson_eloglik_shape_and_finite():
    key = jrnd.key(0)
    state_dim = 2
    observation_dim = 3

    conf = _poisson_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    approx = MVN(dim=state_dim)
    mp = approx.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    y = jnp.ones((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), mp, y, approx, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_gaussian_eloglik_shape_and_finite():
    key = jrnd.key(1)
    state_dim = 2
    observation_dim = 3

    conf = _gaussian_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    approx = MVN(dim=state_dim)
    mp = approx.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    y = jnp.zeros((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), mp, y, approx, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_fa_default_initialize_sets_weight_and_bias():
    key = jrnd.key(2)
    state_dim = 2
    observation_dim = 6

    conf = _poisson_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    batch, time_steps = 8, 5
    t = jnp.arange(time_steps)
    y = jrnd.poisson(key, jnp.ones((batch, time_steps, observation_dim)))
    u = jnp.zeros((batch, time_steps, 0))
    c = jnp.zeros((batch, time_steps, 0))

    initialized = observation.initialize(t, y, u, c)

    # Readout dimensions should match (obs_dim, state_dim)
    chex.assert_shape(initialized.readout.weight, (observation_dim, state_dim))
    chex.assert_shape(initialized.readout.layer.bias, (observation_dim,))

    chex.assert_tree_all_finite(initialized.readout.weight)
    chex.assert_tree_all_finite(initialized.readout.layer.bias)


def test_unknown_readout_init_raises():
    key = jrnd.key(3)
    state_dim = 2
    observation_dim = 4

    conf = _poisson_conf(state_dim, observation_dim)
    conf.readout_init = "does_not_exist"
    observation = GLM(conf, key)

    batch, time_steps = 2, 3
    t = jnp.arange(time_steps)
    y = jnp.ones((batch, time_steps, observation_dim))
    u = jnp.zeros((batch, time_steps, 0))
    c = jnp.zeros((batch, time_steps, 0))

    with pytest.raises(ValueError, match="Unknown readout_init"):
        _ = observation.initialize(t, y, u, c)


def test_set_readout_stationary_smoke():
    key = jrnd.key(4)
    state_dim = 2
    observation_dim = 3

    conf = _poisson_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    weight = jnp.zeros((observation_dim, state_dim))
    bias = jnp.ones((observation_dim,))

    updated = observation.set_readout(weight=weight, bias=bias)
    chex.assert_trees_all_close(updated.readout.weight, weight)
    chex.assert_trees_all_close(updated.readout.layer.bias, bias)
