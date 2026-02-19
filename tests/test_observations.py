import chex
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

from jaxfads.distributions import DiagMVN
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
            likelihood="DiagGaussian",
        )
    )


def _make_observation(conf, key, *, likelihood="Poisson"):
    conf = conf.copy()
    conf.likelihood = likelihood
    return GLM(conf, key)


def test_poisson_eloglik_shape_and_finite():
    key = jrnd.key(0)
    state_dim = 2
    observation_dim = 3
    conf = _poisson_conf(state_dim, observation_dim)
    observation = _make_observation(conf, key, likelihood="Poisson")

    mean = jnp.zeros(state_dim)
    cov = jnp.ones(state_dim)
    moment = DiagMVN.canon_to_moment(mean, cov)
    y = jnp.ones((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), moment, y, DiagMVN, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_diag_gaussian_eloglik_shape_and_finite():
    key = jrnd.key(1)
    state_dim = 2
    observation_dim = 3
    conf = _gaussian_conf(state_dim, observation_dim)
    observation = _make_observation(conf, key, likelihood="DiagGaussian")

    mean = jnp.zeros(state_dim)
    cov = jnp.ones(state_dim)
    moment = DiagMVN.canon_to_moment(mean, cov)
    y = jnp.zeros((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), moment, y, DiagMVN, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_poisson_initialize_biases():
    key = jrnd.key(2)
    state_dim = 2
    observation_dim = 3
    time_steps = 4
    batch = 2
    t = jnp.arange(time_steps)
    y = jnp.ones((batch, time_steps, observation_dim))
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))

    conf = _poisson_conf(state_dim, observation_dim, n_steps=0)
    observation = _make_observation(conf, key, likelihood="Poisson")
    initialized = observation.initialize(t, y, u, c)
    chex.assert_trees_all_close(
        initialized.readout.layer.bias, jnp.zeros(observation_dim)
    )

    conf = _poisson_conf(state_dim, observation_dim, n_steps=time_steps)
    observation = _make_observation(conf, key, likelihood="Poisson")
    initialized = observation.initialize(t, y, u, c)
    chex.assert_trees_all_close(
        initialized.readout.biases, jnp.zeros((time_steps, observation_dim))
    )


def test_diag_gaussian_initialize_biases():
    key = jrnd.key(3)
    state_dim = 2
    observation_dim = 3
    time_steps = 4
    batch = 2
    t = jnp.arange(time_steps)
    y = jnp.zeros((batch, time_steps, observation_dim))
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))

    conf = _gaussian_conf(state_dim, observation_dim, n_steps=0)
    observation = _make_observation(conf, key, likelihood="DiagGaussian")
    initialized = observation.initialize(t, y, u, c)
    chex.assert_trees_all_close(
        initialized.readout.layer.bias, jnp.zeros(observation_dim)
    )

    conf = _gaussian_conf(state_dim, observation_dim, n_steps=time_steps)
    observation = _make_observation(conf, key, likelihood="DiagGaussian")
    initialized = observation.initialize(t, y, u, c)
    chex.assert_trees_all_close(
        initialized.readout.biases, jnp.zeros((time_steps, observation_dim))
    )
