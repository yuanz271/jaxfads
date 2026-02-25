import jax
import jax.numpy as jnp
import chex
import pytest
from omegaconf import OmegaConf

from jaxfads.smoother import XFADS
from jaxfads.trainer import batch_loss
import jaxfads.observations  # noqa: F401 — register GLM subclass


@pytest.fixture
def sample_data():
    key = jax.random.key(42)
    n_trials = 20
    n_timesteps = 10
    obs_dim = 10
    input_dim = 1
    context_dim = 0

    times = jnp.broadcast_to(jnp.arange(n_timesteps), (n_trials, n_timesteps))
    observations = jax.random.poisson(key, jnp.ones((n_trials, n_timesteps, obs_dim)))
    controls = jax.random.normal(key, (n_trials, n_timesteps, input_dim))
    contexts = jnp.zeros((n_trials, n_timesteps, context_dim))

    return times, observations, controls, contexts


@pytest.fixture
def model_conf():
    return OmegaConf.create(
        {
            "mode": "pseudo",
            "observation_dim": 10,
            "state_dim": 2,
            "forward": "MockDynamics",
            "approx": "MVN",
            "approx_kwargs": {},
            "mc_size": 1,
            "seed": 0,
            "n_steps": 10,
            "fb_penalty": 0,
            "noise_penalty": 0.0,
            "dropout": 0.0,
            "dyn_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "observation_dim": 10,
                    "state_dim": 2,
                    "input_dim": 1,
                    "context_dim": 0,
                    "state_noise": 1.0,
                }
            ),
            "enc_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "dropout": 0.0,
                    "observation_dim": 10,
                    "state_dim": 2,
                }
            ),
            "obs_conf": OmegaConf.create(
                {
                    "model": "GLM",
                    "observation_dim": 10,
                    "state_dim": 2,
                    "emission_noise": 1.0,
                    "norm_readout": False,
                    "dropout": 0.0,
                    "likelihood": "Poisson",
                }
            ),
        }
    )


def test_noise_regularizer_is_added(model_conf, sample_data):
    model = XFADS(model_conf, jax.random.key(0))
    model = model.initialize(*sample_data)

    key = jax.random.key(1)
    step = jnp.array(0, dtype=jnp.int32)

    base = batch_loss(model, sample_data, key, step, noise_regularizer=None)

    lam = jnp.array(1e-3, dtype=base.dtype)

    def l2_reg(m):
        return lam * jnp.sum(m.noise_free**2)

    reg = batch_loss(model, sample_data, key, step, noise_regularizer=l2_reg)

    chex.assert_trees_all_close(reg - base, l2_reg(model), atol=1e-6)
