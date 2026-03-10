import pytest
import jax
from jax import Array, numpy as jnp, random as jrnd
import equinox as eqx
import chex
from omegaconf import OmegaConf

from jaxfads.trainer import train
from jaxfads.smoother import XFADS
from jaxfads.base import StateMap
import jaxfads.observations  # noqa: F401 — register GLM subclass


class MockStateMap(StateMap):
    """Mock dynamics for testing — pure deterministic identity."""

    layer: eqx.Module | None

    def __init__(self, conf, key):
        self.conf = conf
        self.layer = None

    def eval(self, z, u, c, *, key=None) -> Array:
        return z


@pytest.fixture
def trainer_config():
    """Default training configuration."""
    return OmegaConf.create(
        {
            "min_iter": 0,
            "max_iter": 5,
            "min_epoch": 0,
            "max_epoch": 5,
            "learning_rate": 1e-3,
            "clip_norm": 5.0,
            "batch_size": 2,
            "weight_decay": 1e-3,
            "beta": 0.95,
            "seed": 42,
            "noise_eta": 0.5,
            "noise_gamma": 0.8,
            "valid_ratio": 0.2,
            "validation_size": 2,
        }
    )


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    key = jrnd.key(42)
    n_trials = 100
    n_timesteps = 20
    obs_dim = 10
    input_dim = 1
    context_dim = 0

    times = jnp.broadcast_to(jnp.arange(n_timesteps), (n_trials, n_timesteps))
    observations = jrnd.poisson(key, jnp.ones((n_trials, n_timesteps, obs_dim)))
    controls = jrnd.normal(key, (n_trials, n_timesteps, input_dim))
    contexts = jnp.zeros((n_trials, n_timesteps, context_dim))

    return times, observations, controls, contexts


@pytest.fixture
def model_conf():
    """Create a minimal model configuration for testing."""
    return OmegaConf.create(
        {
            "mode": "smooth",
            "observation_dim": 10,
            "state_dim": 2,
            "state_map": "MockStateMap",
            "stepper": "DiscreteStepper",
            "approx": "MVN",
            "approx_kwargs": {},
            "mc_size": 1,
            "seed": 0,
            "n_steps": 10,
            "fb_penalty": 0,
            "noise_penalty": 0.01,
            "dropout": 0.0,
            "dyn_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "input_dim": 1,
                    "context_dim": 0,
                    "state_noise": 1.0,
                    "system_type": "discrete",
                }
            ),
            "enc_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "dropout": 0.0,
                }
            ),
            "obs_conf": OmegaConf.create(
                {
                    "model": "GLM",
                    "emission_noise": 1.0,
                    "norm_readout": False,
                    "dropout": 0.0,
                    "likelihood": "Poisson",
                }
            ),
        }
    )


def test_train(model_conf, trainer_config, sample_data):
    """Test that train_fast can run without errors on simple data."""
    # Create model and minimal config for fast test
    model = XFADS(model_conf, jrnd.key(0))
    trainer_config.max_epoch = 10
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32

    # This should run without errors
    trained_model = train(model, sample_data, conf=trainer_config)

    # Basic checks that we got a model back
    assert trained_model is not None
    assert hasattr(trained_model, "conf")
    assert hasattr(trained_model, "state_map")
    assert hasattr(trained_model, "stepper")


def test_train_lora_rank1_end_to_end(trainer_config, sample_data):
    """LoRaMVN rank-1 should train and run end-to-end without NaNs."""
    model_conf = OmegaConf.create(
        {
            "mode": "smooth",
            "observation_dim": 10,
            "state_dim": 2,
            "state_map": "MockStateMap",
            "stepper": "DiscreteStepper",
            "approx": "LoRaMVN",
            "approx_kwargs": {"rank": 1},
            "mc_size": 2,
            "seed": 0,
            "n_steps": 10,
            "fb_penalty": 0,
            "noise_penalty": 0.01,
            "dropout": 0.0,
            "dyn_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "input_dim": 1,
                    "context_dim": 0,
                    "state_noise": 0.1,
                    "system_type": "discrete",
                }
            ),
            "enc_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "dropout": 0.0,
                }
            ),
            "obs_conf": OmegaConf.create(
                {
                    "model": "GLM",
                    "emission_noise": 1.0,
                    "norm_readout": False,
                    "dropout": 0.0,
                    "likelihood": "Poisson",
                }
            ),
        }
    )

    model = XFADS(model_conf, jrnd.key(0))
    trainer_config.max_epoch = 5
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32

    trained_model = train(model, sample_data, conf=trainer_config)

    times, observations, controls, contexts = sample_data
    batch = (
        times[:4],
        observations[:4],
        controls[:4],
        contexts[:4],
    )
    free_energy, post_mom, prior_mom = trained_model(*batch, key=jrnd.key(1))

    assert jnp.isfinite(free_energy).all()
    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


def test_train_freeze_paths_keeps_state_noise_fixed(
    model_conf, trainer_config, sample_data
):
    """freeze_paths can freeze model.noise_free updates."""
    model = XFADS(model_conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise_free)

    trainer_config.max_epoch = 3
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32
    trainer_config.freeze_paths = ["noise_free"]
    trained_model = train(model, sample_data, conf=trainer_config)
    noise_trained = jax.device_get(trained_model.noise_free)
    chex.assert_trees_all_close(noise_trained, noise0, atol=0.0)


def test_train_freeze_paths_can_freeze_arbitrary_leaves(
    model_conf, trainer_config, sample_data
):
    """freeze_paths should freeze user-selected parameter leaves."""
    model = XFADS(model_conf, jrnd.key(0))
    prior0 = jax.device_get(model.unconstrained_prior_natural)

    trainer_config.max_epoch = 3
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32
    trainer_config.freeze_paths = ["unconstrained_prior_natural"]

    trained_model = train(model, sample_data, conf=trainer_config)
    prior_trained = jax.device_get(trained_model.unconstrained_prior_natural)
    chex.assert_trees_all_close(prior_trained, prior0, atol=0.0)


def test_train_freeze_paths_invalid_path_raises(
    model_conf, trainer_config, sample_data
):
    model = XFADS(model_conf, jrnd.key(0))
    trainer_config.max_epoch = 1
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32
    trainer_config.freeze_paths = ["does.not.exist"]

    with pytest.raises(ValueError, match="Invalid freeze path"):
        train(model, sample_data, conf=trainer_config)
