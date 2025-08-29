import pytest
from jax import Array, numpy as jnp, random as jrnd
import equinox as eqx
from omegaconf import OmegaConf

from jaxfads.trainer import train_fast, train, train_xfads
from jaxfads.smoother import XFADS
from jaxfads.dynamics import Dynamics, Noise


class MockDynamics(Dynamics):
    """Mock dynamics for testing."""

    noise: Noise
    layer: eqx.Module | None

    def __init__(self, conf, key):
        from jaxfads.dynamics import DiagGaussian

        self.conf = conf
        state_dim = self.conf.state_dim
        state_noise = self.conf.state_noise
        self.noise = DiagGaussian(jnp.array(state_noise), state_dim)
        self.layer = None

    def forward(self, z, u, c, *, key=None) -> Array:
        return z

    def loss(self):
        return 0.0


@pytest.fixture
def trainer_config():
    """Default training configuration."""
    return OmegaConf.create(dict(
        min_iter=0,
        max_iter=5,
        min_epoch=0,
        max_epoch=5,
        learning_rate=1e-3,
        clip_norm=5.0,
        batch_size=2,
        weight_decay=1e-3,
        beta=0.95,
        seed=42,
        noise_eta=0.5,
        noise_gamma=0.8,
        valid_ratio=0.2,
        validation_size=2,
    ))


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
            "mode": "pseudo",
            "observation_dim": 10,
            "state_dim": 2,
            "forward": "MockDynamics",
            "approx": "DiagMVN",
            "mc_size": 1,
            "seed": 0,
            "observation": "Poisson",
            "n_steps": 10,
            "fb_penalty": 0,
            "noise_penalty": 0.01,
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
                    "approx": "DiagMVN",
                }
            ),
            "obs_conf": OmegaConf.create(
                {
                    "observation_dim": 10,
                    "state_dim": 2,
                    "emission_noise": 1.0,
                    "norm_readout": False,
                    "dropout": 0.0,
                }
            ),
        }
    )


def test_train_fast(model_conf, trainer_config, sample_data):
    """Test that train_fast can run without errors on simple data."""
    # Create model and minimal config for fast test
    model = XFADS(model_conf, jrnd.key(0))
    trainer_config.max_iter = 5
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32

    # This should run without errors
    trained_model = train_fast(model, sample_data, conf=trainer_config)

    # Basic checks that we got a model back
    assert trained_model is not None
    assert hasattr(trained_model, "conf")
    assert hasattr(trained_model, "forward")


def test_train(model_conf, trainer_config, sample_data):
    """Test that train_fast can run without errors on simple data."""
    # Create model and minimal config for fast test
    model = XFADS(model_conf, jrnd.key(0))
    trainer_config.max_iter = 5
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32

    # This should run without errors
    trained_model = train(model, sample_data, conf=trainer_config)

    # Basic checks that we got a model back
    assert trained_model is not None
    assert hasattr(trained_model, "conf")
    assert hasattr(trained_model, "forward")


def test_train_xfads(model_conf, trainer_config, sample_data):
    """Test that train_fast can run without errors on simple data."""
    # Create model and minimal config for fast test
    model = XFADS(model_conf, jrnd.key(0))
    trainer_config.max_epoch = 10
    trainer_config.batch_size = 64
    trainer_config.validation_size = 32

    # This should run without errors
    trained_model = train_xfads(model, sample_data, conf=trainer_config)

    # Basic checks that we got a model back
    assert trained_model is not None
    assert hasattr(trained_model, "conf")
    assert hasattr(trained_model, "forward")
