import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from conftest import MockDynamics  # noqa: F401 - class registration side-effect
from omegaconf import OmegaConf

import jaxfads.observations  # noqa: F401 — register GLM subclass
from jaxfads.smoother import XFADS
from jaxfads.trainer import train


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
    return OmegaConf.create({
        "mode": "smooth",
        "observation_dim": 10,
        "state_dim": 2,
        "dynamics": "MockDynamics",
        "integrator": "Identity",
        "approx": "MVN",
        "approx_kwargs": {},
        "mc_size": 1,
        "seed": 0,
        "n_steps": 10,
        "fb_penalty": 0,
        "noise_penalty": 0.0,
        "dropout": 0.0,
        "dyn_conf": OmegaConf.create({
            "width": 8,
            "depth": 1,
            "input_dim": 1,
            "context_dim": 0,
        }),
        "enc_conf": OmegaConf.create({
            "width": 8,
            "depth": 1,
            "dropout": 0.0,
        }),
        "obs_conf": OmegaConf.create({
            "model": "GLM",
            "emission_noise": 1.0,
            "norm_readout": False,
            "dropout": 0.0,
            "likelihood": "Poisson",
        }),
    })


def test_train_with_param_schedule_anneals_noise(model_conf, sample_data):
    """``train(..., param_schedule=...)`` drives noise through the
    schedule during training; the final model reflects the scheduled Q, and
    the schedule dominates over whatever gradient updates would otherwise
    apply (via ``freeze_paths``)."""
    model = XFADS(model_conf, jax.random.key(0)).initialize(*sample_data)
    approx = model.approx

    max_epoch, batch_size = 5, 5
    n_batches_per_epoch = sample_data[0].shape[0] // batch_size
    total_steps = max_epoch * n_batches_per_epoch

    decay = optax.exponential_decay(2.0, total_steps - 1, 0.01 / 2.0, end_value=0.01)

    def schedule(m, step):
        return eqx.tree_at(lambda x: x.noise, m, approx.free_from_kw(scale=decay(step)))

    conf = OmegaConf.create({
        "max_epoch": max_epoch,
        "batch_size": batch_size,
        "seed": 0,
        "freeze_paths": ["noise"],
    })

    trained = train(model, sample_data, conf=conf, param_schedule=schedule)

    expected_final = approx.free_from_kw(scale=0.01)
    assert jnp.allclose(trained.noise, expected_final, atol=1e-5)


def test_param_schedule_without_freeze_paths_gets_fought_by_optimizer(
    model_conf, sample_data
):
    """Without ``freeze_paths``, the schedule sets ``noise`` at the start
    of each step, but the optimizer's own gradient-based update (plus
    gradient noise / momentum) then moves it away again within the same
    step -- so the final model does *not* end up at the scheduled value.
    This confirms ``freeze_paths`` is necessary, not merely a best practice
    (see :func:`test_train_with_param_schedule_anneals_noise`, which
    passes ``freeze_paths=["noise"]`` and lands exactly on schedule).
    """
    model = XFADS(model_conf, jax.random.key(0)).initialize(*sample_data)
    approx = model.approx

    conf = OmegaConf.create({
        "max_epoch": 3,
        "batch_size": 5,
        "seed": 0,
        "freeze_paths": [],
    })

    def schedule(m, step):
        del step
        return eqx.tree_at(lambda x: x.noise, m, approx.free_from_kw(scale=1.0))

    trained = train(model, sample_data, conf=conf, param_schedule=schedule)
    expected = approx.free_from_kw(scale=1.0)
    assert not jnp.allclose(trained.noise, expected, atol=1e-5)
