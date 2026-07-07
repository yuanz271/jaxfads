import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxfads.smoother import XFADS
from jaxfads.trainer import noise_schedule, train
import jaxfads.observations  # noqa: F401 — register GLM subclass
from conftest import MockDynamics  # noqa: F401 - class registration side-effect


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
            "dyn_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
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


def test_noise_schedule_sets_noise_free_to_scheduled_scale(model_conf, sample_data):
    """``noise_schedule`` produces a callable that replaces ``noise_free``
    with the free-form encoding of the scheduled scale, independent of the
    model's current value."""
    model = XFADS(model_conf, jax.random.key(0)).initialize(*sample_data)
    approx = model.approx

    schedule = noise_schedule(approx, q_hi=2.0, q_lo=0.005, transition_steps=100)

    updated_0 = schedule(model, jnp.array(0, dtype=jnp.int32))
    expected_0 = approx.free_from_kw(scale=2.0)
    assert jnp.allclose(updated_0.noise_free, expected_0)

    updated_end = schedule(model, jnp.array(100, dtype=jnp.int32))
    expected_end = approx.free_from_kw(scale=0.005)
    assert jnp.allclose(updated_end.noise_free, expected_end)

    # Holds at q_lo beyond transition_steps (end_value clamping).
    updated_past_end = schedule(model, jnp.array(500, dtype=jnp.int32))
    assert jnp.allclose(updated_past_end.noise_free, expected_end)


def test_train_with_param_schedule_anneals_noise_free(model_conf, sample_data):
    """``train(..., param_schedule=...)`` drives noise_free through the
    schedule during training; the final model reflects the scheduled Q, and
    the schedule dominates over whatever gradient updates would otherwise
    apply (via ``freeze_paths``)."""
    model = XFADS(model_conf, jax.random.key(0)).initialize(*sample_data)
    approx = model.approx

    max_epoch, batch_size = 5, 5
    n_batches_per_epoch = sample_data[0].shape[0] // batch_size
    total_steps = max_epoch * n_batches_per_epoch

    schedule = noise_schedule(
        approx, q_hi=2.0, q_lo=0.01, transition_steps=total_steps - 1
    )
    conf = OmegaConf.create(
        {
            "max_epoch": max_epoch,
            "batch_size": batch_size,
            "seed": 0,
            "freeze_paths": ["noise_free"],
        }
    )

    trained = train(model, sample_data, conf=conf, param_schedule=schedule)

    expected_final = approx.free_from_kw(scale=0.01)
    assert jnp.allclose(trained.noise_free, expected_final, atol=1e-5)


def test_param_schedule_without_freeze_paths_gets_fought_by_optimizer(
    model_conf, sample_data
):
    """Without ``freeze_paths``, the schedule sets ``noise_free`` at the start
    of each step, but the optimizer's own gradient-based update (plus
    gradient noise / momentum) then moves it away again within the same
    step -- so the final model does *not* end up at the scheduled value.
    This confirms ``freeze_paths`` is necessary, not merely a best practice
    (see :func:`test_train_with_param_schedule_anneals_noise_free`, which
    passes ``freeze_paths=["noise_free"]`` and lands exactly on schedule).
    """
    model = XFADS(model_conf, jax.random.key(0)).initialize(*sample_data)
    approx = model.approx

    conf = OmegaConf.create(
        {"max_epoch": 3, "batch_size": 5, "seed": 0, "freeze_paths": []}
    )
    schedule = noise_schedule(approx, q_hi=1.0, q_lo=1.0, transition_steps=1)

    trained = train(model, sample_data, conf=conf, param_schedule=schedule)
    expected = approx.free_from_kw(scale=1.0)
    assert not jnp.allclose(trained.noise_free, expected, atol=1e-5)
