import pytest
import jax
import optax
import numpy as np
from jax import numpy as jnp, random as jrnd
import chex
from omegaconf import OmegaConf

from jaxfads.trainer import train
from jaxfads.smoother import XFADS
import jaxfads.observations  # noqa: F401 — register GLM subclass
from conftest import MockDynamics  # noqa: F401 - class registration side-effect


@pytest.mark.parametrize("kl_warmup_steps", [0, 1, 4, 10])
def test_kl_warmup_schedule_matches_legacy_formula(kl_warmup_steps):
    """The optax KL-weight curve matches the old inline beta ramp.

    Legacy: beta = where(n > 0, min(1, step/n), 1.0). New: an optax schedule
    evaluated on the loop step. Values agree within float32 tolerance (optax's
    affine ``1 - (1 - step/n)`` form differs from direct ``step/n`` by ~1 ULP;
    the warm-up-off case beta == 1.0 is exact).
    """
    schedule = (
        optax.linear_schedule(0.0, 1.0, kl_warmup_steps)
        if kl_warmup_steps > 0
        else optax.constant_schedule(1.0)
    )
    for step in [0, 1, 2, 3, 4, 5, 9, 10, 11, 50]:
        legacy = (
            min(1.0, step / kl_warmup_steps) if kl_warmup_steps > 0 else 1.0
        )
        chex.assert_trees_all_close(
            jnp.asarray(float(schedule(step))), jnp.asarray(legacy), atol=1e-6
        )


@pytest.fixture
def trainer_config():
    """Default training configuration."""
    return OmegaConf.create(
        {
            "max_epoch": 5,
            "learning_rate": 1e-3,
            "clip_norm": 5.0,
            "batch_size": 2,
            "seed": 42,
            "noise_eta": 0.5,
            "noise_gamma": 0.8,
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
            "dynamics": "MockDynamics",
            "integrator": "Identity",
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

    # This should run without errors
    trained_model = train(model, sample_data, conf=trainer_config)

    # Basic checks that we got a model back
    assert trained_model is not None
    assert hasattr(trained_model, "conf")
    assert hasattr(trained_model, "dynamics")
    assert hasattr(trained_model, "integrator")


def test_train_accepts_user_optimizer(model_conf, trainer_config, sample_data):
    """A user-supplied optax optimizer is used in place of the default."""
    trainer_config.max_epoch = 5
    trainer_config.batch_size = 64

    def run(optimizer):
        model = XFADS(model_conf, jrnd.key(0))
        return train(model, sample_data, conf=trainer_config, optimizer=optimizer)

    default = run(None)
    custom = run(optax.sgd(1e-2))  # very different update rule than the default

    # A different optimizer yields a different trained model.
    assert jnp.any(default.noise_free != custom.noise_free)


@pytest.mark.parametrize(
    "opt_name",
    ["prodigy", "adamw", "lamb", "lars"],
)
def test_train_with_params_aware_optimizer(
    model_conf, trainer_config, sample_data, opt_name
):
    """Params-aware optimizers (read current params at update) work via train().

    These previously failed: the loop passed the full model to ``update`` (bool
    leaves) and donated buffers that ``params0``-storing optimizers alias.
    """
    trainer_config.max_epoch = 4
    trainer_config.batch_size = 64
    steps = trainer_config.max_epoch * 2

    optimizers = {
        "prodigy": optax.contrib.prodigy(
            learning_rate=optax.linear_schedule(1.0, 0.0, steps)
        ),
        "adamw": optax.adamw(1e-3, weight_decay=1e-2),
        "lamb": optax.lamb(1e-3),
        "lars": optax.lars(1e-3),
    }

    model = XFADS(model_conf, jrnd.key(0))
    before = np.asarray(model.noise_free)  # snapshot: train() donates buffers
    trained = train(
        model, sample_data, conf=trainer_config, optimizer=optimizers[opt_name]
    )

    assert trained is not None
    assert jnp.all(jnp.isfinite(trained.noise_free))
    # The optimizer actually updated the model.
    assert np.any(np.asarray(trained.noise_free) != before)


def test_user_optimizer_composes_with_freeze_paths(
    model_conf, trainer_config, sample_data
):
    """``freeze_paths`` is applied on top of a user-supplied optimizer."""
    trainer_config.max_epoch = 5
    trainer_config.batch_size = 64
    trainer_config.freeze_paths = ["noise_free"]

    model = XFADS(model_conf, jrnd.key(0))
    # Snapshot to host: train() donates the input model's buffers.
    before = np.asarray(model.noise_free)
    trained = train(
        model, sample_data, conf=trainer_config, optimizer=optax.sgd(1e-1)
    )

    np.testing.assert_allclose(np.asarray(trained.noise_free), before, atol=0.0)


def test_train_lora_rank1_end_to_end(trainer_config, sample_data):
    """MVN rank-1 should train and run end-to-end without NaNs."""
    model_conf = OmegaConf.create(
        {
            "mode": "smooth",
            "observation_dim": 10,
            "state_dim": 2,
            "dynamics": "MockDynamics",
            "integrator": "Identity",
            "approx": "MVN",
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
    trainer_config.freeze_paths = ["does.not.exist"]

    with pytest.raises(ValueError, match="Invalid freeze path"):
        train(model, sample_data, conf=trainer_config)
