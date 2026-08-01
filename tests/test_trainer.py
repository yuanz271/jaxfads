import chex
import jax
import numpy as np
import optax
import pytest
from conftest import MockDynamics  # noqa: F401 - class registration side-effect
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

import jaxfads.observations  # noqa: F401 — register GLM subclass
from jaxfads.smoother import XFADS
from jaxfads.trainer import batch_loss, train


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
            "batch_size": 2,
            "seed": 42,
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
            "q_scale": 1.0,
            "q_mstep": False,
            "dyn_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "input_dim": 1,
                    "context_dim": 0,
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
    assert jnp.any(default.noise.free != custom.noise.free)


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
    before = np.asarray(model.noise.free)  # snapshot: train() donates buffers
    trained = train(
        model, sample_data, conf=trainer_config, optimizer=optimizers[opt_name]
    )

    assert trained is not None
    assert jnp.all(jnp.isfinite(trained.noise.free))
    # The optimizer actually updated the model.
    assert np.any(np.asarray(trained.noise.free) != before)


def test_user_optimizer_composes_with_freeze_paths(
    model_conf, trainer_config, sample_data
):
    """``freeze_paths`` is applied on top of a user-supplied optimizer."""
    trainer_config.max_epoch = 5
    trainer_config.batch_size = 64
    trainer_config.freeze_paths = ["noise.free"]

    model = XFADS(model_conf, jrnd.key(0))
    # Snapshot to host: train() donates the input model's buffers.
    before = np.asarray(model.noise.free)
    trained = train(
        model, sample_data, conf=trainer_config, optimizer=optax.sgd(1e-1)
    )

    np.testing.assert_allclose(np.asarray(trained.noise.free), before, atol=0.0)


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
            "q_scale": 0.1,
            "dyn_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "input_dim": 1,
                    "context_dim": 0,
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
    free_energy, post_mom, prior_mom, _transition_stat = trained_model(*batch, key=jrnd.key(1))

    assert jnp.isfinite(free_energy).all()
    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


def test_train_freeze_paths_keeps_noise_free_fixed(
    model_conf, trainer_config, sample_data
):
    """freeze_paths can freeze model.noise.free updates."""
    model = XFADS(model_conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise.free)

    trainer_config.max_epoch = 3
    trainer_config.batch_size = 64
    trainer_config.freeze_paths = ["noise.free"]
    trained_model = train(model, sample_data, conf=trainer_config)
    noise_trained = jax.device_get(trained_model.noise.free)
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


@pytest.fixture
def gaussian_model_conf():
    """Minimal Gaussian-likelihood model configuration, for testing the
    always-on M-step update (Poisson has no free covariance to
    estimate this way, so mstep is a no-op for it)."""
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
            "q_scale": 1.0,
            "q_mstep": False,
            "dyn_conf": OmegaConf.create(
                {
                    "width": 8,
                    "depth": 1,
                    "input_dim": 1,
                    "context_dim": 0,
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
                    "norm_readout": False,
                    "dropout": 0.0,
                    "likelihood": "Gaussian",
                    "cov": [1e-4] * 10,
                }
            ),
        }
    )


@pytest.fixture
def gaussian_sample_data():
    """Gaussian-observation sample data (sample_data uses Poisson counts)."""
    key = jrnd.key(7)
    n_trials, n_timesteps, obs_dim, input_dim, context_dim = 32, 10, 10, 1, 0
    times = jnp.broadcast_to(jnp.arange(n_timesteps), (n_trials, n_timesteps))
    observations = jrnd.normal(key, (n_trials, n_timesteps, obs_dim))
    controls = jrnd.normal(key, (n_trials, n_timesteps, input_dim))
    contexts = jnp.zeros((n_trials, n_timesteps, context_dim))
    return times, observations, controls, contexts


def test_mstep_updates_r_unconditionally(gaussian_model_conf, trainer_config, gaussian_sample_data):
    """The default epoch M-step updates R unconditionally -- R must move
    substantially away from a
    deliberately-wrong initial value after training, with no special
    configuration required."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    wrong_cov = jnp.full((10,), 1e-4)
    chex.assert_trees_all_close(
        model.observation.likelihood.cov(), wrong_cov, atol=1e-3
    )

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    new_cov = trained_model.observation.likelihood.cov()
    assert not jnp.allclose(new_cov, wrong_cov, atol=1e-2)
    chex.assert_tree_all_finite(new_cov)


def test_frozen_paths_always_excluded_from_gradients(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """model.observation.frozen_paths() must always be excluded from
    gradient updates, with no conf.freeze_paths entry or flag -- i.e. R's
    value is fully determined by mstep, not perturbed by gradient descent
    on top of it. Verified indirectly: R after training must be close to
    what an independent mstep call on the same (t, moment, y) would give
    (up to the model's inherent mc_size=1 Monte Carlo sampling noise from
    using a different PRNG key -- not exact reproduction of an internal
    implementation detail). A gradient-descent-driven fight on top of mstep
    would show up as a systematic bias much larger than this sampling
    noise, not just a few percent."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    trainer_config.max_epoch = 1
    trainer_config.batch_size = 32  # single batch == whole dataset

    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    expected_model = trained_model.mstep_from_data(
        *gaussian_sample_data, key=jrnd.key(123)
    )

    chex.assert_trees_all_close(
        trained_model.observation.likelihood.cov(),
        expected_model.observation.likelihood.cov(),
        atol=0.1,
    )


def test_mstep_composes_with_on_epoch_end(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """A user-supplied on_epoch_end must keep working unmodified, independent
    of the always-on M-step update -- no composition required
    from the caller."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    trainer_config.max_epoch = 3
    trainer_config.batch_size = 16

    epochs_seen = []

    def on_epoch_end(m, info):
        epochs_seen.append(info["epoch"])
        return False

    trained_model = train(
        model, gaussian_sample_data, conf=trainer_config, on_epoch_end=on_epoch_end
    )

    assert epochs_seen == [0, 1, 2]
    chex.assert_tree_all_finite(trained_model.observation.likelihood.cov())


def test_mstep_composes_with_user_freeze_paths(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """A user's own conf.freeze_paths entries (for an unrelated parameter)
    must keep working correctly alongside the always-derived
    frozen_paths() entries -- the two sources of frozen paths compose,
    neither overwrites the other."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise.free)

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trainer_config.freeze_paths = ["noise.free"]

    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    chex.assert_trees_all_close(
        jax.device_get(trained_model.noise.free), noise0, atol=0.0
    )
    assert not jnp.allclose(
        trained_model.observation.likelihood.cov(), jnp.full((10,), 1e-4), atol=1e-2
    )


def test_batch_loss_remains_scalar(gaussian_model_conf, gaussian_sample_data):
    """The public batch_loss API remains a pure scalar objective."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    loss = batch_loss(model, gaussian_sample_data, jrnd.key(1))
    assert loss.shape == ()


def test_accumulated_stats_match_manual_mstep_for_frozen_model(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """One frozen full-batch epoch finalizes the same R/Q statistics as the
    manual full-data mstep API."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    reference_model = XFADS(gaussian_model_conf, jrnd.key(0))
    trainer_config.max_epoch = 1
    trainer_config.batch_size = 32
    trainer_config.learning_rate = 0.0
    trained = train(
        model,
        gaussian_sample_data,
        conf=trainer_config,
        optimizer=optax.sgd(0.0),
    )

    expected = reference_model.mstep_from_data(*gaussian_sample_data, key=jrnd.key(1))
    chex.assert_trees_all_close(
        trained.observation.likelihood.cov(),
        expected.observation.likelihood.cov(),
        atol=1e-5,
    )
    chex.assert_trees_all_close(trained.noise.free, expected.noise.free, atol=1e-5)


def test_q_mstep_updates_q_and_freezes_noise_free(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """q_mstep=true updates Q through shrink and auto-freezes noise.free."""
    conf = OmegaConf.merge(gaussian_model_conf, {"q_scale": 1.0, "q_mstep": True})
    model = XFADS(conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise.free)

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    assert not jnp.allclose(jax.device_get(trained_model.noise.free), noise0, atol=1e-3)
    chex.assert_tree_all_finite(trained_model.noise.free)


def test_q_mstep_noise_free_matches_independent_mstep(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """q_mstep=true excludes noise.free from SGD, leaving the M-step value."""
    conf = OmegaConf.merge(gaussian_model_conf, {"q_scale": 1.0, "q_mstep": True})
    model = XFADS(conf, jrnd.key(0))

    trainer_config.max_epoch = 1
    trainer_config.batch_size = 32  # single batch == whole dataset

    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    t, y, u, c = gaussian_sample_data
    expected_model = trained_model.mstep_from_data(t, y, u, c, key=jrnd.key(123))

    chex.assert_trees_all_close(
        trained_model.noise.free, expected_model.noise.free, atol=0.2
    )


def test_q_mstep_false_leaves_noise_free_gradient_trained(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """q_mstep=false skips shrink and leaves noise.free SGD-managed."""
    conf = OmegaConf.merge(gaussian_model_conf, {"q_scale": 1.0, "q_mstep": False})
    model = XFADS(conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise.free)

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    assert not jnp.allclose(jax.device_get(trained_model.noise.free), noise0, atol=1e-3)


def test_accumulated_stats_finalize_before_epoch_callback(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """Callbacks/checkpoints see epoch-finalized R rather than stale R."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    initial_cov = jax.device_get(model.observation.likelihood.cov())
    seen = []

    def on_epoch_end(current_model, info):
        del info
        seen.append(current_model.observation.likelihood.cov())
        return False

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    train(model, gaussian_sample_data, conf=trainer_config, on_epoch_end=on_epoch_end)

    assert len(seen) == 2
    assert not jnp.allclose(seen[0], initial_cov, atol=1e-2)
