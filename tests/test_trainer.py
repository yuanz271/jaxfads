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
from jaxfads.trainer import train


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


def test_mstep_frozen_paths_always_excluded_from_gradients(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """model.observation.mstep_frozen_paths() must always be excluded from
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

    t, y, u, c = gaussian_sample_data
    _natural, moment, _predicted, _transition_stat = trained_model(t, y, u, c, key=jrnd.key(123))
    expected_observation = trained_model.observation.mstep(t, moment, y, trained_model.approx)

    chex.assert_trees_all_close(
        trained_model.observation.likelihood.cov(),
        expected_observation.likelihood.cov(),
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
    mstep_frozen_paths() entries -- the two sources of frozen paths compose,
    neither overwrites the other."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise_free)

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trainer_config.freeze_paths = ["noise_free"]

    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    chex.assert_trees_all_close(
        jax.device_get(trained_model.noise_free), noise0, atol=0.0
    )
    assert not jnp.allclose(
        trained_model.observation.likelihood.cov(), jnp.full((10,), 1e-4), atol=1e-2
    )


def test_q_mstep_updates_q_and_freezes_noise_free(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """q_mstep=true updates Q through shrink and auto-freezes noise_free."""
    conf = OmegaConf.merge(gaussian_model_conf, {"q_scale": 1.0, "q_mstep": True})
    model = XFADS(conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise_free)

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    assert not jnp.allclose(jax.device_get(trained_model.noise_free), noise0, atol=1e-3)
    chex.assert_tree_all_finite(trained_model.noise_free)


def test_q_mstep_noise_free_matches_independent_mstep(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """q_mstep=true excludes noise_free from SGD, leaving the M-step value."""
    conf = OmegaConf.merge(gaussian_model_conf, {"q_scale": 1.0, "q_mstep": True})
    model = XFADS(conf, jrnd.key(0))

    trainer_config.max_epoch = 1
    trainer_config.batch_size = 32  # single batch == whole dataset

    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    t, y, u, c = gaussian_sample_data
    expected_model = trained_model.mstep(t, y, u, c, key=jrnd.key(123))

    chex.assert_trees_all_close(
        trained_model.noise_free, expected_model.noise_free, atol=0.2
    )


def test_q_mstep_false_leaves_noise_free_gradient_trained(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """q_mstep=false skips shrink and leaves noise_free SGD-managed."""
    conf = OmegaConf.merge(gaussian_model_conf, {"q_scale": 1.0, "q_mstep": False})
    model = XFADS(conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise_free)

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trained_model = train(model, gaussian_sample_data, conf=trainer_config)

    assert not jnp.allclose(jax.device_get(trained_model.noise_free), noise0, atol=1e-3)


def test_default_mstep_mode_is_epoch_no_redundant_final_call(
    monkeypatch, gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """mstep_mode="epoch" must call model.mstep(...) exactly once per
    epoch (max_epoch times total on normal completion) -- not once more
    per epoch plus one redundant, unconditional final call duplicating
    the last epoch's already-fresh full-dataset update."""
    from jaxfads.smoother import XFADS as XFADSClass

    call_count = 0
    original_mstep = XFADSClass.mstep

    def counting_mstep(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_mstep(self, *args, **kwargs)

    monkeypatch.setattr(XFADSClass, "mstep", counting_mstep)

    model = XFADS(gaussian_model_conf, jrnd.key(0))
    trainer_config.max_epoch = 3
    trainer_config.batch_size = 32  # single batch per epoch

    train(model, gaussian_sample_data, conf=trainer_config)

    assert call_count == 3, (
        f"expected exactly 3 model.mstep(...) calls (one per epoch, no "
        f"redundant final call), got {call_count}"
    )


def test_mstep_minibatch_mode_final_call_not_skipped(
    monkeypatch, gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """mstep_mode="minibatch" must still get the guaranteed full-dataset
    final call -- the mstep_stale tracking that skips the redundant call
    in "epoch" mode must not also (incorrectly) skip it here, since
    finalize_epoch never calls apply_mstep in "minibatch" mode at all."""
    from jaxfads.smoother import XFADS as XFADSClass

    full_dataset_call_count = 0
    original_mstep = XFADSClass.mstep
    t_full = gaussian_sample_data[0]

    def counting_mstep(self, t, y, u, c, **kwargs):
        nonlocal full_dataset_call_count
        if t.shape[0] == t_full.shape[0]:
            full_dataset_call_count += 1
        return original_mstep(self, t, y, u, c, **kwargs)

    monkeypatch.setattr(XFADSClass, "mstep", counting_mstep)

    model = XFADS(gaussian_model_conf, jrnd.key(0))
    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16  # 2 minibatches per epoch, batch-scoped calls

    train(model, gaussian_sample_data, conf=trainer_config, mstep_mode="minibatch")

    assert full_dataset_call_count == 1, (
        f"expected exactly 1 full-dataset-scope model.mstep(...) call (the "
        f"guaranteed final one; per-minibatch calls are batch-scoped, not "
        f"full-dataset), got {full_dataset_call_count}"
    )


def test_mstep_mode_epoch_updates_only_at_epoch_boundaries(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """mstep_mode='epoch' must not update R during a given epoch's minibatch
    training -- R stays at its initial value until that epoch's own
    end-of-epoch mstep call, confirmed via on_epoch_end (which fires before
    that epoch's own mstep update). By the end of training, R has been
    updated."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    # Capture the actual initial cov() (not the raw config value -- cov()
    # always adds _MIN_VARIANCE on top of the constrained config value).
    # device_get: train()'s donate="all" invalidates model's input buffers,
    # so materialize this value on host first (same convention as
    # test_train_freeze_paths_keeps_noise_free_fixed's noise0 capture).
    wrong_cov = jax.device_get(model.observation.likelihood.cov())

    seen = []

    def on_epoch_end(m, info):
        seen.append(m.observation.likelihood.cov())
        return False

    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16
    trained_model = train(
        model,
        gaussian_sample_data,
        conf=trainer_config,
        on_epoch_end=on_epoch_end,
        mstep_mode="epoch",
    )

    # Epoch 0's on_epoch_end fires before epoch 0's own end-of-epoch mstep
    # update; gradient descent is always frozen off this path, so nothing
    # else could have touched R -- it must be exactly unchanged.
    chex.assert_trees_all_close(seen[0], wrong_cov, atol=0.0)

    assert not jnp.allclose(
        trained_model.observation.likelihood.cov(), wrong_cov, atol=1e-2
    )


@pytest.mark.parametrize("mstep_mode", ["minibatch", "epoch"])
def test_mstep_applied_after_final_epoch_regardless_of_mode(
    mstep_mode, gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """Regardless of mstep_mode, mstep must be applied once more, from a
    full-dataset forward pass, immediately after the final epoch's gradient
    steps complete -- verified as a fixed point: an independent mstep call
    on the returned model must reproduce (approximately) the same R value,
    within the model's inherent mc_size=1 Monte Carlo sampling tolerance
    (same reasoning as test_mstep_frozen_paths_always_excluded_from_gradients)."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    trainer_config.max_epoch = 2
    trainer_config.batch_size = 16

    trained_model = train(
        model, gaussian_sample_data, conf=trainer_config, mstep_mode=mstep_mode
    )

    t, y, u, c = gaussian_sample_data
    _natural, moment, _predicted, _transition_stat = trained_model(t, y, u, c, key=jrnd.key(321))
    expected_observation = trained_model.observation.mstep(t, moment, y, trained_model.approx)

    chex.assert_trees_all_close(
        trained_model.observation.likelihood.cov(),
        expected_observation.likelihood.cov(),
        atol=0.1,
    )
