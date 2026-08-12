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
from jaxfads.training import GaussianObservationMstep, MVNNoiseMstep, batch_loss, train


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
        legacy = min(1.0, step / kl_warmup_steps) if kl_warmup_steps > 0 else 1.0
        chex.assert_trees_all_close(
            jnp.asarray(float(schedule(step))), jnp.asarray(legacy), atol=1e-6
        )


@pytest.fixture
def trainer_config():
    """Default training configuration."""
    return OmegaConf.create({
        "max_epoch": 5,
        "learning_rate": 1e-3,
        "batch_size": 2,
        "seed": 42,
        "model_transformations": [],
    })


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
        "noise_penalty": 0.01,
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


def test_q_config_initialization_precedes_optimizer_init(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """The optimizer receives Q initialized from serializable trainer config."""
    model = XFADS(gaussian_model_conf, jrnd.key(0)).initialize(*gaussian_sample_data)
    expected_free = model.approx.free_from_kw(scale=0.25)
    trainer_config.model_transformations = [
        {"name": "mvn_noise", "q_scale": 0.25, "q_prior_fraction": 0.1}
    ]
    seen = []

    def init_fn(params):
        seen.append(params.noise)
        return ()

    def update_fn(updates, state, params=None):
        del params
        return updates, state

    trainer_config.max_epoch = 1
    trainer_config.batch_size = 16
    train(
        model,
        gaussian_sample_data,
        conf=trainer_config,
        optimizer=optax.GradientTransformation(init_fn, update_fn),
    )

    assert len(seen) == 1
    chex.assert_trees_all_close(seen[0], expected_free)


def test_train_with_configured_model_transformations(
    gaussian_model_conf, trainer_config, gaussian_sample_data
):
    """Configured R/Q transforms run after one optimizer forward pass."""
    conf = gaussian_model_conf
    trainer_config.model_transformations = [
        {"name": "gaussian_observation"},
        {"name": "mvn_noise", "q_scale": 1.0, "q_prior_fraction": 0.1},
    ]
    trainer_config.max_epoch = 1
    model = XFADS(conf, jrnd.key(0)).initialize(*gaussian_sample_data)
    trained = train(model, gaussian_sample_data, conf=trainer_config)
    assert jnp.isfinite(trained.observation.likelihood.cov()).all()
    assert jnp.isfinite(trained.noise).all()


def test_mvn_noise_mstep_initializes_isotropic_q(gaussian_model_conf):
    """q_scale initializes Q to q_scale times identity."""
    gaussian_model_conf.state_dim = 3
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    model = MVNNoiseMstep(q_scale=0.25).initialize(model, key=jrnd.key(1))
    approx = model.approx
    _mean, q = approx.unpack(approx.canon_to_moment(approx.free_to_canon(model.noise)))
    chex.assert_trees_all_close(q, 0.25 * jnp.eye(3), atol=1e-5)


@pytest.mark.parametrize("rank", [0, 3])
def test_mvn_noise_mstep_matches_fractional_q_prior(gaussian_model_conf, rank):
    """The Q update uses the plugin's initializer and fractional prior center."""
    d = 3
    gaussian_model_conf.state_dim = d
    gaussian_model_conf.approx_kwargs = {"rank": rank}
    plugin = MVNNoiseMstep(q_scale=0.25, q_prior_fraction=0.5)
    model = plugin.initialize(XFADS(gaussian_model_conf, jrnd.key(0)), key=jrnd.key(1))
    approx = model.approx
    mean_t = jnp.array([1.0, -2.0, 0.5])
    mean_f = jnp.zeros(d)
    cov_t = jnp.diag(jnp.array([0.2, 0.3, 0.4]))
    cov_f = jnp.diag(jnp.array([0.4, 0.5, 0.6]))
    _mean_q, q = approx.unpack(
        approx.canon_to_moment(approx.free_to_canon(model.noise))
    )
    posterior = jnp.stack((
        approx.pack(jnp.zeros(d), cov_t),
        approx.pack(mean_t, cov_t),
    ))[None]
    predictive = jnp.stack((
        approx.pack(jnp.zeros(d), cov_f + q),
        approx.pack(mean_f, cov_f + q),
    ))[None]

    updated = plugin(
        model,
        (None, None, None, None),
        (jnp.zeros_like(posterior), posterior, predictive),
        key=jrnd.key(2),
    )
    _mean, q_updated = approx.unpack(
        approx.canon_to_moment(approx.free_to_canon(updated.noise))
    )
    q_hat = jnp.outer(mean_t - mean_f, mean_t - mean_f) + cov_t + cov_f
    expected = (q_hat + 0.5 * 0.25 * jnp.eye(d)) / 1.5
    if rank == 0:
        expected = jnp.diag(jnp.diagonal(expected))
    chex.assert_trees_all_close(q_updated, expected, atol=1e-5)


@pytest.mark.parametrize("q_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_mvn_noise_mstep_q_scale_must_be_positive_and_finite(q_scale):
    """q_scale is a scalar process variance, not an arbitrary scale."""
    with pytest.raises(ValueError, match="finite and positive"):
        MVNNoiseMstep(q_scale=q_scale)


@pytest.mark.parametrize("update_rate", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_transform_update_rate_must_be_in_unit_interval(update_rate):
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        MVNNoiseMstep(update_rate=update_rate)
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        GaussianObservationMstep(update_rate=update_rate)


def test_gaussian_observation_mstep_ema_averages_variance_space(
    gaussian_model_conf, gaussian_sample_data
):
    model = XFADS(gaussian_model_conf, jrnd.key(0)).initialize(*gaussian_sample_data)
    forward = model(*gaussian_sample_data, key=jrnd.key(1))
    current = model.observation.likelihood.cov()
    direct = GaussianObservationMstep()(
        model, gaussian_sample_data, forward, key=jrnd.key(2)
    )
    target = direct.observation.likelihood.cov()
    averaged = GaussianObservationMstep(update_rate=0.25)(
        model, gaussian_sample_data, forward, key=jrnd.key(2)
    )
    expected = 0.75 * current + 0.25 * target
    chex.assert_trees_all_close(
        averaged.observation.likelihood.cov(), expected, atol=1e-5
    )


@pytest.mark.parametrize("rank", [0, 2])
def test_mvn_noise_mstep_ema_averages_covariance_space(gaussian_model_conf, rank):
    d = 2
    gaussian_model_conf.state_dim = d
    gaussian_model_conf.approx_kwargs = {"rank": rank}
    model = MVNNoiseMstep(q_scale=1.0, update_rate=0.25).initialize(
        XFADS(gaussian_model_conf, jrnd.key(0)), key=jrnd.key(1)
    )
    approx = model.approx
    current = jnp.eye(d)
    posterior = jnp.stack((
        approx.pack(jnp.zeros(d), current),
        approx.pack(jnp.zeros(d), current),
    ))[None]
    predictive = jnp.stack((
        approx.pack(jnp.zeros(d), 2.0 * current),
        approx.pack(jnp.zeros(d), 2.0 * current),
    ))[None]
    updated = MVNNoiseMstep(q_scale=1.0, q_prior_fraction=0.0, update_rate=0.25)(
        model,
        (None, None, None, None),
        (jnp.zeros_like(posterior), posterior, predictive),
        key=jrnd.key(2),
    )
    _, q = approx.unpack(approx.canon_to_moment(approx.free_to_canon(updated.noise)))
    expected = 0.75 * current + 0.25 * (2.0 * current)
    chex.assert_trees_all_close(q, expected, atol=1e-5)


def test_transform_config_rejects_unknown_name(trainer_config, model_conf, sample_data):
    trainer_config.model_transformations = [{"name": "unknown"}]
    with pytest.raises(ValueError, match="Unknown model_transformation"):
        train(XFADS(model_conf, jrnd.key(0)), sample_data, conf=trainer_config)


def test_transform_does_not_run_model_forward(
    gaussian_model_conf, gaussian_sample_data, monkeypatch
):
    model = XFADS(gaussian_model_conf, jrnd.key(0)).initialize(*gaussian_sample_data)
    calls = []
    original = XFADS.__call__

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(XFADS, "__call__", counted)
    forward = model(*gaussian_sample_data, key=jrnd.key(1))
    GaussianObservationMstep(update_rate=0.25)(
        model, gaussian_sample_data, forward, key=jrnd.key(2)
    )
    assert len(calls) == 1


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
    assert jnp.any(default.noise != custom.noise)


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
    before = np.asarray(model.noise)  # snapshot: train() donates buffers
    trained = train(
        model, sample_data, conf=trainer_config, optimizer=optimizers[opt_name]
    )

    assert trained is not None
    assert jnp.all(jnp.isfinite(trained.noise))
    # The optimizer actually updated the model.
    assert np.any(np.asarray(trained.noise) != before)


def test_user_optimizer_composes_with_freeze_paths(
    model_conf, trainer_config, sample_data
):
    """``freeze_paths`` is applied on top of a user-supplied optimizer."""
    trainer_config.max_epoch = 5
    trainer_config.batch_size = 64
    trainer_config.freeze_paths = ["noise"]
    trainer_config.model_transformations = []

    model = XFADS(model_conf, jrnd.key(0))
    # Snapshot to host: train() donates the input model's buffers.
    before = np.asarray(model.noise)
    trained = train(model, sample_data, conf=trainer_config, optimizer=optax.sgd(1e-1))

    np.testing.assert_allclose(np.asarray(trained.noise), before, atol=0.0)


def test_train_lora_rank1_end_to_end(trainer_config, sample_data):
    """MVN rank-1 should train and run end-to-end without NaNs."""
    model_conf = OmegaConf.create({
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


def test_train_freeze_paths_keeps_noise_fixed(model_conf, trainer_config, sample_data):
    """freeze_paths can freeze model.noise updates."""
    model = XFADS(model_conf, jrnd.key(0))
    noise0 = jax.device_get(model.noise)

    trainer_config.max_epoch = 3
    trainer_config.batch_size = 64
    trainer_config.freeze_paths = ["noise"]
    trainer_config.model_transformations = []
    trained_model = train(model, sample_data, conf=trainer_config)
    noise_trained = jax.device_get(trained_model.noise)
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
    """Minimal Gaussian-likelihood configuration for transform tests.

    Poisson has no free covariance for the Gaussian transform to estimate.
    """
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
        "noise_penalty": 0.01,
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
            "norm_readout": False,
            "dropout": 0.0,
            "likelihood": "Gaussian",
            "cov": [1e-4] * 10,
        }),
    })


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


def test_batch_loss_remains_scalar(gaussian_model_conf, gaussian_sample_data):
    """The public batch_loss API remains a pure scalar objective."""
    model = XFADS(gaussian_model_conf, jrnd.key(0))
    loss = batch_loss(model, gaussian_sample_data, jrnd.key(1))
    assert loss.shape == ()
