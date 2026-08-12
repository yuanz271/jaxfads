import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from conftest import MockDynamics  # noqa: F401 - class registration side-effect
from omegaconf import OmegaConf

import jaxfads.observations  # noqa: F401 — register GLM subclass
from jaxfads.smoother import XFADS
from jaxfads.training import batch_loss, train


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


def test_regularizer_adds_its_gradient(model_conf, sample_data):
    """The trainer composes ``loss = -ELBO + regularizer(model)``.

    ``batch_loss`` is a pure objective; the penalty is added in ``train``'s
    ``loss_fn``. Here we replicate that composition and check the regularizer
    contributes exactly its own gradient (additive composition).
    """
    model = XFADS(model_conf, jax.random.key(0))
    model = model.initialize(*sample_data)

    key = jax.random.key(1)
    lam = jnp.array(1e-3)

    def l2_reg(m):
        return lam * jnp.sum(m.noise**2)

    g_obj = eqx.filter_grad(lambda m: batch_loss(m, sample_data, key, beta=1.0))(model)
    g_both = eqx.filter_grad(
        lambda m: batch_loss(m, sample_data, key, beta=1.0) + l2_reg(m)
    )(model)
    g_reg = eqx.filter_grad(l2_reg)(model)

    chex.assert_trees_all_close((g_both.noise - g_obj.noise), g_reg.noise, atol=1e-6)


def test_train_applies_regularizer(model_conf, sample_data):
    """A regularizer passed to ``train`` is wired into the optimized loss."""
    conf = OmegaConf.create({
        "max_epoch": 3,
        "batch_size": 5,
        "seed": 0,
        "model_transformations": [],
    })

    def strong_reg(m):
        return 1e2 * jnp.sum(m.noise**2)

    # train() donates its input model's buffers, so use a fresh (identical)
    # model for each run.
    def fresh_model():
        return XFADS(model_conf, jax.random.key(0)).initialize(*sample_data)

    base = train(fresh_model(), sample_data, conf=conf)
    reg = train(fresh_model(), sample_data, conf=conf, regularizer=strong_reg)

    # A strong penalty on noise changes the optimization outcome.
    assert jnp.any(base.noise != reg.noise)
