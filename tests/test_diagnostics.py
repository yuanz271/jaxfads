"""Tests for host-side diagnostics helpers."""

import pytest
from jax import Array, numpy as jnp, random as jrnd
import equinox as eqx
from omegaconf import OmegaConf

from jaxfads.smoother import XFADS
from jaxfads.dynamics import Dynamics, Noise, DiagGaussian
from jaxfads.diagnostics import (
    LossStats,
    GradStats,
    compute_loss_stats,
    compute_grad_stats,
)


class MockDynamics(Dynamics):
    noise: Noise
    layer: eqx.Module | None

    def __init__(self, conf, key):
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
def model_and_batch():
    """Create a minimal XFADS model and a small batch for diagnostics."""
    model_conf = OmegaConf.create(
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
    model = XFADS(model_conf, jrnd.key(0))

    key = jrnd.key(42)
    n_trials = 4
    n_timesteps = 10
    obs_dim = 10
    input_dim = 1
    context_dim = 0

    times = jnp.broadcast_to(jnp.arange(n_timesteps), (n_trials, n_timesteps))
    observations = jrnd.poisson(key, jnp.ones((n_trials, n_timesteps, obs_dim)))
    controls = jrnd.normal(key, (n_trials, n_timesteps, input_dim))
    contexts = jnp.zeros((n_trials, n_timesteps, context_dim))
    batch = (times, observations, controls, contexts)

    return model, batch


def test_compute_loss_stats(model_and_batch):
    model, batch = model_and_batch
    stats = compute_loss_stats(model, batch, key=jrnd.key(123))

    assert isinstance(stats, LossStats)
    assert isinstance(stats.loss, float)
    assert stats.loss_is_finite
    assert not stats.posterior_has_nonfinite
    assert not stats.prior_has_nonfinite


def test_compute_grad_stats(model_and_batch):
    model, batch = model_and_batch
    stats = compute_grad_stats(model, batch, key=jrnd.key(456))

    assert isinstance(stats, GradStats)
    assert isinstance(stats.grad_global_norm, float)
    assert stats.grad_global_norm > 0.0
    assert not stats.grad_has_nonfinite
