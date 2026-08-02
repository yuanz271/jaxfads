from pathlib import Path
from tempfile import TemporaryDirectory

import chex
import equinox as eqx
import pytest
from conftest import MockDynamics  # noqa: F401 - class registration side-effect
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from omegaconf import DictConfig, OmegaConf

from jaxfads.base import Encoder
from jaxfads.distributions.mvn import MVN
from jaxfads.msteps import MVNNoiseMstep
from jaxfads.smoother import XFADS


class UnregisteredMVN(MVN):
    """Exact-class Noise lookup must not inherit MVN's registered strategy."""


class IdentityEncoder(Encoder):
    """Test encoder: returns first state_dim components of y."""

    def __init__(self, conf: DictConfig, key: Array | None = None):
        del key
        self.conf = conf

    def __call__(self, y: Array, *, key: Array | None = None) -> Array:
        del key
        return y[: int(self.conf.state_dim)]


def test_constructor():
    T = 100
    y_size = 10
    z_size = 2
    u_size = 1
    mc_size = 10
    seed = 0
    likelihood = "Poisson"
    dropout = 0.5
    width = 16
    depth = 2
    emission_noise = 1.0
    normed_readout = True

    model_conf = OmegaConf.create(
        dict(
            mode="smooth",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx="MVN",
            approx_kwargs={},
            mc_size=mc_size,
            seed=seed,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=dropout,
                    dyn_conf=OmegaConf.create(
                dict(
                    width=width,
                    depth=depth,
                    linear_input_size=1,
                    dropout=dropout,
                    input_dim=u_size,
                    context_dim=0,
                        )
            ),
            enc_conf=OmegaConf.create(
                dict(
                    width=width,
                    depth=depth,
                    dropout=dropout,
                )
            ),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    emission_noise=emission_noise,
                    norm_readout=normed_readout,
                    dropout=dropout,
                    likelihood=likelihood,
                )
            ),
        )
    )

    model = XFADS(model_conf, jr.key(seed))

    # Verify noise is on the model, not on the dynamics module
    assert model.noise.free is not None
    assert not hasattr(model.dynamics, "noise.free")

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.zip"
        XFADS.save(model, path)
        loaded_model = XFADS.load(path)

        eqx.tree_equal(model, loaded_model)



def test_top_level_dims_override_subconfig_dims():
    """Top-level state/observation dims must override any sub-config values."""
    T = 7
    y_size = 5
    z_size = 3
    u_size = 2

    seed = 0

    # Intentionally provide *wrong* dims in sub-configs to ensure XFADS overrides
    # them with the top-level dimensions.
    model_conf = OmegaConf.create(
        dict(
            mode="smooth",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=seed,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(
                dict(
                    state_dim=999,
                    observation_dim=999,
                    input_dim=u_size,
                    context_dim=0,
                        )
            ),
            enc_conf=OmegaConf.create(
                dict(
                    observation_dim=999,
                    state_dim=999,
                    width=8,
                    depth=1,
                    dropout=0.0,
                )
            ),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    observation_dim=999,
                    state_dim=999,
                    emission_noise=1.0,
                    norm_readout=False,
                    dropout=0.0,
                    likelihood="Poisson",
                )
            ),
        )
    )

    model = XFADS(model_conf, jr.key(seed))

    # Ensure merged sub-configs reflect the top-level dims.
    assert int(model.dynamics.conf.state_dim) == z_size
    assert int(model.dynamics.conf.observation_dim) == y_size
    assert int(model.observation.conf.state_dim) == z_size
    assert int(model.observation.conf.observation_dim) == y_size

    # Readout weights should use (obs_dim, state_dim) from the top-level config.
    assert model.observation.readout.weight.shape == (y_size, z_size)

    # Smoke-run inference using top-level shapes.
    key = jr.key(123)
    times = jnp.broadcast_to(jnp.arange(T), (1, T))
    y = jr.poisson(key, jnp.ones((1, T, y_size)))
    u = jnp.zeros((1, T, u_size))
    c = jnp.zeros((1, T, 0))

    model = model.initialize(times, y, u, c)
    key, k = jr.split(key)
    free_energy, post_mom, prior_mom = model(times, y, u, c, key=k)

    assert jnp.isfinite(free_energy).all()
    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


@pytest.mark.parametrize(
    "mode,approx_kwargs",
    [
        ("smooth", {"rank": 2}),
        ("causal", {}),
    ],
)
def test_mode_smoke_forward_pass(mode, approx_kwargs):
    """XFADS should run end-to-end for each standard inference mode."""
    T = 5
    y_size = 4
    z_size = 3

    model_conf = OmegaConf.create(
        dict(
            mode=mode,
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx="MVN",
            approx_kwargs=approx_kwargs,
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(dict(input_dim=0, context_dim=0)),
            enc_conf=OmegaConf.create(
                dict(
                    width=8,
                    depth=1,
                    dropout=0.0,
                )
            ),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    emission_noise=1.0,
                    norm_readout=False,
                    dropout=0.0,
                    likelihood="Poisson",
                )
            ),
        )
    )

    key = jr.key(0)
    model = XFADS(model_conf, key)

    times = jnp.broadcast_to(jnp.arange(T), (1, T))
    y = jr.poisson(jr.key(1), jnp.ones((1, T, y_size)))
    u = jnp.zeros((1, T, 0))
    c = jnp.zeros((1, T, 0))

    model = model.initialize(times, y, u, c)
    _, post_mom, prior_mom = model(times, y, u, c, key=jr.key(2))

    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


def test_invalid_mode_error_lists_filter_smooth_causal():
    T = 3
    y_size = 2
    z_size = 2

    model_conf = OmegaConf.create(
        dict(
            mode="unknown",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(dict(input_dim=0, context_dim=0)),
            enc_conf=OmegaConf.create(dict(width=8, depth=1, dropout=0.0)),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    emission_noise=1.0,
                    norm_readout=False,
                    dropout=0.0,
                    likelihood="Poisson",
                )
            ),
        )
    )

    model = XFADS(model_conf, jr.key(0))
    times = jnp.broadcast_to(jnp.arange(T), (1, T))
    y = jr.poisson(jr.key(1), jnp.ones((1, T, y_size)))
    u = jnp.zeros((1, T, 0))
    c = jnp.zeros((1, T, 0))

    model = model.initialize(times, y, u, c)
    with pytest.raises(ValueError, match="filter, smooth, causal, nofilt"):
        model(times, y, u, c, key=jr.key(2))


def test_filter_mode_skips_beta_encoder():
    """mode='filter' should not evaluate beta encoder."""
    T = 4
    y_size = 3
    z_size = 2

    model_conf = OmegaConf.create(
        dict(
            mode="filter",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(dict(input_dim=0, context_dim=0)),
            enc_conf=OmegaConf.create(dict(width=8, depth=1, dropout=0.0)),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    emission_noise=1.0,
                    norm_readout=False,
                    dropout=0.0,
                    likelihood="Poisson",
                )
            ),
        )
    )

    model = XFADS(model_conf, jr.key(0))

    def _boom(_x, *, key=None):
        del key
        raise AssertionError("beta encoder should not run in filter mode")

    model = eqx.tree_at(lambda m: m.beta_encoder, model, _boom)

    times = jnp.broadcast_to(jnp.arange(T), (1, T))
    y = jr.poisson(jr.key(1), jnp.ones((1, T, y_size)))
    u = jnp.zeros((1, T, 0))
    c = jnp.zeros((1, T, 0))

    model = model.initialize(times, y, u, c)
    _, post_mom, prior_mom = model(times, y, u, c, key=jr.key(2))

    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


def test_xfads_nofilt_mode():
    """XFADS should run end-to-end with mode='nofilt' and custom Encoder."""
    T = 5
    y_size = 6
    z_size = 2

    model_conf = OmegaConf.create(
        dict(
            mode="nofilt",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            nofilt_eps=1e-6,
            dyn_conf=OmegaConf.create(dict(input_dim=0, context_dim=0)),
            enc_conf=OmegaConf.create(
                dict(alpha_encoder="IdentityEncoder", width=8, depth=1, dropout=0.0)
            ),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    likelihood="Gaussian",
                    cov=[1e-3] * y_size,
                    norm_readout=False,
                    readout_init="fa",
                    readout_init_conf=dict(obs_noise_var=0.0),
                )
            ),
        )
    )

    model = XFADS(model_conf, jr.key(0))
    assert model.beta_encoder is None

    times = jnp.broadcast_to(jnp.arange(T), (1, T))
    y = jr.normal(jr.key(1), (1, T, y_size))
    u = jnp.zeros((1, T, 0))
    c = jnp.zeros((1, T, 0))

    model = model.initialize(times, y, u, c)
    nature, post_mom, prior_mom = model(times, y, u, c, key=jr.key(2))

    assert jnp.isfinite(nature).all()
    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


def _gaussian_model(
    T,
    y_size,
    z_size,
    *,
    approx="MVN",
    rank=None,
):
    model_conf = OmegaConf.create(
        dict(
            mode="smooth",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx=approx,
            approx_kwargs={} if rank is None else {"rank": rank},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=dict(input_dim=0, context_dim=0),
            enc_conf=dict(width=8, depth=1, dropout=0.0),
            obs_conf=dict(
                model="GLM",
                likelihood="Gaussian",
                cov=[1e-3] * y_size,
                norm_readout=False,
                readout_init="fa",
                readout_init_conf=dict(obs_noise_var=0.0),
            ),
        )
    )
    return XFADS(model_conf, jr.key(0))


def test_q_scale_initializes_isotropic_q():
    """q_scale initializes Q to q_scale times identity."""
    model = _gaussian_model(5, 4, 3)
    model = MVNNoiseMstep(q_scale=0.25).initialize(model, key=jr.key(1))
    _mean, q = model.approx.unpack(
        model.approx.canon_to_moment(model.approx.free_to_canon(model.noise.free))
    )
    chex.assert_trees_all_close(q, 0.25 * jnp.eye(3), atol=1e-5)

@pytest.mark.parametrize("rank", [0, 3])
def test_mvn_noise_mstep_matches_fractional_q_prior(rank):
    """The Q update uses the plugin's initializer and fractional prior center."""
    d = 3
    plugin = MVNNoiseMstep(q_scale=0.25, q_prior_fraction=0.5)
    model = plugin.initialize(_gaussian_model(2, 4, d, rank=rank), key=jr.key(1))
    approx = model.approx
    mean_t = jnp.array([1.0, -2.0, 0.5])
    mean_f = jnp.zeros(d)
    cov_t = jnp.diag(jnp.array([0.2, 0.3, 0.4]))
    cov_f = jnp.diag(jnp.array([0.4, 0.5, 0.6]))
    _mean_q, q = approx.unpack(model.noise.moment())
    posterior = jnp.stack(
        (approx.pack(jnp.zeros(d), cov_t), approx.pack(mean_t, cov_t))
    )[None]
    predictive = jnp.stack(
        (approx.pack(jnp.zeros(d), cov_f + q), approx.pack(mean_f, cov_f + q))
    )[None]

    updated = plugin(
        model,
        (None, None, None, None),
        (jnp.zeros_like(posterior), posterior, predictive),
        key=jr.key(2),
    )
    _mean, q_updated = approx.unpack(updated.noise.moment())
    q_hat = jnp.outer(mean_t - mean_f, mean_t - mean_f) + cov_t + cov_f
    expected = (q_hat + 0.5 * 0.25 * jnp.eye(d)) / 1.5
    if rank == 0:
        expected = jnp.diag(jnp.diagonal(expected))
    chex.assert_trees_all_close(q_updated, expected, atol=1e-5)


@pytest.mark.parametrize("q_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_q_scale_must_be_positive_and_finite(q_scale):
    """q_scale is a scalar process variance, not an arbitrary scale."""
    with pytest.raises(ValueError, match="finite and positive"):
        MVNNoiseMstep(q_scale=q_scale)
