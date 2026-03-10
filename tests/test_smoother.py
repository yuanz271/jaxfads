from typing import override
from pathlib import Path
from tempfile import TemporaryDirectory

from jax import Array, random as jr
from jax import numpy as jnp
from omegaconf import OmegaConf, DictConfig
import equinox as eqx
import pytest
from jaxfads.base import StateMap
from jaxfads.smoother import XFADS


class Mock(StateMap):
    """Mock dynamics — pure deterministic transition."""

    layer: eqx.Module | None

    def __init__(self, conf: DictConfig, key: Array):
        self.conf = conf
        self.layer = None

    @override
    def eval(self, z: Array, u: Array, c: Array, *, key: Array | None = None) -> Array:
        return z


def test_constructor():
    T = 100
    y_size = 10
    z_size = 2
    u_size = 1
    state_noise = 1.0
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
            state_map="Mock",
            stepper="DiscreteStepper",
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
                    state_noise=state_noise,
                    system_type="discrete",
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

    # Verify noise is on the model, not on state-map module
    assert model.noise_free is not None
    assert not hasattr(model.state_map, "noise_free")

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
            state_map="Mock",
            stepper="DiscreteStepper",
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
                    state_noise=1.0,
                    system_type="discrete",
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
    assert int(model.state_map.conf.state_dim) == z_size
    assert int(model.state_map.conf.observation_dim) == y_size
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


def test_low_rank_mvn_smoke_constructor_and_forward_pass():
    """XFADS should run end-to-end with low-rank MVN encoder outputs."""
    T = 5
    y_size = 4
    z_size = 3

    model_conf = OmegaConf.create(
        dict(
            mode="smooth",
            observation_dim=y_size,
            state_dim=z_size,
            state_map="Mock",
            stepper="DiscreteStepper",
            approx="MVN",
            approx_kwargs={"rank": 2},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(
                dict(
                    input_dim=0,
                    context_dim=0,
                    state_noise=1.0,
                    system_type="discrete",
                )
            ),
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


def test_causal_mode_smoke_constructor_and_forward_pass():
    """XFADS should run end-to-end with mode='causal'."""
    T = 5
    y_size = 4
    z_size = 3

    model_conf = OmegaConf.create(
        dict(
            mode="causal",
            observation_dim=y_size,
            state_dim=z_size,
            state_map="Mock",
            stepper="DiscreteStepper",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(
                dict(
                    input_dim=0,
                    context_dim=0,
                    state_noise=1.0,
                    system_type="discrete",
                )
            ),
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


def test_filter_mode_smoke_constructor_and_forward_pass():
    """XFADS should run end-to-end with mode='filter'."""
    T = 5
    y_size = 4
    z_size = 3

    model_conf = OmegaConf.create(
        dict(
            mode="filter",
            observation_dim=y_size,
            state_dim=z_size,
            state_map="Mock",
            stepper="DiscreteStepper",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(
                dict(
                    input_dim=0,
                    context_dim=0,
                    state_noise=1.0,
                    system_type="discrete",
                )
            ),
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
            state_map="Mock",
            stepper="DiscreteStepper",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(
                dict(
                    input_dim=0,
                    context_dim=0,
                    state_noise=1.0,
                    system_type="discrete",
                )
            ),
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
    with pytest.raises(ValueError, match="filter, smooth, causal"):
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
            state_map="Mock",
            stepper="DiscreteStepper",
            approx="MVN",
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            dyn_conf=OmegaConf.create(
                dict(
                    input_dim=0,
                    context_dim=0,
                    state_noise=1.0,
                    system_type="discrete",
                )
            ),
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
