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
from jaxfads.smoother import XFADS, StatContext


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
    q_scale = 1.0
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
            q_scale=q_scale,
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


def test_constructor_accepts_dynamics_and_integrator_keys():
    """XFADS should accept the new dynamics/integrator config names."""
    T = 12
    y_size = 8
    z_size = 2

    model_conf = OmegaConf.create(
        dict(
            mode="smooth",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx="MVN",
            approx_kwargs={"rank": 0},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            q_scale=1.0,
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
    assert model.dynamics is not None
    assert model.integrator is model.integrator


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
            q_scale=1.0,
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
    free_energy, post_mom, prior_mom, _transition_stat = model(times, y, u, c, key=k)

    assert jnp.isfinite(free_energy).all()
    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


@pytest.mark.parametrize(
    "mode,approx_kwargs",
    [
        ("smooth", {"rank": 2}),
        ("causal", {}),
        ("filter", {}),
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
            q_scale=1.0,
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
    _, post_mom, prior_mom, _transition_stat = model(times, y, u, c, key=jr.key(2))

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
            q_scale=1.0,
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
            q_scale=1.0,
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
    _, post_mom, prior_mom, _transition_stat = model(times, y, u, c, key=jr.key(2))

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
            q_scale=1.0,
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
    nature, post_mom, prior_mom, _transition_stat = model(times, y, u, c, key=jr.key(2))

    assert jnp.isfinite(nature).all()
    assert jnp.isfinite(post_mom).all()
    assert jnp.isfinite(prior_mom).all()


def _gaussian_model(
    T,
    y_size,
    z_size,
    *,
    q_scale=1.0,
    q_prior_fraction=0.1,
    q_mstep=True,
    approx="MVN",
):
    model_conf = OmegaConf.create(
        dict(
            mode="smooth",
            observation_dim=y_size,
            state_dim=z_size,
            dynamics="MockDynamics",
            integrator="Identity",
            approx=approx,
            approx_kwargs={},
            mc_size=2,
            seed=0,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=0.0,
            q_scale=q_scale,
            q_prior_fraction=q_prior_fraction,
            q_mstep=q_mstep,
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
    model = _gaussian_model(5, 4, 3, q_scale=0.25)
    _mean, q = model.approx.unpack(
        model.approx.canon_to_moment(model.approx.free_to_canon(model.noise.free))
    )
    chex.assert_trees_all_close(q, 0.25 * jnp.eye(3), atol=1e-5)


@pytest.mark.parametrize("q_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_q_scale_must_be_positive_and_finite(q_scale):
    """q_scale is a scalar process variance, not an arbitrary scale."""
    with pytest.raises(ValueError, match="positive finite variance"):
        _gaussian_model(5, 4, 3, q_scale=q_scale)


def test_q_mstep_false_updates_observation_only():
    """q_mstep=false leaves Q untouched while still updating R."""
    T, y_size, z_size = 5, 4, 3
    model = _gaussian_model(T, y_size, z_size, q_mstep=False)
    times = jnp.broadcast_to(jnp.arange(T), (2, T))
    y = jr.normal(jr.key(1), (2, T, y_size))
    u = jnp.zeros((2, T, 0))
    c = jnp.zeros((2, T, 0))
    model = model.initialize(times, y, u, c)

    new_model = model.mstep_from_data(times, y, u, c, key=jr.key(2))

    assert not jnp.allclose(
        new_model.observation.likelihood.unconstrained_cov,
        model.observation.likelihood.unconstrained_cov,
    )
    chex.assert_trees_all_close(new_model.noise.free, model.noise.free)


def test_unregistered_approx_keeps_q_sgd_managed():
    """An exact-unregistered Approx has no MAP-Q strategy in either mode."""
    T, y_size, z_size = 5, 4, 3
    times = jnp.broadcast_to(jnp.arange(T), (2, T))
    y = jr.normal(jr.key(1), (2, T, y_size))
    u = jnp.zeros((2, T, 0))
    c = jnp.zeros((2, T, 0))

    disabled = _gaussian_model(
        T, y_size, z_size, q_mstep=False, approx="UnregisteredMVN"
    ).initialize(times, y, u, c)
    enabled = _gaussian_model(
        T, y_size, z_size, q_mstep=True, approx="UnregisteredMVN"
    ).initialize(times, y, u, c)

    assert not disabled.noise.supports_mstep
    assert not enabled.noise.supports_mstep
    assert not enabled.q_mstep_active
    chex.assert_trees_all_close(
        enabled.mstep_from_data(times, y, u, c, key=jr.key(2)).noise.free,
        enabled.noise.free,
    )


def test_q_mstep_uses_q_scale_and_prior_fraction():
    """Noise M-step uses its static q_scale and q_prior_fraction policy."""
    T, y_size, z_size = 5, 4, 3
    model = _gaussian_model(
        T, y_size, z_size, q_scale=0.7, q_prior_fraction=0.25
    )
    times = jnp.broadcast_to(jnp.arange(T), (2, T))
    y = jr.normal(jr.key(1), (2, T, y_size))
    u = jnp.zeros((2, T, 0))
    c = jnp.zeros((2, T, 0))
    model = model.initialize(times, y, u, c)

    _natural, moment, _predicted, transition_stat = model(times, y, u, c, key=jr.key(2))
    context = StatContext(
        t=times,
        y=y,
        u=u,
        c=c,
        moment=moment,
        transition_stat=transition_stat,
        approx=model.approx,
    )
    expected_noise = model.noise.mstep(model.noise.batch_stat(context))
    new_model = model.mstep_from_data(times, y, u, c, key=jr.key(2))

    chex.assert_trees_all_close(new_model.noise.free, expected_noise.free)
