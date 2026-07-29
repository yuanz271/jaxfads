from functools import partial

import chex
import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

from jaxfads.constraints import unconstrain_positive
from jaxfads.distributions import MVN
from jaxfads.observations import GLM, mstep_gaussian_cov
from jaxfads.smoother import XFADS
from conftest import MockDynamics  # noqa: F401 - class registration side-effect


def _poisson_conf(state_dim: int, observation_dim: int, *, n_steps: int = 0):
    return OmegaConf.create(
        dict(
            model="GLM",
            state_dim=state_dim,
            observation_dim=observation_dim,
            n_steps=n_steps,
            norm_readout=False,
            likelihood="Poisson",
            _approx_name="MVN",
            # Default readout initializer is "fa".
        )
    )


def _gaussian_conf(state_dim: int, observation_dim: int, *, n_steps: int = 0):
    return OmegaConf.create(
        dict(
            model="GLM",
            state_dim=state_dim,
            observation_dim=observation_dim,
            cov=[1.0] * observation_dim,
            n_steps=n_steps,
            norm_readout=False,
            likelihood="Gaussian",
            _approx_name="MVN",
            # Default readout initializer is "fa".
        )
    )


def test_poisson_eloglik_shape_and_finite():
    key = jrnd.key(0)
    state_dim = 2
    observation_dim = 3

    conf = _poisson_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    approx = MVN(dim=state_dim, rank=state_dim)
    mp = approx.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    y = jnp.ones((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), mp, y, approx, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_gaussian_eloglik_shape_and_finite():
    key = jrnd.key(1)
    state_dim = 2
    observation_dim = 3

    conf = _gaussian_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    approx = MVN(dim=state_dim, rank=state_dim)
    mp = approx.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    y = jnp.zeros((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), mp, y, approx, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_fa_default_initialize_sets_weight_and_bias():
    key = jrnd.key(2)
    state_dim = 2
    observation_dim = 6

    conf = _poisson_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    batch, time_steps = 8, 5
    t = jnp.arange(time_steps)
    y = jrnd.poisson(key, jnp.ones((batch, time_steps, observation_dim)))
    u = jnp.zeros((batch, time_steps, 0))
    c = jnp.zeros((batch, time_steps, 0))

    initialized = observation.initialize(t, y, u, c)

    # Readout dimensions should match (obs_dim, state_dim)
    chex.assert_shape(initialized.readout.weight, (observation_dim, state_dim))
    chex.assert_shape(initialized.readout.layer.bias, (observation_dim,))

    chex.assert_tree_all_finite(initialized.readout.weight)
    chex.assert_tree_all_finite(initialized.readout.layer.bias)


def test_unknown_readout_init_raises():
    key = jrnd.key(3)
    state_dim = 2
    observation_dim = 4

    conf = _poisson_conf(state_dim, observation_dim)
    conf.readout_init = "does_not_exist"
    observation = GLM(conf, key)

    batch, time_steps = 2, 3
    t = jnp.arange(time_steps)
    y = jnp.ones((batch, time_steps, observation_dim))
    u = jnp.zeros((batch, time_steps, 0))
    c = jnp.zeros((batch, time_steps, 0))

    with pytest.raises(ValueError, match="Unknown readout_init"):
        _ = observation.initialize(t, y, u, c)


def test_set_readout_stationary_smoke():
    key = jrnd.key(4)
    state_dim = 2
    observation_dim = 3

    conf = _poisson_conf(state_dim, observation_dim)
    observation = GLM(conf, key)

    weight = jnp.zeros((observation_dim, state_dim))
    bias = jnp.ones((observation_dim,))

    updated = observation.set_readout(weight=weight, bias=bias)
    chex.assert_trees_all_close(updated.readout.weight, weight)
    chex.assert_trees_all_close(updated.readout.layer.bias, bias)


def _xfads_gaussian_model(*, state_dim, observation_dim, T, seed=0):
    model_conf = OmegaConf.create(
        dict(
            mode="smooth",
            observation_dim=observation_dim,
            state_dim=state_dim,
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
            dyn_conf=OmegaConf.create(dict(input_dim=0, context_dim=0, state_noise=1.0)),
            enc_conf=OmegaConf.create(dict(width=8, depth=1, dropout=0.0)),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    cov=[1.0] * observation_dim,
                    norm_readout=False,
                    dropout=0.0,
                    likelihood="Gaussian",
                )
            ),
        )
    )
    return XFADS(model_conf, jrnd.key(seed))


def test_mstep_gaussian_cov_matches_independently_computed_residual_stat():
    """mstep_gaussian_cov's returned unconstrained_cov must correspond exactly to
    the mean of (residual^2 + propagated posterior variance) over the *original*
    model's posterior -- i.e. the closed-form M-step given the current E-step,
    not some other quantity."""
    state_dim, observation_dim, T, batch = 2, 4, 5, 3
    model = _xfads_gaussian_model(state_dim=state_dim, observation_dim=observation_dim, T=T)

    key = jrnd.key(1)
    times = jnp.broadcast_to(jnp.arange(T), (batch, T))
    y = jrnd.normal(key, (batch, T, observation_dim))
    u = jnp.zeros((batch, T, 0))
    c = jnp.zeros((batch, T, 0))
    model = model.initialize(times, y, u, c)

    # Deliberately mismatched initial covariance (mimicking a Heywood-degenerate
    # starting point) -- the M-step must correct it based on the actual residual,
    # not merely leave a value near the (wrong) prior.
    wrong_cov = jnp.full((observation_dim,), 1e-4)
    model = eqx.tree_at(
        lambda m: m.observation.likelihood.unconstrained_cov,
        model,
        unconstrain_positive(wrong_cov),
    )

    mstep_key = jrnd.key(2)
    updated = mstep_gaussian_cov(model, (times, y, u, c), key=mstep_key)

    # Independently recompute the expected M-step statistic using the same
    # (original, wrong-cov) model and the same key, mirroring mstep_gaussian_cov's
    # own aggregation but written independently here.
    approx = model.approx
    readout = model.observation.readout
    likelihood = model.observation.likelihood
    _natural, moment, _pred = model(times, y, u, c, key=mstep_key)

    stat = jax.vmap(jax.vmap(partial(likelihood.mstep_stat, approx=approx, readout=readout)))(
        times, moment, y
    )
    expected_r = jnp.mean(stat, axis=(0, 1))

    chex.assert_trees_all_close(updated.observation.likelihood.cov(), expected_r, atol=1e-4)
    # And it must differ substantially from the deliberately-wrong starting point.
    assert not jnp.allclose(updated.observation.likelihood.cov(), wrong_cov, atol=1e-2)


def test_glm_mstep_matches_mstep_gaussian_cov():
    """GLM.mstep, called directly on (t, moment, y, approx), must produce the
    same unconstrained_cov as mstep_gaussian_cov's full driver, since it's the
    same math (mean of mstep_stat), just invoked without the driver's own
    forward pass / batch_size chunking."""
    state_dim, observation_dim, T, batch = 2, 4, 5, 3
    model = _xfads_gaussian_model(state_dim=state_dim, observation_dim=observation_dim, T=T)

    key = jrnd.key(1)
    times = jnp.broadcast_to(jnp.arange(T), (batch, T))
    y = jrnd.normal(key, (batch, T, observation_dim))
    u = jnp.zeros((batch, T, 0))
    c = jnp.zeros((batch, T, 0))
    model = model.initialize(times, y, u, c)

    wrong_cov = jnp.full((observation_dim,), 1e-4)
    model = eqx.tree_at(
        lambda m: m.observation.likelihood.unconstrained_cov,
        model,
        unconstrain_positive(wrong_cov),
    )

    mstep_key = jrnd.key(2)
    via_driver = mstep_gaussian_cov(model, (times, y, u, c), key=mstep_key)

    _natural, moment, _pred = model(times, y, u, c, key=mstep_key)
    new_observation = model.observation.mstep(times, moment, y, model.approx)

    chex.assert_trees_all_close(
        new_observation.likelihood.cov(),
        via_driver.observation.likelihood.cov(),
        atol=1e-6,
    )
    assert not jnp.allclose(new_observation.likelihood.cov(), wrong_cov, atol=1e-2)


def test_glm_mstep_is_noop_for_poisson():
    """GLM.mstep must fall back to a no-op (return self unchanged) when the
    wrapped likelihood (e.g. Poisson) doesn't implement mstep -- verifying the
    internal hasattr-based dispatch, not just that Gaussian's own path works."""
    state_dim, observation_dim, T, batch = 2, 3, 4, 2
    conf = _poisson_conf(state_dim, observation_dim, n_steps=T)
    observation = GLM(conf, jrnd.key(0))

    approx = MVN(dim=state_dim, rank=state_dim)
    times = jnp.broadcast_to(jnp.arange(T), (batch, T))
    y = jrnd.poisson(jrnd.key(1), jnp.ones((batch, T, observation_dim)))
    single = approx.pack(jnp.zeros(state_dim), jnp.eye(state_dim))
    moment = jnp.broadcast_to(single, (batch, T) + single.shape)

    updated = observation.mstep(times, moment, y, approx)
    assert updated is observation


def test_glm_mstep_frozen_paths():
    """GLM.mstep_frozen_paths must report the Gaussian likelihood's frozen
    path with the correct GLM-relative nesting prefix ("likelihood."), and
    must be empty ([]) for a Poisson-backed GLM."""
    state_dim, observation_dim, T = 2, 3, 4

    gaussian_conf = _gaussian_conf(state_dim, observation_dim, n_steps=T)
    gaussian_observation = GLM(gaussian_conf, jrnd.key(0))
    assert gaussian_observation.mstep_frozen_paths() == ["likelihood.unconstrained_cov"]

    poisson_conf = _poisson_conf(state_dim, observation_dim, n_steps=T)
    poisson_observation = GLM(poisson_conf, jrnd.key(0))
    assert poisson_observation.mstep_frozen_paths() == []
