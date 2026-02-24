import chex
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

from jaxfads.distributions import MVN
from jaxfads.observations import GLM, register_readout_init

_diag = MVN(rank=0)
_full = MVN(rank=-1)


def _poisson_conf(state_dim: int, observation_dim: int, *, n_steps: int = 0):
    return OmegaConf.create(
        dict(
            model="GLM",
            state_dim=state_dim,
            observation_dim=observation_dim,
            n_steps=n_steps,
            norm_readout=False,
            likelihood="Poisson",
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
        )
    )


def _make_observation(conf, key, *, likelihood="Poisson"):
    conf = conf.copy()
    conf.likelihood = likelihood
    return GLM(conf, key)


def test_poisson_eloglik_shape_and_finite():
    key = jrnd.key(0)
    state_dim = 2
    observation_dim = 3
    conf = _poisson_conf(state_dim, observation_dim)
    observation = _make_observation(conf, key, likelihood="Poisson")

    mean = jnp.zeros(state_dim)
    cov = jnp.ones(state_dim)
    moment = _diag.canon_to_moment(mean, cov)
    y = jnp.ones((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), moment, y, _diag, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_poisson_eloglik_full_mvn():
    key = jrnd.key(10)
    state_dim = 2
    observation_dim = 3
    conf = _poisson_conf(state_dim, observation_dim)
    observation = _make_observation(conf, key, likelihood="Poisson")

    mean = jnp.zeros(state_dim)
    cov = jnp.eye(state_dim)
    moment = _full.canon_to_moment(mean, cov)
    y = jnp.ones((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), moment, y, _full, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_diag_gaussian_eloglik_shape_and_finite():
    key = jrnd.key(1)
    state_dim = 2
    observation_dim = 3
    conf = _gaussian_conf(state_dim, observation_dim)
    observation = _make_observation(conf, key, likelihood="Gaussian")

    mean = jnp.zeros(state_dim)
    cov = jnp.ones(state_dim)
    moment = _diag.canon_to_moment(mean, cov)
    y = jnp.zeros((observation_dim,))

    ll = observation.eloglik(key, jnp.array(0), moment, y, _diag, mc_size=1)
    chex.assert_shape(ll, ())
    chex.assert_tree_all_finite(ll)


def test_poisson_initialize_biases():
    key = jrnd.key(2)
    state_dim = 2
    observation_dim = 3
    time_steps = 4
    batch = 2
    t = jnp.arange(time_steps)
    y = jnp.ones((batch, time_steps, observation_dim))
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))

    conf = _poisson_conf(state_dim, observation_dim, n_steps=0)
    observation = _make_observation(conf, key, likelihood="Poisson")
    initialized = observation.initialize(t, y, u, c)
    # Bias is log(mean) via Poisson link — log(1.0) = 0.0
    chex.assert_trees_all_close(
        initialized.readout.layer.bias, jnp.zeros(observation_dim)
    )

    conf = _poisson_conf(state_dim, observation_dim, n_steps=time_steps)
    observation = _make_observation(conf, key, likelihood="Poisson")
    initialized = observation.initialize(t, y, u, c)
    # Per-step log(mean of ones) = 0.0
    chex.assert_trees_all_close(
        initialized.readout.biases, jnp.zeros((time_steps, observation_dim))
    )


def test_diag_gaussian_initialize_biases():
    key = jrnd.key(3)
    state_dim = 2
    observation_dim = 3
    time_steps = 4
    batch = 2
    t = jnp.arange(time_steps)
    y = jnp.zeros((batch, time_steps, observation_dim))
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))

    conf = _gaussian_conf(state_dim, observation_dim, n_steps=0)
    observation = _make_observation(conf, key, likelihood="Gaussian")
    initialized = observation.initialize(t, y, u, c)
    chex.assert_trees_all_close(
        initialized.readout.layer.bias, jnp.zeros(observation_dim)
    )

    conf = _gaussian_conf(state_dim, observation_dim, n_steps=time_steps)
    observation = _make_observation(conf, key, likelihood="Gaussian")
    initialized = observation.initialize(t, y, u, c)
    chex.assert_trees_all_close(
        initialized.readout.biases, jnp.zeros((time_steps, observation_dim))
    )


# ---------------------------------------------------------------------------
# Readout initializer registry tests
# ---------------------------------------------------------------------------

def _make_synthetic_data(key, state_dim, observation_dim, batch, time_steps):
    """Generate synthetic y = C @ z + b + noise for testing FA init."""
    k1, k2, k3, k4 = jrnd.split(key, 4)
    C_true = jrnd.normal(k1, (observation_dim, state_dim))
    b_true = jrnd.normal(k2, (observation_dim,))
    z = jrnd.normal(k3, (batch, time_steps, state_dim))
    noise = 0.1 * jrnd.normal(k4, (batch, time_steps, observation_dim))
    y = jnp.einsum("ds,bts->btd", C_true, z) + b_true + noise
    return y, C_true, b_true


def test_fa_default_sets_weight_and_bias():
    """FA (default) sets weight from data covariance and bias from mean."""
    key = jrnd.key(10)
    state_dim, observation_dim = 2, 6
    batch, time_steps = 32, 50
    obs_noise_var = 0.01

    y, C_true, b_true = _make_synthetic_data(
        key, state_dim, observation_dim, batch, time_steps
    )

    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[obs_noise_var] * observation_dim,
        n_steps=0,
        norm_readout=False,
        likelihood="Gaussian",
    ))
    glm = GLM(conf, jrnd.key(0))
    weight_before = glm.readout.weight.copy()

    t = jnp.arange(time_steps)
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))
    initialized = glm.initialize(t, y, u, c)

    # Weight should have changed
    assert not jnp.allclose(initialized.readout.weight, weight_before)
    # Weight shape is correct
    chex.assert_shape(initialized.readout.weight, (observation_dim, state_dim))
    # Bias should be close to mean(y)
    mean_y = jnp.mean(y.reshape(-1, observation_dim), axis=0)
    chex.assert_trees_all_close(
        initialized.readout.layer.bias, mean_y, atol=1e-5
    )



def test_none_skips_readout_init():
    """readout_init=None leaves readout completely untouched."""
    key = jrnd.key(30)
    state_dim, observation_dim = 2, 4
    batch, time_steps = 8, 10

    y = jrnd.normal(key, (batch, time_steps, observation_dim))

    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[1.0] * observation_dim,
        n_steps=0,
        norm_readout=False,
        likelihood="Gaussian",
        readout_init=None,
    ))
    glm = GLM(conf, jrnd.key(2))
    weight_before = glm.readout.weight.copy()
    bias_before = glm.readout.layer.bias.copy()

    t = jnp.arange(time_steps)
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))
    initialized = glm.initialize(t, y, u, c)

    chex.assert_trees_all_close(initialized.readout.weight, weight_before)
    chex.assert_trees_all_close(initialized.readout.layer.bias, bias_before)


def test_set_readout_stationary():
    """GLM.set_readout sets weight and bias on StationaryLinear."""
    state_dim, observation_dim = 3, 5
    conf = _gaussian_conf(state_dim, observation_dim)
    glm = GLM(conf, jrnd.key(0))

    new_weight = jnp.ones((observation_dim, state_dim)) * 2.0
    new_bias = jnp.ones(observation_dim) * 3.0
    updated = glm.set_readout(weight=new_weight, bias=new_bias)

    chex.assert_trees_all_close(updated.readout.weight, new_weight)
    chex.assert_trees_all_close(updated.readout.layer.bias, new_bias)


def test_set_readout_variant_bias():
    """GLM.set_readout sets weight and bias on VariantBiasLinear."""
    state_dim, observation_dim = 3, 5
    n_steps = 10
    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[1.0] * observation_dim,
        n_steps=n_steps,
        norm_readout=False,
        likelihood="Gaussian",
    ))
    glm = GLM(conf, jrnd.key(0))

    new_weight = jnp.ones((observation_dim, state_dim)) * 2.0
    new_biases = jnp.ones((n_steps, observation_dim)) * 3.0
    updated = glm.set_readout(weight=new_weight, bias=new_biases)

    chex.assert_trees_all_close(updated.readout.weight, new_weight)
    chex.assert_trees_all_close(updated.readout.biases, new_biases)


def test_set_readout_variant_bias_broadcast():
    """VariantBiasLinear.set_bias broadcasts 1-D bias to all steps."""
    state_dim, observation_dim = 3, 5
    n_steps = 10
    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[1.0] * observation_dim,
        n_steps=n_steps,
        norm_readout=False,
        likelihood="Gaussian",
    ))
    glm = GLM(conf, jrnd.key(0))

    scalar_bias = jnp.ones(observation_dim) * 5.0
    updated = glm.set_readout(bias=scalar_bias)
    expected = jnp.broadcast_to(scalar_bias, (n_steps, observation_dim))
    chex.assert_trees_all_close(updated.readout.biases, expected)


def test_variant_bias_fa_init():
    """FA init with VariantBiasLinear sets weight and per-step bias."""
    key = jrnd.key(40)
    state_dim, observation_dim = 2, 6
    batch, time_steps = 16, 20

    y, _, _ = _make_synthetic_data(
        key, state_dim, observation_dim, batch, time_steps
    )

    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[0.01] * observation_dim,
        n_steps=time_steps,
        norm_readout=False,
        likelihood="Gaussian",
    ))
    glm = GLM(conf, jrnd.key(0))
    weight_before = glm.readout.weight.copy()

    t = jnp.arange(time_steps)
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))
    initialized = glm.initialize(t, y, u, c)

    # Weight changed
    assert not jnp.allclose(initialized.readout.weight, weight_before)
    chex.assert_shape(initialized.readout.weight, (observation_dim, state_dim))
    # Per-step biases have correct shape
    chex.assert_shape(initialized.readout.biases, (time_steps, observation_dim))
    # Per-step bias should be mean over batch
    expected_bias = jnp.mean(y, axis=0)  # (T, obs_dim)
    chex.assert_trees_all_close(
        initialized.readout.biases, expected_bias, atol=1e-5
    )


def test_poisson_fa_init():
    """FA init with Poisson: obs_noise_var defaults to 0, bias is log(mean)."""
    key = jrnd.key(50)
    state_dim, observation_dim = 2, 4
    batch, time_steps = 16, 20

    # Positive count-like data
    y = jrnd.poisson(key, lam=5.0, shape=(batch, time_steps, observation_dim)).astype(
        jnp.float32
    )

    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        n_steps=0,
        norm_readout=False,
        likelihood="Poisson",
    ))
    glm = GLM(conf, jrnd.key(0))

    t = jnp.arange(time_steps)
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))
    initialized = glm.initialize(t, y, u, c)

    # Weight has correct shape
    chex.assert_shape(initialized.readout.weight, (observation_dim, state_dim))
    # Bias is log(mean) via Poisson link
    mean_y = jnp.mean(y.reshape(-1, observation_dim), axis=0)
    expected_bias = jnp.log(jnp.maximum(mean_y, 1e-7))
    chex.assert_trees_all_close(
        initialized.readout.layer.bias, expected_bias, atol=1e-5
    )


def test_custom_readout_initializer():
    """Third-party initializers registered via register_readout_init work."""
    from jaxfads.observations import _READOUT_INIT

    @register_readout_init("_test_custom")
    def _custom_init(y, conf):
        obs_dim = y.shape[-1]
        state_dim = conf.state_dim
        C = jnp.eye(obs_dim, state_dim) * 42.0
        b_raw = jnp.zeros(obs_dim) + 7.0
        return C, b_raw

    try:
        state_dim, observation_dim = 2, 4
        batch, time_steps = 4, 5
        y = jrnd.normal(jrnd.key(0), (batch, time_steps, observation_dim))

        conf = OmegaConf.create(dict(
            model="GLM",
            state_dim=state_dim,
            observation_dim=observation_dim,
            cov=[1.0] * observation_dim,
            n_steps=0,
            norm_readout=False,
            likelihood="Gaussian",
            readout_init="_test_custom",
        ))
        glm = GLM(conf, jrnd.key(1))

        t = jnp.arange(time_steps)
        u = jnp.zeros((batch, time_steps, 1))
        c = jnp.zeros((batch, time_steps, 1))
        initialized = glm.initialize(t, y, u, c)

        expected_weight = jnp.eye(observation_dim, state_dim) * 42.0
        expected_bias = jnp.zeros(observation_dim) + 7.0
        chex.assert_trees_all_close(initialized.readout.weight, expected_weight)
        chex.assert_trees_all_close(initialized.readout.layer.bias, expected_bias)
    finally:
        _READOUT_INIT.pop("_test_custom", None)


def test_fa_init_with_explicit_obs_noise_var():
    """readout_init_conf.obs_noise_var overrides conf.cov for FA."""
    key = jrnd.key(60)
    state_dim, observation_dim = 2, 6
    batch, time_steps = 32, 50

    y, _, _ = _make_synthetic_data(
        key, state_dim, observation_dim, batch, time_steps
    )

    # conf.cov = 999 (very wrong), but readout_init_conf overrides
    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[999.0] * observation_dim,
        n_steps=0,
        norm_readout=False,
        likelihood="Gaussian",
        readout_init_conf=dict(obs_noise_var=0.01),
    ))
    glm = GLM(conf, jrnd.key(0))

    t = jnp.arange(time_steps)
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))
    initialized = glm.initialize(t, y, u, c)

    # Weight should be finite and non-zero (not blown up by cov=999)
    chex.assert_tree_all_finite(initialized.readout.weight)
    assert jnp.any(initialized.readout.weight != 0)


def test_unknown_readout_init_raises():
    """Unknown readout_init raises ValueError."""
    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=2,
        observation_dim=4,
        cov=[1.0] * 4,
        n_steps=0,
        norm_readout=False,
        likelihood="Gaussian",
        readout_init="nonexistent",
    ))
    glm = GLM(conf, jrnd.key(0))

    t = jnp.arange(5)
    y = jnp.zeros((2, 5, 4))
    u = jnp.zeros((2, 5, 1))
    c = jnp.zeros((2, 5, 1))

    try:
        glm.initialize(t, y, u, c)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_norm_readout_set_weight_and_bias():
    """set_weight/set_bias work through NormalizedLinear wrapper."""
    state_dim, observation_dim = 3, 5
    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[1.0] * observation_dim,
        n_steps=0,
        norm_readout=True,
        likelihood="Gaussian",
    ))
    glm = GLM(conf, jrnd.key(0))

    new_weight = jnp.ones((observation_dim, state_dim)) * 2.0
    new_bias = jnp.ones(observation_dim) * 3.0
    updated = glm.set_readout(weight=new_weight, bias=new_bias)

    # Weight is normalized (direction preserved, unit norm)
    expected_dir = new_weight / jnp.linalg.norm(new_weight)
    chex.assert_trees_all_close(updated.readout.weight, expected_dir, atol=1e-5)
    # Bias is set exactly
    chex.assert_trees_all_close(updated.readout.layer.bias, new_bias)


def test_norm_readout_fa_init():
    """FA init sets weight and bias through NormalizedLinear wrapper."""
    key = jrnd.key(70)
    state_dim, observation_dim = 2, 6
    batch, time_steps = 32, 50

    y, _, _ = _make_synthetic_data(
        key, state_dim, observation_dim, batch, time_steps
    )

    conf = OmegaConf.create(dict(
        model="GLM",
        state_dim=state_dim,
        observation_dim=observation_dim,
        cov=[0.01] * observation_dim,
        n_steps=0,
        norm_readout=True,
        likelihood="Gaussian",
    ))
    glm = GLM(conf, jrnd.key(0))
    weight_before = glm.readout.weight.copy()

    t = jnp.arange(time_steps)
    u = jnp.zeros((batch, time_steps, 1))
    c = jnp.zeros((batch, time_steps, 1))
    initialized = glm.initialize(t, y, u, c)

    # Weight changed and is unit-norm
    assert not jnp.allclose(initialized.readout.weight, weight_before)
    chex.assert_shape(initialized.readout.weight, (observation_dim, state_dim))
    norm = jnp.linalg.norm(initialized.readout.weight)
    chex.assert_trees_all_close(norm, 1.0, atol=1e-5)
    # Bias is mean(y)
    mean_y = jnp.mean(y.reshape(-1, observation_dim), axis=0)
    chex.assert_trees_all_close(
        initialized.readout.layer.bias, mean_y, atol=1e-5
    )
