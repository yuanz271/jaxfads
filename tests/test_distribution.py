from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import MVN


# ---------------------------------------------------------------------------
# Diagonal MVN (rank=0)
# ---------------------------------------------------------------------------


def test_diagmvn(diag, spec):
    state_dim = spec["state_dim"]

    m1 = jnp.ones(state_dim)
    cov1 = jnp.ones(state_dim)
    m2 = jnp.zeros(state_dim)
    cov2 = jnp.ones(state_dim) * 2

    moment = diag.canon_to_moment(m1, cov1)
    nat1 = diag.moment_to_natural(moment)
    moment1 = diag.natural_to_moment(nat1)

    chex.assert_trees_all_close(moment, moment1)

    moment2 = diag.canon_to_moment(m2, cov2)
    kl = diag.kl(moment1, moment2)
    chex.assert_tree_all_finite(kl)


def test_reparameterization(spec):
    state_dim = spec["state_dim"]
    m1 = jnp.ones(state_dim)
    cov1 = jnp.eye(state_dim)
    assert (
        tfp.MultivariateNormalFullCovariance(m1, cov1).reparameterization_type
        == tfp.FULLY_REPARAMETERIZED
    )


def test_diagmvn_near_zero_cov_stability(diag):
    """Near-zero covariance must not produce extreme natural parameters."""
    state_dim = 2
    mean = jnp.ones(state_dim)

    eps = jnp.finfo(jnp.float32).eps
    tiny_cov = jnp.full(state_dim, eps)

    moment = diag.canon_to_moment(mean, tiny_cov)
    natural = diag.moment_to_natural(moment)
    chex.assert_tree_all_finite(natural)

    _, nat2 = jnp.split(natural, 2)
    assert float(jnp.max(jnp.abs(nat2)).item()) < 1e7

    moment_rt = diag.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment_rt)

    normal_cov = jnp.ones(state_dim)
    moment_normal = diag.canon_to_moment(mean, normal_cov)
    natural_normal = diag.moment_to_natural(moment_normal)
    moment_normal_rt = diag.natural_to_moment(natural_normal)
    chex.assert_trees_all_close(moment_normal, moment_normal_rt)


def test_diagmvn_kl_matches_closed_form(diag):
    """MVN(rank=0).kl() must match the closed-form diagonal Gaussian KL."""

    def _kl_closed_form(m1, v1, m2, v2):
        return 0.5 * jnp.sum(jnp.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    m1 = jnp.array([1.0, -0.5, 0.3])
    v1 = jnp.array([0.5, 2.0, 0.1])
    m2 = jnp.array([0.0, 0.0, 1.0])
    v2 = jnp.array([1.0, 1.0, 0.5])

    # Need dim=3 for this test since diag fixture has dim=2
    diag3 = MVN(dim=3, rank=0)
    moment1 = diag3.canon_to_moment(m1, v1)
    moment2 = diag3.canon_to_moment(m2, v2)

    kl_actual = diag3.kl(moment1, moment2)
    kl_expected = _kl_closed_form(m1, v1, m2, v2)
    chex.assert_tree_all_finite(kl_actual)
    chex.assert_trees_all_close(kl_actual, kl_expected, atol=1e-5)

    m1b = jnp.array([3.0, -2.0])
    v1b = jnp.array([0.1, 10.0])
    m2b = jnp.array([0.0, 1.0])
    v2b = jnp.array([5.0, 0.3])

    moment1b = diag.canon_to_moment(m1b, v1b)
    moment2b = diag.canon_to_moment(m2b, v2b)

    kl_actual_b = diag.kl(moment1b, moment2b)
    kl_expected_b = _kl_closed_form(m1b, v1b, m2b, v2b)
    chex.assert_tree_all_finite(kl_actual_b)
    chex.assert_trees_all_close(kl_actual_b, kl_expected_b, atol=1e-4)

    kl_self = diag.kl(moment1b, moment1b)
    chex.assert_trees_all_close(kl_self, jnp.array(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Full-rank MVN (rank=D)
# ---------------------------------------------------------------------------


def test_fullrank_constrain_moment():
    """constrain_moment produces valid structured moment."""
    state_dim = 3
    full = MVN(dim=state_dim, rank=state_dim)
    unc_size = state_dim * (2 + state_dim)
    unconstrained = jrnd.normal(jrnd.key(0), (unc_size,))

    moment = full.constrain_moment(unconstrained)
    chex.assert_shape(moment, (unc_size,))
    chex.assert_tree_all_finite(moment)

    mean, cov = full.moment_to_canon(moment)
    chex.assert_shape(mean, (state_dim,))
    chex.assert_shape(cov, (state_dim, state_dim))
    chex.assert_trees_all_close(cov, cov.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"


def test_fullrank_constrain_natural():
    """constrain_natural produces valid natural params."""
    state_dim = 3
    full = MVN(dim=state_dim, rank=state_dim)
    param_sz = full.param_size(state_dim)
    unconstrained = jrnd.normal(jrnd.key(1), (param_sz,))

    natural = full.constrain_natural(unconstrained)
    chex.assert_shape(natural, (param_sz,))
    chex.assert_tree_all_finite(natural)

    nat1, nat2_flat = jnp.split(natural, [state_dim])
    nat2 = jnp.reshape(nat2_flat, (state_dim, state_dim))
    eigenvalues = jnp.linalg.eigvalsh(nat2)
    assert jnp.all(eigenvalues < 0), f"nat2 not neg-def: eigenvalues={eigenvalues}"

    moment = full.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment)


def test_fullrank_unconstrain_natural_roundtrip():
    """unconstrain_natural inverts constrain_natural."""
    state_dim = 3
    full = MVN(dim=state_dim, rank=state_dim)
    natural = full.prior_natural(state_dim)
    unconstrained = full.unconstrain_natural(natural)
    chex.assert_tree_all_finite(unconstrained)

    natural_rt = full.constrain_natural(unconstrained)
    chex.assert_tree_all_finite(natural_rt)

    moment = full.natural_to_moment(natural)
    moment_rt = full.natural_to_moment(natural_rt)
    chex.assert_trees_all_close(moment, moment_rt, atol=1e-4)


def test_fullrank_natural_moment_roundtrip():
    """natural ↔ moment roundtrip via structured format."""
    d = 3
    full = MVN(dim=d, rank=d)
    natural = full.prior_natural(d)
    moment = full.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment)

    mean, cov = full.moment_to_canon(moment)
    chex.assert_trees_all_close(mean, jnp.zeros(d), atol=1e-5)
    chex.assert_trees_all_close(cov, jnp.eye(d), atol=1e-4)

    natural_rt = full.moment_to_natural(moment)
    chex.assert_trees_all_close(natural, natural_rt, atol=1e-4)


def test_lowrankcov():
    loc = jnp.ones(2)
    cov_diag = jnp.ones(2)
    cov_lr = jnp.ones((2, 1))
    _ = tfp.MultivariateNormalDiagPlusLowRankCovariance(loc, cov_diag, cov_lr)


# ---------------------------------------------------------------------------
# Unconstrain / constrain roundtrip
# ---------------------------------------------------------------------------


def test_diagmvn_unconstrain_moment_roundtrip():
    """constrain_moment(unconstrain_moment(m)) ≈ m for rank=0."""
    state_dim = 4
    diag4 = MVN(dim=state_dim, rank=0)
    mean = jrnd.normal(jrnd.key(10), (state_dim,))
    cov = jnp.abs(jrnd.normal(jrnd.key(11), (state_dim,))) + 0.1
    moment = diag4.canon_to_moment(mean, cov)

    unconstrained = diag4.unconstrain_moment(moment)
    recovered = diag4.constrain_moment(unconstrained)
    chex.assert_trees_all_close(recovered, moment, atol=1e-5)


def test_fullrank_unconstrain_moment_roundtrip():
    """constrain_moment(unconstrain_moment(m)) ≈ m for rank=D."""
    state_dim = 3
    full = MVN(dim=state_dim, rank=state_dim)
    mean = jrnd.normal(jrnd.key(20), (state_dim,))
    cov_diag = jnp.full(state_dim, 2.0)
    cov_factor = jnp.zeros((state_dim, state_dim))
    moment = jnp.concatenate((mean, cov_diag, cov_factor.flatten()))

    unconstrained = full.unconstrain_moment(moment)
    recovered = full.constrain_moment(unconstrained)
    chex.assert_trees_all_close(recovered, moment, atol=1e-4)


def test_init_noise_produces_expected_moment(diag):
    """init_noise → constrain_moment gives correct noise distribution."""
    state_dim = 2
    cov_init = 2.0

    unconstrained = diag.init_noise(cov_init, state_dim)
    moment = diag.constrain_moment(unconstrained)
    mean, var = diag.moment_to_canon(moment)

    chex.assert_trees_all_close(mean, jnp.zeros(state_dim), atol=1e-6)
    chex.assert_trees_all_close(var, jnp.full(state_dim, cov_init), atol=1e-5)


# ---------------------------------------------------------------------------
# param_size / moment_size
# ---------------------------------------------------------------------------


def test_param_size():
    diag3 = MVN(dim=3, rank=0)
    full3 = MVN(dim=3, rank=3)
    assert diag3.param_size(3) == 6
    assert full3.param_size(3) == 12
    assert diag3.moment_size(3) == 6


def test_moment_size_full():
    full3 = MVN(dim=3, rank=3)
    assert full3.moment_size(3) == 15


# ---------------------------------------------------------------------------
# Low-rank MVN (rank=1, rank=2)
# ---------------------------------------------------------------------------


def test_lowrank_param_and_moment_sizes():
    d = 4
    lr1 = MVN(dim=d, rank=1)
    lr2 = MVN(dim=d, rank=2)
    assert lr1.param_size(d) == d + d * d
    assert lr1.moment_size(d) == d * (2 + 1)
    assert lr2.moment_size(d) == d * (2 + 2)


def test_lowrank_natural_moment_roundtrip():
    """moment_to_natural → natural_to_moment preserves structure for low-rank MVN."""
    d = 4
    lr1 = MVN(dim=d, rank=1)
    key = jrnd.key(42)

    loc = jrnd.normal(key, (d,))
    cov_diag = jnp.ones(d)
    cov_factor = jrnd.normal(key, (d, 1)) * 0.5
    moment = jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    natural = lr1.moment_to_natural(moment)
    chex.assert_tree_all_finite(natural)
    chex.assert_shape(natural, (lr1.param_size(d),))

    moment_rt = lr1.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment_rt)
    chex.assert_shape(moment_rt, (lr1.moment_size(d),))

    _, cov_orig = lr1.moment_to_canon(moment)
    _, cov_rt = lr1.moment_to_canon(moment_rt)
    chex.assert_trees_all_close(jnp.diag(cov_orig), jnp.diag(cov_rt), atol=1e-4)
    eigenvalues = jnp.linalg.eigvalsh(cov_rt)
    assert jnp.all(eigenvalues > 0), f"Non-PD roundtrip cov: {eigenvalues}"


def test_lowrank_sample():
    """Sampling from low-rank MVN produces correct shapes."""
    d = 3
    lr2 = MVN(dim=d, rank=2)
    cov_factor = jrnd.normal(jrnd.key(0), (d, 2)) * 0.3
    moment = jnp.concatenate((jnp.zeros(d), jnp.ones(d), cov_factor.flatten()))

    samples = lr2.sample_by_moment(jrnd.key(1), moment, 50)
    chex.assert_shape(samples, (50, d))
    chex.assert_tree_all_finite(samples)


def test_lowrank_kl_self_zero():
    """KL(q, q) = 0 for low-rank MVN."""
    d = 3
    lr1 = MVN(dim=d, rank=1)
    cov_factor = jrnd.normal(jrnd.key(5), (d, 1)) * 0.5
    moment = jnp.concatenate(
        (jnp.array([1.0, -0.5, 0.3]), jnp.array([0.5, 1.0, 0.8]), cov_factor.flatten())
    )

    kl = lr1.kl(moment, moment)
    chex.assert_trees_all_close(kl, jnp.array(0.0), atol=1e-4)


def test_lowrank_constrain_moment_roundtrip():
    """constrain(unconstrain(moment)) ≈ moment for low-rank MVN."""
    d = 4
    lr2 = MVN(dim=d, rank=2)
    loc = jrnd.normal(jrnd.key(10), (d,))
    cov_diag = jnp.abs(jrnd.normal(jrnd.key(11), (d,))) + 0.1
    cov_factor = jrnd.normal(jrnd.key(12), (d, 2)) * 0.3
    moment = jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    unc = lr2.unconstrain_moment(moment)
    recovered = lr2.constrain_moment(unc)
    chex.assert_trees_all_close(recovered, moment, atol=1e-5)


def test_lowrank_init_noise():
    """init_noise produces isotropic noise moment for low-rank MVN."""
    d = 3
    lr1 = MVN(dim=d, rank=1)
    scale = 2.5
    unc = lr1.init_noise(scale, d)
    moment = lr1.constrain_moment(unc)

    loc, cov_diag, cov_factor = lr1._split_moment(moment)
    chex.assert_trees_all_close(loc, jnp.zeros(d), atol=1e-6)
    chex.assert_trees_all_close(cov_diag, jnp.full(d, scale), atol=1e-5)
    chex.assert_trees_all_close(cov_factor, jnp.zeros((d, 1)), atol=1e-6)


def test_rank_validation():
    """MVN must reject invalid rank values."""
    import pytest

    with pytest.raises(ValueError, match="rank must satisfy"):
        MVN(dim=3, rank=-1)
    with pytest.raises(ValueError, match="rank must satisfy"):
        MVN(dim=3, rank=4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lookup():
    """SubclassRegistryMixin finds MVN by name."""
    from jaxfads.base import Approx

    assert Approx.get_subclass("MVN") is MVN
