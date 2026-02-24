import pytest
from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import MVN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mean(mvn: MVN, key) -> jnp.ndarray:
    """Build a valid structured mean for any rank."""
    d, r = mvn._dim, mvn._rank
    k1, k2, k3 = jrnd.split(key, 3)
    loc = jrnd.normal(k1, (d,))
    cov_diag = jnp.abs(jrnd.normal(k2, (d,))) + 0.1
    cov_factor = jrnd.normal(k3, (d, r)) * 0.5 if r > 0 else jnp.zeros((d, 0))
    return mvn._pack_mean(loc, cov_diag, cov_factor)


# ---------------------------------------------------------------------------
# natural ↔ mean roundtrip (unified across ranks)
# ---------------------------------------------------------------------------

_ROUNDTRIP_CASES = [
    pytest.param(4, 0, id="diag"),
    pytest.param(4, 1, id="lowrank-1"),
    pytest.param(4, 2, id="lowrank-2"),
    pytest.param(3, 3, id="full"),
]


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_natural_mean_roundtrip(dim, rank):
    """mean_to_natural → natural_to_mean preserves covariance structure."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(42))

    natural = mvn.mean_to_natural(mp)
    chex.assert_tree_all_finite(natural)
    chex.assert_shape(natural, (mvn.param_size(dim),))

    mp_rt = mvn.natural_to_mean(natural)
    chex.assert_tree_all_finite(mp_rt)
    chex.assert_shape(mp_rt, (mvn.mean_size(dim),))

    _, cov_orig = mvn.mean_to_canon(mp)
    _, cov_rt = mvn.mean_to_canon(mp_rt)
    cov_orig_full = mvn.full_cov(cov_orig)
    cov_rt_full = mvn.full_cov(cov_rt)
    chex.assert_trees_all_close(jnp.diag(cov_orig_full), jnp.diag(cov_rt_full), atol=1e-4)
    eigenvalues = jnp.linalg.eigvalsh(cov_rt_full)
    assert jnp.all(eigenvalues > 0), f"Non-PD roundtrip cov: {eigenvalues}"


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_natural_mean_roundtrip_from_prior(dim, rank):
    """natural_to_mean on prior_natural recovers N(0, I)."""
    mvn = MVN(dim=dim, rank=rank)
    natural = mvn.prior_natural(dim)

    mp = mvn.natural_to_mean(natural)
    chex.assert_tree_all_finite(mp)

    loc, cov = mvn.mean_to_canon(mp)
    chex.assert_trees_all_close(loc, jnp.zeros(dim), atol=1e-5)
    chex.assert_trees_all_close(mvn.full_cov(cov), jnp.eye(dim), atol=1e-4)

    natural_rt = mvn.mean_to_natural(mp)
    chex.assert_trees_all_close(natural, natural_rt, atol=1e-4)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_near_zero_cov_stability(diag):
    """Near-zero covariance must not produce extreme natural parameters."""
    state_dim = 2
    mean = jnp.ones(state_dim)

    eps = jnp.finfo(jnp.float32).eps
    tiny_cov = jnp.full(state_dim, eps)

    mp = diag.canon_to_mean(mean, tiny_cov)
    natural = diag.mean_to_natural(mp)
    chex.assert_tree_all_finite(natural)

    _, nat2 = jnp.split(natural, 2)
    assert float(jnp.max(jnp.abs(nat2)).item()) < 1e7

    mp_rt = diag.natural_to_mean(natural)
    chex.assert_tree_all_finite(mp_rt)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def test_kl_matches_closed_form(diag):
    """MVN(rank=0).kl() must match the closed-form diagonal Gaussian KL."""

    def _kl_closed_form(m1, v1, m2, v2):
        return 0.5 * jnp.sum(jnp.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    m1 = jnp.array([1.0, -0.5, 0.3])
    v1 = jnp.array([0.5, 2.0, 0.1])
    m2 = jnp.array([0.0, 0.0, 1.0])
    v2 = jnp.array([1.0, 1.0, 0.5])

    diag3 = MVN(dim=3, rank=0)
    mp1 = diag3.canon_to_mean(m1, v1)
    mp2 = diag3.canon_to_mean(m2, v2)

    kl_actual = diag3.kl(mp1, mp2)
    kl_expected = _kl_closed_form(m1, v1, m2, v2)
    chex.assert_tree_all_finite(kl_actual)
    chex.assert_trees_all_close(kl_actual, kl_expected, atol=1e-5)


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_kl_self_zero(dim, rank):
    """KL(q, q) = 0 for any rank."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(5))
    kl = mvn.kl(mp, mp)
    chex.assert_trees_all_close(kl, jnp.array(0.0), atol=1e-4)


def test_reparameterization(spec):
    state_dim = spec["state_dim"]
    m1 = jnp.ones(state_dim)
    cov1 = jnp.eye(state_dim)
    assert (
        tfp.MultivariateNormalFullCovariance(m1, cov1).reparameterization_type
        == tfp.FULLY_REPARAMETERIZED
    )


# ---------------------------------------------------------------------------
# constrain / unconstrain roundtrips (unified)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_constrain_mean_roundtrip(dim, rank):
    """constrain_mean(unconstrain_mean(m)) ≈ m."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(10))

    unconstrained = mvn.unconstrain_mean(mp)
    recovered = mvn.constrain_mean(unconstrained)
    chex.assert_trees_all_close(recovered, mp, atol=1e-5)


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_constrain_mean_produces_valid_cov(dim, rank):
    """constrain_mean produces PD covariance."""
    mvn = MVN(dim=dim, rank=rank)
    unc_size = mvn.mean_size(dim)
    unconstrained = jrnd.normal(jrnd.key(0), (unc_size,))

    mp = mvn.constrain_mean(unconstrained)
    chex.assert_shape(mp, (unc_size,))
    chex.assert_tree_all_finite(mp)

    loc, cov = mvn.mean_to_canon(mp)
    cov_full = mvn.full_cov(cov)
    chex.assert_trees_all_close(cov_full, cov_full.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov_full)
    assert jnp.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"


def test_constrain_natural_neg_def():
    """constrain_natural produces negative-definite η₂ (rank > 0)."""
    state_dim = 3
    full = MVN(dim=state_dim, rank=state_dim)
    param_sz = full.param_size(state_dim)
    unconstrained = jrnd.normal(jrnd.key(1), (param_sz,))

    natural = full.constrain_natural(unconstrained)
    chex.assert_shape(natural, (param_sz,))
    chex.assert_tree_all_finite(natural)

    _, nat2_flat = jnp.split(natural, [state_dim])
    nat2 = jnp.reshape(nat2_flat, (state_dim, state_dim))
    eigenvalues = jnp.linalg.eigvalsh(nat2)
    assert jnp.all(eigenvalues < 0), f"nat2 not neg-def: eigenvalues={eigenvalues}"


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_unconstrain_natural_roundtrip(dim, rank):
    """unconstrain_natural inverts constrain_natural."""
    mvn = MVN(dim=dim, rank=rank)
    natural = mvn.prior_natural(dim)
    unconstrained = mvn.unconstrain_natural(natural)
    chex.assert_tree_all_finite(unconstrained)

    natural_rt = mvn.constrain_natural(unconstrained)
    chex.assert_tree_all_finite(natural_rt)

    mp = mvn.natural_to_mean(natural)
    mp_rt = mvn.natural_to_mean(natural_rt)
    chex.assert_trees_all_close(mp, mp_rt, atol=1e-4)


# ---------------------------------------------------------------------------
# init_noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_init_noise(dim, rank):
    """init_noise → constrain_mean gives isotropic N(0, scale·I)."""
    mvn = MVN(dim=dim, rank=rank)
    scale = 2.5

    unc = mvn.init_noise(scale, dim)
    mp = mvn.constrain_mean(unc)

    loc, cov = mvn.mean_to_canon(mp)
    cov_full = mvn.full_cov(cov)
    chex.assert_trees_all_close(loc, jnp.zeros(dim), atol=1e-6)
    chex.assert_trees_all_close(cov_full, jnp.eye(dim) * scale, atol=1e-5)


# ---------------------------------------------------------------------------
# param_size / mean_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank, expected_param, expected_mean", [
    pytest.param(3, 0, 6, 6, id="diag"),
    pytest.param(3, 3, 12, 15, id="full"),
    pytest.param(4, 1, 20, 12, id="lowrank-1"),
    pytest.param(4, 2, 20, 16, id="lowrank-2"),
])
def test_param_and_mean_sizes(dim, rank, expected_param, expected_mean):
    mvn = MVN(dim=dim, rank=rank)
    assert mvn.param_size(dim) == expected_param
    assert mvn.mean_size(dim) == expected_mean


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_sample_shape(dim, rank):
    """Sampling produces correct shapes and finite values."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(0))

    samples = mvn.sample_by_mean(jrnd.key(1), mp, 50)
    chex.assert_shape(samples, (50, dim))
    chex.assert_tree_all_finite(samples)


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
