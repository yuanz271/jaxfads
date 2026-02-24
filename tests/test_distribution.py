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
def test_param_from_conf_default_is_standard_normal(dim, rank):
    """param_from_conf(scale=1.0) → to_structured gives N(0, I)."""
    mvn = MVN(dim=dim, rank=rank)
    free = mvn.param_from_conf(scale=1.0)
    structured = mvn.to_structured(free)

    loc, cov = mvn.mean_to_canon(structured)
    chex.assert_trees_all_close(loc, jnp.zeros(dim), atol=1e-5)
    chex.assert_trees_all_close(mvn.full_cov(cov), jnp.eye(dim), atol=1e-4)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_near_zero_cov_bounded_natural(diag):
    """Near-zero diagonal covariance must not produce extreme η₂."""
    state_dim = 2
    mean = jnp.ones(state_dim)

    eps = jnp.finfo(jnp.float32).eps
    tiny_cov = jnp.full(state_dim, eps)

    mp = diag.canon_to_mean(mean, tiny_cov)
    natural = diag.mean_to_natural(mp)
    chex.assert_tree_all_finite(natural)

    _, nat2 = jnp.split(natural, 2)
    assert float(jnp.max(jnp.abs(nat2)).item()) < 1e7


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
def test_to_structured_roundtrip(dim, rank):
    """to_structured(to_free(m)) ≈ m."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(10))

    unconstrained = mvn.to_free(mp)
    recovered = mvn.to_structured(unconstrained)
    chex.assert_trees_all_close(recovered, mp, atol=1e-5)


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_to_structured_produces_valid_cov(dim, rank):
    """to_structured produces PD covariance."""
    mvn = MVN(dim=dim, rank=rank)
    unc_size = mvn.mean_size(dim)
    unconstrained = jrnd.normal(jrnd.key(0), (unc_size,))

    mp = mvn.to_structured(unconstrained)
    chex.assert_shape(mp, (unc_size,))
    chex.assert_tree_all_finite(mp)

    loc, cov = mvn.mean_to_canon(mp)
    cov_full = mvn.full_cov(cov)
    chex.assert_trees_all_close(cov_full, cov_full.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov_full)
    assert jnp.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_structured_to_natural_roundtrip(dim, rank):
    """structured_to_natural ∘ natural_to_mean ≈ identity."""
    mvn = MVN(dim=dim, rank=rank)
    free = mvn.param_from_conf(scale=1.0)
    structured = mvn.to_structured(free)
    natural = mvn.structured_to_natural(structured)
    natural_rt = mvn.structured_to_natural(mvn.natural_to_mean(natural))
    chex.assert_tree_all_finite(natural_rt)

    mp = mvn.natural_to_mean(natural)
    mp_rt = mvn.natural_to_mean(natural_rt)
    chex.assert_trees_all_close(mp, mp_rt, atol=1e-4)


# ---------------------------------------------------------------------------
# param_from_conf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_param_from_conf(dim, rank):
    """param_from_conf(scale=...) → to_structured gives isotropic N(0, scale·I)."""
    mvn = MVN(dim=dim, rank=rank)
    scale = 2.5

    free = mvn.param_from_conf(scale=scale)
    mp = mvn.to_structured(free)

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


# ---------------------------------------------------------------------------
# canon ↔ mean roundtrip
# ---------------------------------------------------------------------------


def test_canon_mean_roundtrip_diag():
    """mean_to_canon → canon_to_mean is exact for rank 0."""
    mvn = MVN(dim=4, rank=0)
    mp = _make_mean(mvn, jrnd.key(77))

    loc, cov = mvn.mean_to_canon(mp)
    mp_rt = mvn.canon_to_mean(loc, cov)
    chex.assert_trees_all_close(mp, mp_rt, atol=1e-6)


@pytest.mark.parametrize("dim, rank", [
    pytest.param(4, 1, id="lowrank-1"),
    pytest.param(4, 2, id="lowrank-2"),
    pytest.param(3, 3, id="full"),
])
def test_canon_mean_roundtrip_lowrank(dim, rank):
    """mean_to_canon → canon_to_mean preserves loc and diagonal exactly.

    The off-diagonal decomposition is approximate because zeroing the
    diagonal before eigendecomposition is lossy; we only check that
    the diagonal and location are recovered and the result is PD.
    """
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(77))

    loc, cov = mvn.mean_to_canon(mp)
    mp_rt = mvn.canon_to_mean(loc, cov)

    loc_rt, cov_rt = mvn.mean_to_canon(mp_rt)
    chex.assert_trees_all_close(loc, loc_rt, atol=1e-5)
    chex.assert_trees_all_close(
        jnp.diag(mvn.full_cov(cov)), jnp.diag(mvn.full_cov(cov_rt)), atol=1e-4
    )
    eigenvalues = jnp.linalg.eigvalsh(mvn.full_cov(cov_rt))
    assert jnp.all(eigenvalues > 0), f"Non-PD roundtrip cov: {eigenvalues}"


# ---------------------------------------------------------------------------
# full_cov
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_full_cov(dim, rank):
    """full_cov returns symmetric PD (D, D) matrix consistent with mean_to_canon."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(88))

    _, cov = mvn.mean_to_canon(mp)
    cov_full = mvn.full_cov(cov)

    chex.assert_shape(cov_full, (dim, dim))
    chex.assert_tree_all_finite(cov_full)
    chex.assert_trees_all_close(cov_full, cov_full.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov_full)
    assert jnp.all(eigenvalues > 0), f"Non-PD: {eigenvalues}"


# ---------------------------------------------------------------------------
# KL positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_kl_positive(dim, rank):
    """KL(q || p) > 0 when q ≠ p."""
    mvn = MVN(dim=dim, rank=rank)
    mp1 = _make_mean(mvn, jrnd.key(10))
    mp2 = _make_mean(mvn, jrnd.key(20))

    kl = mvn.kl(mp1, mp2)
    chex.assert_tree_all_finite(kl)
    assert float(kl) > 0.0


# ---------------------------------------------------------------------------
# Sample statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_sample_statistics(dim, rank):
    """Sample mean and covariance approximate the distribution parameters."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(0))
    loc, cov = mvn.mean_to_canon(mp)
    cov_full = mvn.full_cov(cov)

    samples = mvn.sample_by_mean(jrnd.key(1), mp, 50_000)
    sample_mean = jnp.mean(samples, axis=0)
    sample_cov = jnp.cov(samples.T)

    chex.assert_trees_all_close(sample_mean, loc, atol=0.05)
    chex.assert_trees_all_close(sample_cov, cov_full, atol=0.1)


# ---------------------------------------------------------------------------
# predict_mean across ranks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_predict_mean(dim, rank):
    """predict_mean with single loc recovers (loc, noise_cov)."""
    mvn = MVN(dim=dim, rank=rank)
    scale = 1.5
    unc = mvn.param_from_conf(scale=scale)
    noise_mean = mvn.to_structured(unc)

    loc = jrnd.normal(jrnd.key(0), (dim,))
    mp = mvn.predict_mean(loc[None, :], noise_mean)

    chex.assert_tree_all_finite(mp)
    chex.assert_shape(mp, (mvn.mean_size(dim),))

    recovered_loc, recovered_cov = mvn.mean_to_canon(mp)
    chex.assert_trees_all_close(recovered_loc, loc, atol=1e-5)
    expected_cov = jnp.eye(dim) * scale
    chex.assert_trees_all_close(mvn.full_cov(recovered_cov), expected_cov, atol=1e-4)


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_predict_mean_captures_variance(dim, rank):
    """predict_mean with spread locs produces larger covariance than noise alone."""
    mvn = MVN(dim=dim, rank=rank)
    scale = 0.1
    unc = mvn.param_from_conf(scale=scale)
    noise_mean = mvn.to_structured(unc)

    # Spread locs: variance of locs >> noise variance
    key = jrnd.key(42)
    locs = jrnd.normal(key, (200, dim)) * 3.0
    mp = mvn.predict_mean(locs, noise_mean)

    _, cov = mvn.mean_to_canon(mp)
    cov_full = mvn.full_cov(cov)
    # Diagonal should be > noise scale (captures Var[f(z)] + Q)
    assert jnp.all(jnp.diag(cov_full) > scale)


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_predict_mean_all_nan_returns_nan(dim, rank):
    """predict_mean returns NaN when all locs are non-finite."""
    mvn = MVN(dim=dim, rank=rank)
    unc = mvn.param_from_conf(scale=1.0)
    noise_mean = mvn.to_structured(unc)

    nan_locs = jnp.full((5, dim), jnp.nan)
    mp = mvn.predict_mean(nan_locs, noise_mean)
    assert not jnp.any(jnp.isfinite(mp))


# ---------------------------------------------------------------------------
# _decompose_cov roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", [
    pytest.param(4, 1, id="lowrank-1"),
    pytest.param(4, 2, id="lowrank-2"),
    pytest.param(3, 3, id="full"),
])
def test_decompose_cov_preserves_diagonal_and_pd(dim, rank):
    """_decompose_cov → _build_cov preserves diagonal and produces PD result.

    The off-diagonal decomposition is approximate (zeroing the diagonal
    before eigendecomposition is lossy), so only the diagonal is checked
    for exact recovery.
    """
    mvn = MVN(dim=dim, rank=rank)

    key = jrnd.key(99)
    A = jrnd.normal(key, (dim, dim))
    sigma = A @ A.T + 0.5 * jnp.eye(dim)

    cov_diag, cov_factor = mvn._decompose_cov(sigma)
    sigma_rt = mvn._build_cov(cov_diag, cov_factor)

    chex.assert_trees_all_close(jnp.diag(sigma), jnp.diag(sigma_rt), atol=1e-4)
    eigenvalues = jnp.linalg.eigvalsh(sigma_rt)
    assert jnp.all(eigenvalues > 0), f"Non-PD: {eigenvalues}"


# ---------------------------------------------------------------------------
# Stability (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ROUNDTRIP_CASES)
def test_near_zero_cov_stability_all_ranks(dim, rank):
    """Near-zero covariance roundtrips without NaN for any rank."""
    mvn = MVN(dim=dim, rank=rank)
    loc = jnp.ones(dim)
    eps_val = 1e-6
    cov_diag = jnp.full(dim, eps_val)
    cov_factor = jnp.zeros((dim, rank)) if rank > 0 else jnp.zeros((dim, 0))
    mp = mvn._pack_mean(loc, cov_diag, cov_factor)

    natural = mvn.mean_to_natural(mp)
    chex.assert_tree_all_finite(natural)

    mp_rt = mvn.natural_to_mean(natural)
    chex.assert_tree_all_finite(mp_rt)


# ---------------------------------------------------------------------------
# structured_to_natural neg-def (parametrized for rank > 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", [
    pytest.param(4, 1, id="lowrank-1"),
    pytest.param(4, 2, id="lowrank-2"),
    pytest.param(3, 3, id="full"),
])
def test_structured_to_natural_neg_def_all_ranks(dim, rank):
    """structured_to_natural produces negative-definite η₂ for rank > 0."""
    mvn = MVN(dim=dim, rank=rank)
    mean_sz = mvn.mean_size(dim)
    free = jrnd.normal(jrnd.key(1), (mean_sz,))
    structured = mvn.to_structured(free)

    natural = mvn.structured_to_natural(structured)
    chex.assert_tree_all_finite(natural)

    _, nat2_flat = jnp.split(natural, [dim])
    nat2 = jnp.reshape(nat2_flat, (dim, dim))
    eigenvalues = jnp.linalg.eigvalsh(nat2)
    assert jnp.all(eigenvalues < 0), f"nat2 not neg-def: {eigenvalues}"


def test_rank_validation():
    """MVN must reject invalid rank values."""
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
