import pytest
from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import MVN


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

_ALL_RANKS = [
    pytest.param(4, 0, id="diag"),
    pytest.param(4, 1, id="lowrank-1"),
    pytest.param(4, 2, id="lowrank-2"),
    pytest.param(3, 3, id="full"),
]

_LOWRANK_ONLY = [
    pytest.param(4, 1, id="lowrank-1"),
    pytest.param(4, 2, id="lowrank-2"),
    pytest.param(3, 3, id="full"),
]


def _make_mean(mvn: MVN, key) -> jnp.ndarray:
    """Build a valid structured mean for any rank."""
    d, r = mvn._dim, mvn._rank
    k1, k2, k3 = jrnd.split(key, 3)
    loc = jrnd.normal(k1, (d,))
    cov_diag = jnp.abs(jrnd.normal(k2, (d,))) + 0.1
    cov_factor = jrnd.normal(k3, (d, r)) * 0.5 if r > 0 else jnp.zeros((d, 0))
    return mvn._pack_mean(loc, cov_diag, cov_factor)


# ---------------------------------------------------------------------------
# natural ↔ mean roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_natural_mean_roundtrip(dim, rank):
    """mean_to_natural → natural_to_mean preserves covariance structure."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(42))

    natural = mvn.mean_to_natural(mp)
    chex.assert_tree_all_finite(natural)
    chex.assert_shape(natural, (mvn.param_size(dim),))

    mp_rt = mvn.natural_to_mean(natural)
    chex.assert_tree_all_finite(mp_rt)

    _, cov_orig = mvn.unpack(mp)
    _, cov_rt = mvn.unpack(mp_rt)
    chex.assert_trees_all_close(
        jnp.diag(mvn.full_cov(cov_orig)), jnp.diag(mvn.full_cov(cov_rt)), atol=1e-4
    )
    eigenvalues = jnp.linalg.eigvalsh(mvn.full_cov(cov_rt))
    assert jnp.all(eigenvalues > 0), f"Non-PD roundtrip cov: {eigenvalues}"


# ---------------------------------------------------------------------------
# param_from_conf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_param_from_conf(dim, rank, scale):
    """param_from_conf → to_structured gives isotropic N(0, scale·I)."""
    mvn = MVN(dim=dim, rank=rank)
    free = mvn.param_from_conf(scale=scale)
    structured = mvn.to_structured(free)

    loc, cov = mvn.unpack(structured)
    chex.assert_trees_all_close(loc, jnp.zeros(dim), atol=1e-6)
    chex.assert_trees_all_close(mvn.full_cov(cov), jnp.eye(dim) * scale, atol=1e-5)


# ---------------------------------------------------------------------------
# to_structured / to_free roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_to_structured_roundtrip(dim, rank):
    """to_structured(to_free(m)) ≈ m."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(10))
    recovered = mvn.to_structured(mvn.to_free(mp))
    chex.assert_trees_all_close(recovered, mp, atol=1e-5)


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_to_structured_produces_valid_cov(dim, rank):
    """to_structured produces PD covariance from arbitrary free-form."""
    mvn = MVN(dim=dim, rank=rank)
    free = jrnd.normal(jrnd.key(0), (mvn.mean_size(dim),))
    mp = mvn.to_structured(free)

    _, cov = mvn.unpack(mp)
    cov_full = mvn.full_cov(cov)
    chex.assert_trees_all_close(cov_full, cov_full.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov_full)
    assert jnp.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"


# ---------------------------------------------------------------------------
# structured_to_natural (MVN-specific, not on ABC)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _LOWRANK_ONLY)
def test_structured_to_natural_neg_def(dim, rank):
    """structured_to_natural produces negative-definite η₂ for rank > 0."""
    mvn = MVN(dim=dim, rank=rank)
    structured = mvn.to_structured(jrnd.normal(jrnd.key(1), (mvn.mean_size(dim),)))
    natural = mvn.structured_to_natural(structured)
    chex.assert_tree_all_finite(natural)

    _, nat2_flat = jnp.split(natural, [dim])
    nat2 = jnp.reshape(nat2_flat, (dim, dim))
    eigenvalues = jnp.linalg.eigvalsh(nat2)
    assert jnp.all(eigenvalues < 0), f"nat2 not neg-def: {eigenvalues}"


# ---------------------------------------------------------------------------
# canon ↔ mean roundtrip
# ---------------------------------------------------------------------------


def test_canon_mean_roundtrip_diag():
    """unpack → pack is exact for rank 0."""
    mvn = MVN(dim=4, rank=0)
    mp = _make_mean(mvn, jrnd.key(77))
    loc, cov = mvn.unpack(mp)
    chex.assert_trees_all_close(mvn.pack(loc, cov), mp, atol=1e-6)


@pytest.mark.parametrize("dim, rank", _LOWRANK_ONLY)
def test_canon_mean_roundtrip_lowrank(dim, rank):
    """Diagonal and loc are preserved; off-diagonal is approximate."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(77))
    loc, cov = mvn.unpack(mp)
    mp_rt = mvn.pack(loc, cov)

    loc_rt, cov_rt = mvn.unpack(mp_rt)
    chex.assert_trees_all_close(loc, loc_rt, atol=1e-5)
    chex.assert_trees_all_close(
        jnp.diag(mvn.full_cov(cov)), jnp.diag(mvn.full_cov(cov_rt)), atol=1e-4
    )
    eigenvalues = jnp.linalg.eigvalsh(mvn.full_cov(cov_rt))
    assert jnp.all(eigenvalues > 0), f"Non-PD roundtrip cov: {eigenvalues}"


# ---------------------------------------------------------------------------
# full_cov
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_full_cov(dim, rank):
    """full_cov returns symmetric PD (D, D) matrix."""
    mvn = MVN(dim=dim, rank=rank)
    _, cov = mvn.unpack(_make_mean(mvn, jrnd.key(88)))
    cov_full = mvn.full_cov(cov)

    chex.assert_shape(cov_full, (dim, dim))
    chex.assert_tree_all_finite(cov_full)
    chex.assert_trees_all_close(cov_full, cov_full.T, atol=1e-6)
    assert jnp.all(jnp.linalg.eigvalsh(cov_full) > 0)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def test_kl_matches_closed_form():
    """MVN(rank=0).kl() matches closed-form diagonal Gaussian KL."""

    def _kl_diag(m1, v1, m2, v2):
        return 0.5 * jnp.sum(jnp.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    mvn = MVN(dim=3, rank=0)
    mp1 = mvn.pack(jnp.array([1.0, -0.5, 0.3]), jnp.array([0.5, 2.0, 0.1]))
    mp2 = mvn.pack(jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, 0.5]))

    chex.assert_trees_all_close(
        mvn.kl(mp1, mp2),
        _kl_diag(jnp.array([1.0, -0.5, 0.3]), jnp.array([0.5, 2.0, 0.1]),
                 jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, 0.5])),
        atol=1e-5,
    )


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_kl_properties(dim, rank):
    """KL(q, q) = 0 and KL(q, p) > 0 when q ≠ p."""
    mvn = MVN(dim=dim, rank=rank)
    mp1 = _make_mean(mvn, jrnd.key(10))
    mp2 = _make_mean(mvn, jrnd.key(20))

    chex.assert_trees_all_close(mvn.kl(mp1, mp1), jnp.array(0.0), atol=1e-4)
    assert float(mvn.kl(mp1, mp2)) > 0.0


def test_reparameterization(spec):
    assert (
        tfp.MultivariateNormalFullCovariance(
            jnp.ones(spec["state_dim"]), jnp.eye(spec["state_dim"])
        ).reparameterization_type
        == tfp.FULLY_REPARAMETERIZED
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_sample_shape_and_statistics(dim, rank):
    """Samples have correct shape; mean and cov approximate parameters."""
    mvn = MVN(dim=dim, rank=rank)
    mp = _make_mean(mvn, jrnd.key(0))
    loc, cov = mvn.unpack(mp)
    cov_full = mvn.full_cov(cov)

    samples = mvn.sample_by_mean(jrnd.key(1), mp, 50_000)
    chex.assert_shape(samples, (50_000, dim))
    chex.assert_tree_all_finite(samples)
    chex.assert_trees_all_close(jnp.mean(samples, axis=0), loc, atol=0.05)
    chex.assert_trees_all_close(jnp.cov(samples.T), cov_full, atol=0.1)


# ---------------------------------------------------------------------------
# predict_mean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_predict_mean_single_loc(dim, rank):
    """Single loc recovers (loc, noise_cov)."""
    mvn = MVN(dim=dim, rank=rank)
    scale = 1.5
    noise_mean = mvn.to_structured(mvn.param_from_conf(scale=scale))
    loc = jrnd.normal(jrnd.key(0), (dim,))

    mp = mvn.predict_mean(loc[None, :], noise_mean)
    recovered_loc, recovered_cov = mvn.unpack(mp)
    chex.assert_trees_all_close(recovered_loc, loc, atol=1e-5)
    chex.assert_trees_all_close(mvn.full_cov(recovered_cov), jnp.eye(dim) * scale, atol=1e-4)


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_predict_mean_captures_variance(dim, rank):
    """Spread locs produce larger covariance than noise alone."""
    mvn = MVN(dim=dim, rank=rank)
    scale = 0.1
    noise_mean = mvn.to_structured(mvn.param_from_conf(scale=scale))
    locs = jrnd.normal(jrnd.key(42), (200, dim)) * 3.0

    _, cov = mvn.unpack(mvn.predict_mean(locs, noise_mean))
    assert jnp.all(jnp.diag(mvn.full_cov(cov)) > scale)


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_predict_mean_all_nan(dim, rank):
    """All-NaN locs → NaN output."""
    mvn = MVN(dim=dim, rank=rank)
    noise_mean = mvn.to_structured(mvn.param_from_conf(scale=1.0))
    mp = mvn.predict_mean(jnp.full((5, dim), jnp.nan), noise_mean)
    assert not jnp.any(jnp.isfinite(mp))


# ---------------------------------------------------------------------------
# _decompose_cov (lowrank only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _LOWRANK_ONLY)
def test_decompose_cov_preserves_diagonal(dim, rank):
    """_decompose_cov → _build_cov preserves diagonal and PD."""
    mvn = MVN(dim=dim, rank=rank)
    A = jrnd.normal(jrnd.key(99), (dim, dim))
    sigma = A @ A.T + 0.5 * jnp.eye(dim)

    cov_diag, cov_factor = mvn._decompose_cov(sigma)
    sigma_rt = mvn._build_cov(cov_diag, cov_factor)

    chex.assert_trees_all_close(jnp.diag(sigma), jnp.diag(sigma_rt), atol=1e-4)
    assert jnp.all(jnp.linalg.eigvalsh(sigma_rt) > 0)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, rank", _ALL_RANKS)
def test_near_zero_cov_stability(dim, rank):
    """Near-zero covariance roundtrips without NaN or extreme values."""
    mvn = MVN(dim=dim, rank=rank)
    loc = jnp.ones(dim)
    cov_diag = jnp.full(dim, 1e-6)
    cov_factor = jnp.zeros((dim, rank)) if rank > 0 else jnp.zeros((dim, 0))
    mp = mvn._pack_mean(loc, cov_diag, cov_factor)

    natural = mvn.mean_to_natural(mp)
    chex.assert_tree_all_finite(natural)
    if rank == 0:
        _, nat2 = jnp.split(natural, 2)
        assert float(jnp.max(jnp.abs(nat2)).item()) < 1e7

    mp_rt = mvn.natural_to_mean(natural)
    chex.assert_tree_all_finite(mp_rt)


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
# Validation & registry
# ---------------------------------------------------------------------------


def test_rank_validation():
    """MVN must reject invalid rank values."""
    with pytest.raises(ValueError, match="rank must satisfy"):
        MVN(dim=3, rank=-1)
    with pytest.raises(ValueError, match="rank must satisfy"):
        MVN(dim=3, rank=4)


def test_registry_lookup():
    """SubclassRegistryMixin finds MVN by name."""
    from jaxfads.base import Approx

    assert Approx.get_subclass("MVN") is MVN
