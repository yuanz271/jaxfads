import pytest
from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import MVN

# rank=0 → diagonal, rank=dim → full
_RANK_CASES = [(4, 4), (4, 0)]  # (dim, rank)


def _random_pd(key, dim: int, jitter: float = 1e-1) -> jnp.ndarray:
    A = jrnd.normal(key, (dim, dim))
    return A @ A.T + jitter * jnp.eye(dim)


def _random_diag_pd(key, dim: int, jitter: float = 1e-1) -> jnp.ndarray:
    v = jrnd.uniform(key, (dim,), minval=jitter, maxval=1.0 + jitter)
    return jnp.diag(v)


def _random_cov(key, dim: int, rank: int):
    return _random_pd(key, dim) if rank > 0 else _random_diag_pd(key, dim)


@pytest.mark.parametrize("dim,rank", [(1, 1), (1, 0), (4, 4), (4, 0)])
def test_param_size(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    expected = (dim + dim * dim) if rank > 0 else (2 * dim)
    assert mvn.param_size() == expected


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_pack_unpack_roundtrip(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    mean = jrnd.normal(jrnd.key(0), (dim,))
    cov = _random_cov(jrnd.key(1), dim, rank)

    moment = mvn.pack(mean, cov)
    mean_rt, cov_rt = mvn.unpack(moment)
    chex.assert_trees_all_close(mean_rt, mean, atol=1e-6)
    chex.assert_trees_all_close(cov_rt, cov, atol=1e-5)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_natural_moment_roundtrip(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    mean = jrnd.normal(jrnd.key(0), (dim,))
    cov = _random_cov(jrnd.key(1), dim, rank)
    moment = mvn.pack(mean, cov)

    natural = mvn.moment_to_natural(moment)
    moment_rt = mvn.natural_to_moment(natural)

    mean_rt, cov_rt = mvn.unpack(moment_rt)
    chex.assert_trees_all_close(mean_rt, mean, atol=1e-4)
    chex.assert_trees_all_close(cov_rt, cov, atol=1e-4)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_free_canon_roundtrip(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    loc = jrnd.normal(jrnd.key(0), (dim,))

    if rank > 0:
        chol_free = jrnd.normal(jrnd.key(1), (dim, dim))
        free = jnp.concatenate((loc, chol_free.ravel()))
    else:
        chol_diag_free = jrnd.normal(jrnd.key(1), (dim,))
        free = jnp.concatenate((loc, chol_diag_free))

    canon = mvn.free_to_canon(free)
    free_rt = mvn.canon_to_free(canon)
    canon_rt = mvn.free_to_canon(free_rt)

    # Tolerance is relaxed because the precision parameterization involves
    # two matrix inversions per roundtrip leg, accumulating float32 error.
    chex.assert_trees_all_close(canon.loc, canon_rt.loc, atol=1e-4)
    chex.assert_trees_all_close(canon.chol, canon_rt.chol, atol=1e-4)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_free_from_kw(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    free = mvn.free_from_kw(scale=2.0)
    canon = mvn.free_to_canon(free)
    moment = mvn.canon_to_moment(canon)
    mean, cov = mvn.unpack(moment)

    chex.assert_trees_all_close(mean, jnp.zeros(dim), atol=1e-6)
    chex.assert_trees_all_close(cov, 2.0 * jnp.eye(dim), atol=1e-5)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_predictive_moment_matches_closed_form(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    z = jrnd.normal(jrnd.key(0), (dim,))
    Q = _random_cov(jrnd.key(1), dim, rank)

    noise = mvn.pack(jnp.zeros(dim), Q)
    stats = mvn.predictive_moment(z, noise)

    expected = mvn.pack(z, Q)
    chex.assert_trees_all_close(stats, expected, atol=1e-6)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_sampling_matches_moments(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    mean = jrnd.normal(jrnd.key(0), (dim,))
    cov = _random_cov(jrnd.key(1), dim, rank)
    moment = mvn.pack(mean, cov)

    samples = mvn.sample_by_moment(jrnd.key(2), moment, 10_000)
    chex.assert_shape(samples, (10_000, dim))
    chex.assert_tree_all_finite(samples)

    chex.assert_trees_all_close(jnp.mean(samples, axis=0), mean, atol=0.08)
    chex.assert_trees_all_close(jnp.cov(samples.T), cov, atol=0.2)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_kl_matches_tfp(dim, rank):
    mvn = MVN(dim=dim, rank=rank)
    mean1 = jrnd.normal(jrnd.key(0), (dim,))
    mean2 = jrnd.normal(jrnd.key(2), (dim,))
    cov1 = _random_cov(jrnd.key(1), dim, rank)
    cov2 = _random_cov(jrnd.key(3), dim, rank)

    if rank > 0:
        expected = tfp.kl_divergence(
            tfp.MultivariateNormalFullCovariance(mean1, cov1),
            tfp.MultivariateNormalFullCovariance(mean2, cov2),
            allow_nan_stats=False,
        )
    else:
        expected = tfp.kl_divergence(
            tfp.MultivariateNormalDiag(mean1, scale_diag=jnp.sqrt(jnp.diag(cov1))),
            tfp.MultivariateNormalDiag(mean2, scale_diag=jnp.sqrt(jnp.diag(cov2))),
            allow_nan_stats=False,
        )

    moment1 = mvn.pack(mean1, cov1)
    moment2 = mvn.pack(mean2, cov2)

    chex.assert_trees_all_close(mvn.kl(moment1, moment2), expected, atol=1e-5)


# ---- Encoder free_to_natural tests ----


@pytest.mark.parametrize("rank", [0, 2, 3])
def test_encoder_free_to_natural_shapes_and_psd(rank):
    """free_to_natural produces correct-size natural params with PSD J."""
    dim = 3
    mvn = MVN(dim=dim, rank=rank)
    assert mvn.free_size() == 2 * dim + dim * rank

    free = jrnd.normal(jrnd.key(0), (mvn.free_size(),))
    natural = mvn.free_to_natural(free)
    chex.assert_shape(natural, (mvn.param_size(),))

    if rank > 0:
        J = jnp.reshape(natural[dim:], (dim, dim))
        chex.assert_trees_all_close(J, J.T, atol=1e-6)
        eigvals = jnp.linalg.eigvalsh(J)
        assert jnp.all(eigvals >= -1e-5)


def test_encoder_free_to_natural_diag_matches_softplus():
    """For rank=0, free_to_natural returns [h, softplus(d)]."""
    dim = 4
    mvn = MVN(dim=dim, rank=0)
    assert mvn.free_size() == 2 * dim

    free = jrnd.normal(jrnd.key(0), (mvn.free_size(),))
    natural = mvn.free_to_natural(free)

    from jaxfads.constraints import constrain_positive

    expected_h = free[:dim]
    expected_j = constrain_positive(free[dim:])
    chex.assert_trees_all_close(natural[:dim], expected_h, atol=1e-6)
    chex.assert_trees_all_close(natural[dim:], expected_j, atol=1e-6)


def test_encoder_free_hooks_low_rank_produces_psd_precision_update():
    dim, rank = 4, 2
    approx = MVN(dim=dim, rank=rank)

    assert approx.free_size() == 2 * dim + rank * dim
    assert approx.param_size() == dim + dim * dim

    free = jrnd.normal(jrnd.key(0), (approx.free_size(),))
    natural = approx.free_to_natural(free)
    chex.assert_shape(natural, (approx.param_size(),))

    J = jnp.reshape(natural[dim:], (dim, dim))
    chex.assert_trees_all_close(J, 0.5 * (J + J.T), atol=1e-6)

    x = jrnd.normal(jrnd.key(2), (dim,))
    quad = x @ J @ x
    assert quad >= -1e-5


@pytest.mark.parametrize("rank", [0, 2, 5])
def test_encoder_free_zero_baseline_matches_across_ranks(rank):
    """At free=0, all ranks produce the same isotropic baseline precision."""
    dim = 5
    mvn = MVN(dim=dim, rank=rank)
    full = MVN(dim=dim, rank=dim)

    nat = mvn.free_to_natural(jnp.zeros((mvn.free_size(),)))
    full_nat = full.free_to_natural(jnp.zeros((full.free_size(),)))

    chex.assert_shape(nat, (mvn.param_size(),))
    # h should be zero (no mean shift at free=0)
    chex.assert_trees_all_close(nat[:dim], jnp.zeros(dim), atol=1e-8)
    # J diagonal should match full-rank baseline
    if rank > 0:
        chex.assert_trees_all_close(nat[dim:], full_nat[dim:], atol=1e-6)
    else:
        # diag: precision = softplus(0) per element
        from jaxfads.constraints import constrain_positive

        expected_j = constrain_positive(jnp.zeros(dim))
        chex.assert_trees_all_close(nat[dim:], expected_j, atol=1e-6)


def test_registry_lookup():
    from jaxfads.base import Approx

    assert Approx.get_subclass("MVN") is MVN
