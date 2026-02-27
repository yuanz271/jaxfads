import pytest
from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import LoRaMVN, MVN


def _random_pd(key, dim: int, jitter: float = 1e-1) -> jnp.ndarray:
    A = jrnd.normal(key, (dim, dim))
    return A @ A.T + jitter * jnp.eye(dim)


def _random_diag_pd(key, dim: int, jitter: float = 1e-1) -> jnp.ndarray:
    v = jrnd.uniform(key, (dim,), minval=jitter, maxval=1.0 + jitter)
    return jnp.diag(v)


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [1, 4])
def test_param_size(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    expected = (dim + dim * dim) if structure == "full" else (2 * dim)
    assert mvn.param_size() == expected


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [4])
def test_pack_unpack_roundtrip(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    key = jrnd.key(0)
    mean = jrnd.normal(key, (dim,))

    if structure == "full":
        cov = _random_pd(jrnd.key(1), dim)
    else:
        cov = _random_diag_pd(jrnd.key(1), dim)

    moment = mvn.pack(mean, cov)
    mean_rt, cov_rt = mvn.unpack(moment)
    chex.assert_trees_all_close(mean_rt, mean, atol=1e-6)
    chex.assert_trees_all_close(cov_rt, cov, atol=1e-5)


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [4])
def test_natural_moment_roundtrip(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    mean = jrnd.normal(jrnd.key(0), (dim,))

    if structure == "full":
        cov = _random_pd(jrnd.key(1), dim)
    else:
        cov = _random_diag_pd(jrnd.key(1), dim)

    moment = mvn.pack(mean, cov)

    natural = mvn.moment_to_natural(moment)
    moment_rt = mvn.natural_to_moment(natural)

    mean_rt, cov_rt = mvn.unpack(moment_rt)
    chex.assert_trees_all_close(mean_rt, mean, atol=1e-4)
    chex.assert_trees_all_close(cov_rt, cov, atol=1e-4)


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [4])
def test_free_canon_roundtrip(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    key = jrnd.key(0)
    loc = jrnd.normal(key, (dim,))

    if structure == "full":
        chol_free = jrnd.normal(jrnd.key(1), (dim, dim))
        free = jnp.concatenate((loc, chol_free.ravel()))
    else:
        chol_diag_free = jrnd.normal(jrnd.key(1), (dim,))
        free = jnp.concatenate((loc, chol_diag_free))

    canon = mvn.free_to_canon(free)
    free_rt = mvn.canon_to_free(canon)
    canon_rt = mvn.free_to_canon(free_rt)

    # Roundtrip should preserve the constrained representation
    chex.assert_trees_all_close(canon.loc, canon_rt.loc, atol=1e-6)
    chex.assert_trees_all_close(canon.chol, canon_rt.chol, atol=1e-6)


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [4])
def test_free_from_kw(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    free = mvn.free_from_kw(scale=2.0)
    canon = mvn.free_to_canon(free)
    moment = mvn.canon_to_moment(canon)
    mean, cov = mvn.unpack(moment)

    chex.assert_trees_all_close(mean, jnp.zeros(dim), atol=1e-6)
    chex.assert_trees_all_close(cov, 2.0 * jnp.eye(dim), atol=1e-5)


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [4])
def test_predictive_moment_matches_closed_form(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    z = jrnd.normal(jrnd.key(0), (dim,))

    if structure == "full":
        Q = _random_pd(jrnd.key(1), dim)
    else:
        Q = _random_diag_pd(jrnd.key(1), dim)

    noise = mvn.pack(jnp.zeros(dim), Q)
    stats = mvn.predictive_moment(z, noise)

    expected = mvn.pack(z, Q)
    chex.assert_trees_all_close(stats, expected, atol=1e-6)


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [4])
def test_sampling_matches_moments(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    mean = jrnd.normal(jrnd.key(0), (dim,))

    if structure == "full":
        cov = _random_pd(jrnd.key(1), dim)
    else:
        cov = _random_diag_pd(jrnd.key(1), dim)

    moment = mvn.pack(mean, cov)

    samples = mvn.sample_by_moment(jrnd.key(2), moment, 10_000)
    chex.assert_shape(samples, (10_000, dim))
    chex.assert_tree_all_finite(samples)

    chex.assert_trees_all_close(jnp.mean(samples, axis=0), mean, atol=0.08)
    chex.assert_trees_all_close(jnp.cov(samples.T), cov, atol=0.2)


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("dim", [4])
def test_kl_matches_tfp(structure, dim):
    mvn = MVN(dim=dim, structure=structure)
    mean1 = jrnd.normal(jrnd.key(0), (dim,))
    mean2 = jrnd.normal(jrnd.key(2), (dim,))

    if structure == "full":
        cov1 = _random_pd(jrnd.key(1), dim)
        cov2 = _random_pd(jrnd.key(3), dim)
        expected = tfp.kl_divergence(
            tfp.MultivariateNormalFullCovariance(mean1, cov1),
            tfp.MultivariateNormalFullCovariance(mean2, cov2),
            allow_nan_stats=False,
        )
    else:
        cov1 = _random_diag_pd(jrnd.key(1), dim)
        cov2 = _random_diag_pd(jrnd.key(3), dim)
        expected = tfp.kl_divergence(
            tfp.MultivariateNormalDiag(mean1, scale_diag=jnp.sqrt(jnp.diag(cov1))),
            tfp.MultivariateNormalDiag(mean2, scale_diag=jnp.sqrt(jnp.diag(cov2))),
            allow_nan_stats=False,
        )

    moment1 = mvn.pack(mean1, cov1)
    moment2 = mvn.pack(mean2, cov2)

    chex.assert_trees_all_close(mvn.kl(moment1, moment2), expected, atol=1e-5)


def test_encoder_free_hooks_default_mvn_matches_standard_chain():
    mvn = MVN(dim=3, structure="full")
    assert mvn.encoder_free_size() == mvn.param_size()

    free = mvn.free_from_kw(scale=1.5)
    natural = mvn.encoder_free_to_natural(free)
    chex.assert_shape(natural, (mvn.param_size(),))

    canon = mvn.free_to_canon(free)
    moment = mvn.canon_to_moment(canon)
    expected = mvn.moment_to_natural(moment)
    chex.assert_trees_all_close(natural, expected, atol=1e-6)


def test_encoder_free_hooks_lora_mvn_produces_psd_precision_update():
    dim, rank = 4, 2
    approx = LoRaMVN(dim=dim, rank=rank)

    assert approx.encoder_free_size() == rank + rank * dim
    assert approx.param_size() == dim + dim * dim

    b = jrnd.normal(jrnd.key(0), (rank,))
    K = jrnd.normal(jrnd.key(1), (rank, dim))
    free = jnp.concatenate((b, K.ravel()))

    natural = approx.encoder_free_to_natural(free)
    chex.assert_shape(natural, (approx.param_size(),))

    h = natural[:dim]
    J = jnp.reshape(natural[dim:], (dim, dim))

    scale = 5.0
    chex.assert_trees_all_close(
        h, (scale * jnp.tanh(K)).T @ (scale * jnp.tanh(b)), atol=1e-6
    )
    chex.assert_trees_all_close(J, 0.5 * (J + J.T), atol=1e-6)

    # PSD sanity check: x^T J x >= 0 for random x.
    x = jrnd.normal(jrnd.key(2), (dim,))
    quad = x @ J @ x
    assert quad >= -1e-5


def test_registry_lookup():
    from jaxfads.base import Approx

    assert Approx.get_subclass("MVN") is MVN
    assert Approx.get_subclass("LoRaMVN") is LoRaMVN
