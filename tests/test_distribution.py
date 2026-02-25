import pytest
from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import MVN
from jaxfads.distributions.mvn import MVNParam


def _random_pd(key, dim: int, jitter: float = 1e-1) -> jnp.ndarray:
    A = jrnd.normal(key, (dim, dim))
    return A @ A.T + jitter * jnp.eye(dim)


@pytest.mark.parametrize("dim", [1, 2, 4])
def test_param_size(dim):
    mvn = MVN(dim=dim)
    assert mvn.param_size(dim) == dim + dim * dim


@pytest.mark.parametrize("dim", [2, 4])
def test_pack_unpack_roundtrip(dim):
    mvn = MVN(dim=dim)
    key = jrnd.key(0)
    mean = jrnd.normal(key, (dim,))
    cov = _random_pd(jrnd.key(1), dim)

    moment = mvn.pack(mean, cov)
    mean_rt, cov_rt = mvn.unpack(moment)
    chex.assert_trees_all_close(mean_rt, mean, atol=1e-6)
    chex.assert_trees_all_close(cov_rt, cov, atol=1e-5)


@pytest.mark.parametrize("dim", [2, 4])
def test_natural_moment_roundtrip(dim):
    mvn = MVN(dim=dim)
    mean = jrnd.normal(jrnd.key(0), (dim,))
    cov = _random_pd(jrnd.key(1), dim)
    moment = mvn.pack(mean, cov)

    natural = mvn.moment_to_natural(moment)
    moment_rt = mvn.natural_to_moment(natural)

    mean_rt, cov_rt = mvn.unpack(moment_rt)
    chex.assert_trees_all_close(mean_rt, mean, atol=1e-4)
    chex.assert_trees_all_close(cov_rt, cov, atol=1e-4)


@pytest.mark.parametrize("dim", [2, 4])
def test_free_canon_roundtrip(dim):
    mvn = MVN(dim=dim)
    key = jrnd.key(0)
    loc = jrnd.normal(key, (dim,))
    chol_free = jrnd.normal(jrnd.key(1), (dim, dim))
    free = MVNParam(loc=loc, chol=chol_free)

    canon = mvn.free_to_canon(free)
    free_rt = mvn.canon_to_free(canon)
    canon_rt = mvn.free_to_canon(free_rt)

    # Roundtrip should preserve the constrained representation
    chex.assert_trees_all_close(canon.loc, canon_rt.loc, atol=1e-6)
    chex.assert_trees_all_close(canon.chol, canon_rt.chol, atol=1e-6)


@pytest.mark.parametrize("dim", [2, 4])
def test_free_from_kw(dim):
    mvn = MVN(dim=dim)
    free = mvn.free_from_kw(scale=2.0)
    canon = mvn.free_to_canon(free)
    moment = mvn.canon_to_moment(canon)
    mean, cov = mvn.unpack(moment)

    chex.assert_trees_all_close(mean, jnp.zeros(dim), atol=1e-6)
    chex.assert_trees_all_close(cov, 2.0 * jnp.eye(dim), atol=1e-5)


@pytest.mark.parametrize("dim", [2, 4])
def test_predictive_moment_matches_closed_form(dim):
    mvn = MVN(dim=dim)
    z = jrnd.normal(jrnd.key(0), (dim,))
    Q = _random_pd(jrnd.key(1), dim)

    noise = mvn.pack(jnp.zeros(dim), Q)
    stats = mvn.predictive_moment(z, noise)

    expected = mvn.pack(z, Q)
    chex.assert_trees_all_close(stats, expected, atol=1e-6)


@pytest.mark.parametrize("dim", [2, 4])
def test_sampling_matches_moments(dim):
    mvn = MVN(dim=dim)
    mean = jrnd.normal(jrnd.key(0), (dim,))
    cov = _random_pd(jrnd.key(1), dim)
    moment = mvn.pack(mean, cov)

    samples = mvn.sample_by_moment(jrnd.key(2), moment, 50_000)
    chex.assert_shape(samples, (50_000, dim))
    chex.assert_tree_all_finite(samples)

    chex.assert_trees_all_close(jnp.mean(samples, axis=0), mean, atol=0.05)
    chex.assert_trees_all_close(jnp.cov(samples.T), cov, atol=0.1)


@pytest.mark.parametrize("dim", [2, 4])
def test_kl_matches_tfp(dim):
    mvn = MVN(dim=dim)
    mean1 = jrnd.normal(jrnd.key(0), (dim,))
    cov1 = _random_pd(jrnd.key(1), dim)
    mean2 = jrnd.normal(jrnd.key(2), (dim,))
    cov2 = _random_pd(jrnd.key(3), dim)

    moment1 = mvn.pack(mean1, cov1)
    moment2 = mvn.pack(mean2, cov2)

    expected = tfp.kl_divergence(
        tfp.MultivariateNormalFullCovariance(mean1, cov1),
        tfp.MultivariateNormalFullCovariance(mean2, cov2),
        allow_nan_stats=False,
    )
    chex.assert_trees_all_close(mvn.kl(moment1, moment2), expected, atol=1e-5)


def test_registry_lookup():
    from jaxfads.base import Approx

    assert Approx.get_subclass("MVN") is MVN
