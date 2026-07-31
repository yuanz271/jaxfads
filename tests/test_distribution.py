import chex
import jax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfp
from jax import numpy as jnp
from jax import random as jrnd

from jaxfads.base import Approx
from jaxfads.core import propagate_transition_points
from jaxfads.distributions import MVN
from jaxfads.noise import Noise

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


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_transition_points_sigma_points_shape_and_weights(dim, rank):
    """MVN(use_sigma_points=True).transition_points returns 2*dim+1 points
    and weights summing to 1, for both diag (rank=0) and full (rank=dim)
    layouts."""
    mvn = MVN(dim=dim, rank=rank, use_sigma_points=True)
    mean = jrnd.normal(jrnd.key(0), (dim,))
    cov = _random_cov(jrnd.key(1), dim, rank)
    moment = mvn.pack(mean, cov)

    points, weights = mvn.transition_points(jrnd.key(2), moment, mc_size=4)

    chex.assert_shape(points, (2 * dim + 1, dim))
    chex.assert_shape(weights, (2 * dim + 1,))
    chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-5)
    chex.assert_trees_all_close(points[0], mean, atol=1e-5)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_transition_points_explicit_mc_matches_base_contract(dim, rank):
    """MVN(use_sigma_points=False) (opt-out; MVN defaults to True) still
    reproduces the base class's plain Monte Carlo contract exactly:
    mc_size samples, uniform weights 1/mc_size."""
    mvn = MVN(dim=dim, rank=rank, use_sigma_points=False)
    mean = jrnd.normal(jrnd.key(0), (dim,))
    cov = _random_cov(jrnd.key(1), dim, rank)
    moment = mvn.pack(mean, cov)
    mc_size = 8

    points, weights = mvn.transition_points(jrnd.key(2), moment, mc_size)

    chex.assert_shape(points, (mc_size, dim))
    chex.assert_trees_all_close(weights, jnp.full((mc_size,), 1.0 / mc_size), atol=1e-8)


def test_noise_strategy_registration_is_exact_class_only():
    """Noise uses exact Approx-class registration, not MRO fallback."""

    class ExactMVN(MVN):
        pass

    approx = ExactMVN(dim=2, rank=2)
    noise = Noise(approx=approx, free=approx.free_from_kw(scale=1.0))
    assert not noise.supports_mstep

    Noise.register_mstep(ExactMVN, Noise._mstep_strategies[MVN])
    try:
        assert noise.supports_mstep
    finally:
        del Noise._mstep_strategies[ExactMVN]


def test_unregistered_noise_strategy_is_noop():
    """An exact-unregistered Approx remains usable without MAP-Q behavior."""

    class _NoStat(Approx):
        def natural_to_moment(self, natural):
            return natural

        def moment_to_natural(self, moment):
            return moment

        def sample_by_moment(self, key, moment, mc_size):
            return moment

        def param_size(self):
            return 1

        def kl(self, moment1, moment2):
            return jnp.array(0.0)

        def free_to_canon(self, free):
            return free

        def canon_to_moment(self, canon):
            return canon

        def canon_to_free(self, canon):
            return canon

        def moment_to_canon(self, moment):
            return moment

        def predictive_moment(self, z, noise):
            return z

        def free_from_kw(self, **kwargs):
            return jnp.array(0.0)

    approx = _NoStat()
    noise = Noise(approx=approx, free=jnp.array(0.0))
    assert not noise.supports_mstep
    assert noise.collect_minibatch_stat(jnp.zeros((1, 2, 1)), None) is None
    assert noise.mstep(None, prior=None) is noise


def test_approx_transition_stat_default_is_identity():
    """Approx.transition_stat's base-class default is a no-op identity
    pass-through of (zs, weights) unchanged -- since core.py calls this
    unconditionally for every model regardless of whether shrink/Q-
    estimation is configured, a non-overriding subclass must keep
    behaving exactly as if this method didn't exist at all (the raw
    point set passed straight through as transition_stat)."""

    class _NoReduce(Approx):
        def natural_to_moment(self, natural):
            return natural

        def moment_to_natural(self, moment):
            return moment

        def sample_by_moment(self, key, moment, mc_size):
            return moment

        def param_size(self):
            return 1

        def kl(self, moment1, moment2):
            return jnp.array(0.0)

        def free_to_canon(self, free):
            return free

        def canon_to_moment(self, canon):
            return canon

        def canon_to_free(self, canon):
            return canon

        def moment_to_canon(self, moment):
            return moment

        def predictive_moment(self, z, noise):
            return z

        def free_from_kw(self, **kwargs):
            return jnp.array(0.0)

    zs = jrnd.normal(jrnd.key(0), (5, 3))
    weights = jnp.full((5,), 0.2)
    out_zs, out_weights = _NoReduce().transition_stat(zs, weights)
    chex.assert_trees_all_close(out_zs, zs)
    chex.assert_trees_all_close(out_weights, weights)


@pytest.mark.parametrize("dim,rank", _RANK_CASES)
def test_mvn_noise_mstep_matches_independent_reference(dim, rank):
    """The exact registered MVN Noise strategy matches an independent Q
    MAP reference for both diagonal and full MVN layouts."""
    approx = MVN(dim=dim, rank=rank)
    noise = Noise(approx=approx, free=approx.free_from_kw(scale=1.0))

    A = 0.9 * jnp.eye(dim) + 0.05 * jrnd.normal(jrnd.key(9), (dim, dim))
    b = jrnd.normal(jrnd.key(10), (dim,))

    n_batch, n_time = 2, 4
    n_pairs = n_batch * (n_time - 1)
    key = jrnd.key(0)
    keys = jrnd.split(key, 2)
    means = jrnd.normal(keys[0], (n_batch, n_time, dim))
    flat_cov_keys = jrnd.split(keys[1], n_batch * n_time)
    flat_covs = jax.vmap(lambda k: _random_cov(k, dim, rank))(flat_cov_keys)
    covs = flat_covs.reshape(n_batch, n_time, dim, dim)

    moment = jax.vmap(jax.vmap(approx.pack))(means, covs)

    prior_value = 0.5
    prior_dof = 2.0
    prior = (prior_value, prior_dof)

    # transition_stat = approx.transition_stat(zs, weights) -- exactly as
    # core._site_filter/nofilt/causal would compute it: propagate via
    # core.propagate_transition_points (core.py's own agnostic recursion),
    # then reduce via this same class's own transition_stat override
    # (MVN reduces to a weighted mean/covariance pair; core.py never
    # assumes that reduction itself). Since the transition is affine, UT
    # (MVN's default transition_points policy) recovers mean_f = A @
    # mean_tm1 + b, cov_f = A @ cov_tm1 @ A.T exactly.
    def f(z, u, c, *, key=None):
        del u, c, key
        return A @ z + b

    moment_tm1 = moment[:, :-1, :]
    u_zeros = jnp.zeros((n_batch, n_time - 1, 0))
    c_zeros = jnp.zeros((n_batch, n_time - 1, 0))
    keys = jrnd.split(jrnd.key(11), n_batch * (n_time - 1)).reshape(n_batch, n_time - 1)

    def _propagate_and_reduce(key_i, moment_i, u_i, c_i):
        zs, weights = propagate_transition_points(
            key_i, moment_i, u_i, c_i, f, approx, mc_size=1
        )
        return approx.transition_stat(zs, weights)

    transition_stat = jax.vmap(jax.vmap(_propagate_and_reduce))(
        keys, moment_tm1, u_zeros, c_zeros
    )

    stat = noise.collect_minibatch_stat(moment, transition_stat)
    updated_noise = noise.mstep(stat, prior=prior)
    canon = approx.free_to_canon(updated_noise.free)
    got_cov = canon.chol @ canon.chol.T

    raw_stats = []
    for bi in range(n_batch):
        for ti in range(n_time - 1):
            r = means[bi, ti + 1] - (A @ means[bi, ti] + b)
            raw_stats.append(
                jnp.outer(r, r) + covs[bi, ti + 1] + A @ covs[bi, ti] @ A.T
            )
    mean_stat = jnp.mean(jnp.stack(raw_stats), axis=0)
    expected_value = prior_value * jnp.eye(dim)
    expected_cov = (n_pairs * mean_stat + prior_dof * expected_value) / (
        n_pairs + prior_dof
    )
    if rank == 0:
        expected_cov = jnp.diag(jnp.diagonal(expected_cov))

    chex.assert_trees_all_close(got_cov, expected_cov, atol=1e-4)
    chex.assert_trees_all_close(canon.loc, jnp.zeros(dim), atol=1e-6)
