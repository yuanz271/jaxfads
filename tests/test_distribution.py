from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import MVN, DiagMVN, FullMVN


# ---------------------------------------------------------------------------
# DiagMVN (rank=0) — backward-compatible tests
# ---------------------------------------------------------------------------


def test_diagmvn(spec):
    state_dim = spec["state_dim"]

    m1 = jnp.ones(state_dim)
    cov1 = jnp.ones(state_dim)
    m2 = jnp.zeros(state_dim)
    cov2 = jnp.ones(state_dim) * 2

    moment = DiagMVN.canon_to_moment(m1, cov1)
    nat1 = DiagMVN.moment_to_natural(moment)
    moment1 = DiagMVN.natural_to_moment(nat1)

    chex.assert_trees_all_close(moment, moment1)

    moment2 = DiagMVN.canon_to_moment(m2, cov2)
    kl = DiagMVN.kl(moment1, moment2)
    chex.assert_tree_all_finite(kl)


def test_reparameterization(spec):
    state_dim = spec["state_dim"]

    m1 = jnp.ones(state_dim)
    cov1 = jnp.eye(state_dim)
    assert (
        tfp.MultivariateNormalFullCovariance(m1, cov1).reparameterization_type
        == tfp.FULLY_REPARAMETERIZED
    )


def test_diagmvn_near_zero_cov_stability():
    """Near-zero covariance must not produce extreme natural parameters."""
    state_dim = 2
    mean = jnp.ones(state_dim)

    eps = jnp.finfo(jnp.float32).eps
    tiny_cov = jnp.full(state_dim, eps)

    moment = DiagMVN.canon_to_moment(mean, tiny_cov)
    natural = DiagMVN.moment_to_natural(moment)

    chex.assert_tree_all_finite(natural)

    _, nat2 = jnp.split(natural, 2)
    assert float(jnp.max(jnp.abs(nat2)).item()) < 1e7

    moment_rt = DiagMVN.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment_rt)

    normal_cov = jnp.ones(state_dim)
    moment_normal = DiagMVN.canon_to_moment(mean, normal_cov)
    natural_normal = DiagMVN.moment_to_natural(moment_normal)
    moment_normal_rt = DiagMVN.natural_to_moment(natural_normal)
    chex.assert_trees_all_close(moment_normal, moment_normal_rt)


def test_diagmvn_kl_matches_closed_form():
    """DiagMVN.kl() must match the closed-form diagonal Gaussian KL."""

    def _kl_closed_form(m1, v1, m2, v2):
        return 0.5 * jnp.sum(jnp.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    m1 = jnp.array([1.0, -0.5, 0.3])
    v1 = jnp.array([0.5, 2.0, 0.1])
    m2 = jnp.array([0.0, 0.0, 1.0])
    v2 = jnp.array([1.0, 1.0, 0.5])

    moment1 = DiagMVN.canon_to_moment(m1, v1)
    moment2 = DiagMVN.canon_to_moment(m2, v2)

    kl_actual = DiagMVN.kl(moment1, moment2)
    kl_expected = _kl_closed_form(m1, v1, m2, v2)

    chex.assert_tree_all_finite(kl_actual)
    chex.assert_trees_all_close(kl_actual, kl_expected, atol=1e-5)

    m1b = jnp.array([3.0, -2.0])
    v1b = jnp.array([0.1, 10.0])
    m2b = jnp.array([0.0, 1.0])
    v2b = jnp.array([5.0, 0.3])

    moment1b = DiagMVN.canon_to_moment(m1b, v1b)
    moment2b = DiagMVN.canon_to_moment(m2b, v2b)

    kl_actual_b = DiagMVN.kl(moment1b, moment2b)
    kl_expected_b = _kl_closed_form(m1b, v1b, m2b, v2b)

    chex.assert_tree_all_finite(kl_actual_b)
    chex.assert_trees_all_close(kl_actual_b, kl_expected_b, atol=1e-4)

    kl_self = DiagMVN.kl(moment1, moment1)
    chex.assert_trees_all_close(kl_self, jnp.array(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# FullMVN (rank=-1) — backward-compatible tests
# ---------------------------------------------------------------------------


def test_fullmvn_constrain_moment():
    """FullMVN.constrain_moment produces valid structured moment."""
    state_dim = 3
    r = state_dim  # full rank
    unc_size = state_dim * (2 + r)
    unconstrained = jrnd.normal(jrnd.key(0), (unc_size,))

    moment = FullMVN.constrain_moment(unconstrained)
    chex.assert_shape(moment, (unc_size,))
    chex.assert_tree_all_finite(moment)

    mean, cov = FullMVN.moment_to_canon(moment)
    chex.assert_shape(mean, (state_dim,))
    chex.assert_shape(cov, (state_dim, state_dim))

    chex.assert_trees_all_close(cov, cov.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"


def test_fullmvn_constrain_natural():
    """FullMVN.constrain_natural produces valid natural params."""
    state_dim = 3
    param_sz = FullMVN.param_size(state_dim)  # D + D²
    unconstrained = jrnd.normal(jrnd.key(1), (param_sz,))

    natural = FullMVN.constrain_natural(unconstrained)
    chex.assert_shape(natural, (param_sz,))
    chex.assert_tree_all_finite(natural)

    nat1, nat2_flat = jnp.split(natural, [state_dim])
    nat2 = jnp.reshape(nat2_flat, (state_dim, state_dim))
    eigenvalues = jnp.linalg.eigvalsh(nat2)
    assert jnp.all(eigenvalues < 0), f"nat2 not neg-def: eigenvalues={eigenvalues}"

    moment = FullMVN.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment)


def test_fullmvn_unconstrain_natural_roundtrip():
    """unconstrain_natural inverts constrain_natural for FullMVN."""
    state_dim = 3
    natural = FullMVN.prior_natural(state_dim)
    unconstrained = FullMVN.unconstrain_natural(natural)
    chex.assert_tree_all_finite(unconstrained)

    natural_rt = FullMVN.constrain_natural(unconstrained)
    chex.assert_tree_all_finite(natural_rt)

    moment = FullMVN.natural_to_moment(natural)
    moment_rt = FullMVN.natural_to_moment(natural_rt)
    chex.assert_trees_all_close(moment, moment_rt, atol=1e-4)


def test_lowrankcov(capsys):
    loc = jnp.ones(2)
    cov_diag = jnp.ones(2)
    cov_lr = jnp.ones((2, 1))

    _ = tfp.MultivariateNormalDiagPlusLowRankCovariance(loc, cov_diag, cov_lr)


# ---------------------------------------------------------------------------
# Unconstrain / constrain roundtrip
# ---------------------------------------------------------------------------


def test_diagmvn_unconstrain_moment_roundtrip():
    """constrain_moment(unconstrain_moment(m)) ≈ m for DiagMVN."""
    state_dim = 4
    mean = jrnd.normal(jrnd.key(10), (state_dim,))
    cov = jnp.abs(jrnd.normal(jrnd.key(11), (state_dim,))) + 0.1
    moment = DiagMVN.canon_to_moment(mean, cov)

    unconstrained = DiagMVN.unconstrain_moment(moment)
    recovered = DiagMVN.constrain_moment(unconstrained)
    chex.assert_trees_all_close(recovered, moment, atol=1e-5)


def test_fullmvn_unconstrain_moment_roundtrip():
    """constrain_moment(unconstrain_moment(m)) ≈ m for FullMVN."""
    state_dim = 3
    r = state_dim

    # Build a structured moment: isotropic N(mean, 2I)
    mean = jrnd.normal(jrnd.key(20), (state_dim,))
    cov_diag = jnp.full(state_dim, 2.0)
    cov_factor = jnp.zeros((state_dim, r))
    moment = jnp.concatenate((mean, cov_diag, cov_factor.flatten()))

    unconstrained = FullMVN.unconstrain_moment(moment)
    recovered = FullMVN.constrain_moment(unconstrained)
    chex.assert_trees_all_close(recovered, moment, atol=1e-4)


def test_init_noise_produces_expected_moment():
    """Approx.init_noise → constrain_moment gives correct noise distribution."""
    state_dim = 3
    cov_init = 2.0

    unconstrained = DiagMVN.init_noise(cov_init, state_dim)
    moment = DiagMVN.constrain_moment(unconstrained)
    mean, var = DiagMVN.moment_to_canon(moment)

    chex.assert_trees_all_close(mean, jnp.zeros(state_dim), atol=1e-6)
    chex.assert_trees_all_close(var, jnp.full(state_dim, cov_init), atol=1e-5)


# ---------------------------------------------------------------------------
# Unified MVN — param_size / moment_size
# ---------------------------------------------------------------------------


def test_param_size():
    assert DiagMVN.param_size(3) == 6
    assert FullMVN.param_size(3) == 12
    assert DiagMVN.moment_size(3) == 6  # D*(2+0)


def test_moment_size_full():
    # FullMVN: rank=-1 → effective rank = D = 3, moment_size = 3*(2+3) = 15
    assert FullMVN.moment_size(3) == 15


# ---------------------------------------------------------------------------
# Low-rank MVN (rank=1, rank=2)
# ---------------------------------------------------------------------------


class LowRank1(MVN):
    _rank = 1


class LowRank2(MVN):
    _rank = 2


def test_lowrank_param_and_moment_sizes():
    d = 4
    assert LowRank1.param_size(d) == d + d * d  # 4 + 16 = 20
    assert LowRank1.moment_size(d) == d * (2 + 1)  # 4 * 3 = 12
    assert LowRank2.moment_size(d) == d * (2 + 2)  # 4 * 4 = 16


def test_lowrank_natural_moment_roundtrip():
    """moment_to_natural → natural_to_moment preserves structure for low-rank MVN."""
    d = 4
    key = jrnd.key(42)

    # Build a structured moment: diag + rank-1 factor
    loc = jrnd.normal(key, (d,))
    cov_diag = jnp.ones(d)
    cov_factor = jrnd.normal(key, (d, 1)) * 0.5
    moment = jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    natural = LowRank1.moment_to_natural(moment)
    chex.assert_tree_all_finite(natural)
    chex.assert_shape(natural, (LowRank1.param_size(d),))

    moment_rt = LowRank1.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment_rt)
    chex.assert_shape(moment_rt, (LowRank1.moment_size(d),))

    # Diagonal of covariance should be preserved closely
    _, cov_orig = LowRank1.moment_to_canon(moment)
    _, cov_rt = LowRank1.moment_to_canon(moment_rt)
    chex.assert_trees_all_close(
        jnp.diag(cov_orig), jnp.diag(cov_rt), atol=1e-4
    )
    # Roundtrip covariance must be PSD
    eigenvalues = jnp.linalg.eigvalsh(cov_rt)
    assert jnp.all(eigenvalues > 0), f"Non-PD roundtrip cov: {eigenvalues}"


def test_lowrank_sample():
    """Sampling from low-rank MVN produces correct shapes."""
    d = 3
    loc = jnp.zeros(d)
    cov_diag = jnp.ones(d)
    cov_factor = jrnd.normal(jrnd.key(0), (d, 2)) * 0.3
    moment = jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    samples = LowRank2.sample_by_moment(jrnd.key(1), moment, 50)
    chex.assert_shape(samples, (50, d))
    chex.assert_tree_all_finite(samples)


def test_lowrank_kl_self_zero():
    """KL(q, q) = 0 for low-rank MVN."""
    d = 3
    loc = jnp.array([1.0, -0.5, 0.3])
    cov_diag = jnp.array([0.5, 1.0, 0.8])
    cov_factor = jrnd.normal(jrnd.key(5), (d, 1)) * 0.5
    moment = jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    kl = LowRank1.kl(moment, moment)
    chex.assert_trees_all_close(kl, jnp.array(0.0), atol=1e-4)


def test_lowrank_constrain_moment_roundtrip():
    """constrain(unconstrain(moment)) ≈ moment for low-rank MVN."""
    d = 4
    loc = jrnd.normal(jrnd.key(10), (d,))
    cov_diag = jnp.abs(jrnd.normal(jrnd.key(11), (d,))) + 0.1
    cov_factor = jrnd.normal(jrnd.key(12), (d, 2)) * 0.3
    moment = jnp.concatenate((loc, cov_diag, cov_factor.flatten()))

    unc = LowRank2.unconstrain_moment(moment)
    recovered = LowRank2.constrain_moment(unc)
    chex.assert_trees_all_close(recovered, moment, atol=1e-5)


def test_lowrank_init_noise():
    """init_noise produces isotropic noise moment for low-rank MVN."""
    d = 3
    scale = 2.5
    unc = LowRank1.init_noise(scale, d)
    moment = LowRank1.constrain_moment(unc)

    loc, cov_diag, cov_factor = LowRank1._split_moment(moment)
    chex.assert_trees_all_close(loc, jnp.zeros(d), atol=1e-6)
    chex.assert_trees_all_close(cov_diag, jnp.full(d, scale), atol=1e-5)
    chex.assert_trees_all_close(cov_factor, jnp.zeros((d, 1)), atol=1e-6)


def test_fullmvn_natural_moment_roundtrip():
    """FullMVN natural ↔ moment roundtrip via structured format."""
    d = 3
    natural = FullMVN.prior_natural(d)  # N(0, I)
    moment = FullMVN.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment)

    mean, cov = FullMVN.moment_to_canon(moment)
    chex.assert_trees_all_close(mean, jnp.zeros(d), atol=1e-5)
    chex.assert_trees_all_close(cov, jnp.eye(d), atol=1e-4)

    natural_rt = FullMVN.moment_to_natural(moment)
    chex.assert_trees_all_close(natural, natural_rt, atol=1e-4)


def test_registry_lookup():
    """SubclassRegistryMixin finds DiagMVN and FullMVN by name."""
    from jaxfads.base import Approx

    assert Approx.get_subclass("DiagMVN") is DiagMVN
    assert Approx.get_subclass("FullMVN") is FullMVN
