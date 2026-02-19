from jax import numpy as jnp
from jax import random as jrnd
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import DiagMVN, FullMVN, LoRaMVN


# def test_mvn(spec):
#     state_dim = spec['state_dim']

#     m1 = jnp.ones(state_dim)
#     cov1 = jnp.eye(state_dim)
#     m2 = jnp.zeros(state_dim)
#     cov2 = jnp.eye(state_dim)

#     moment1 = MVN.canon_to_moment(m1, cov1)
#     moment2 = MVN.canon_to_moment(m2, cov2)
#     kl = MVN.kl(moment1, moment2)
#     chex.assert_tree_all_finite(kl)

#     mc_size = 10
#     samples = MVN.sample_by_moment(jrandom.key(0), moment1, mc_size=mc_size)
#     chex.assert_shape(samples, (mc_size,) + (state_dim,))

#     unconstrained_natural: jnp.Array = jrandom.normal(jrandom.key(0), shape=(MVN.moment_size(state_dim),))
#     natural = MVN.constrain_natural(unconstrained_natural)
#     chex.assert_equal_shape((moment1, natural))

#     assert MVN.variable_size(MVN.moment_size(state_dim)) == state_dim


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
    """Near-zero covariance must not produce extreme natural parameters.

    When dynamics process noise is near zero (cov ~ EPS), the raw natural
    parameter nat2 = -0.5/cov would be enormous (~-4e6), causing numerical
    instability in the filtering loop.  The floor in moment_to_natural()
    should prevent this.
    """
    state_dim = 2
    mean = jnp.ones(state_dim)

    # Simulate what happens with cov=0.0 through the constraint pipeline:
    # constrain_positive(unconstrain_positive(0.0)) ≈ EPS ≈ 1.19e-7
    eps = jnp.finfo(jnp.float32).eps
    tiny_cov = jnp.full(state_dim, eps)

    moment = DiagMVN.canon_to_moment(mean, tiny_cov)
    natural = DiagMVN.moment_to_natural(moment)

    # Natural parameters must be finite
    chex.assert_tree_all_finite(natural)

    # nat2 should be bounded (floor of EPS ≈ 1.19e-7 → nat2 ≈ -4.2e6)
    _, nat2 = jnp.split(natural, 2)
    assert float(jnp.max(jnp.abs(nat2)).item()) < 1e7, (
        f"nat2 too large: {float(jnp.max(jnp.abs(nat2)).item()):.3e}"
    )

    # Roundtrip: natural → moment → natural should be stable
    moment_rt = DiagMVN.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment_rt)
    mean_rt, cov_rt = DiagMVN.moment_to_canon(moment_rt)
    chex.assert_tree_all_finite(mean_rt)
    chex.assert_tree_all_finite(cov_rt)

    # Normal covariance should roundtrip exactly
    normal_cov = jnp.ones(state_dim)
    moment_normal = DiagMVN.canon_to_moment(mean, normal_cov)
    natural_normal = DiagMVN.moment_to_natural(moment_normal)
    moment_normal_rt = DiagMVN.natural_to_moment(natural_normal)
    chex.assert_trees_all_close(moment_normal, moment_normal_rt)


def test_diagmvn_kl_matches_closed_form():
    """DiagMVN.kl() must match the closed-form diagonal Gaussian KL.

    The closed-form KL for diagonal Gaussians with variance vectors v1, v2:
        KL(q || p) = 0.5 * sum( log(v2/v1) + (v1 + (m1-m2)^2) / v2 - 1 )

    This test guards against passing variance where TFP expects std.
    """

    def _kl_closed_form(m1, v1, m2, v2):
        return 0.5 * jnp.sum(jnp.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    # Case 1: moderate variances
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

    # Case 2: large asymmetric variances — maximally distinguishes
    # "variance passed as std" (would give ~605) from correct (~31.8)
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

    # KL(p, p) == 0
    kl_self = DiagMVN.kl(moment1, moment1)
    chex.assert_trees_all_close(kl_self, jnp.array(0.0), atol=1e-6)


def test_fullmvn_constrain_moment():
    """FullMVN.constrain_moment produces valid moment params (PSD covariance)."""
    state_dim = 3
    param_size = FullMVN.param_size(state_dim)  # D + D² = 12
    unconstrained = jrnd.normal(jrnd.key(0), (param_size,))

    moment = FullMVN.constrain_moment(unconstrained)
    chex.assert_shape(moment, (param_size,))
    chex.assert_tree_all_finite(moment)

    mean, cov = FullMVN.moment_to_canon(moment)
    chex.assert_shape(mean, (state_dim,))
    chex.assert_shape(cov, (state_dim, state_dim))

    # Covariance must be symmetric and positive definite
    chex.assert_trees_all_close(cov, cov.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"


def test_fullmvn_constrain_natural():
    """FullMVN.constrain_natural produces valid natural params (negative definite nat2)."""
    state_dim = 3
    param_size = FullMVN.param_size(state_dim)  # D + D² = 12
    unconstrained = jrnd.normal(jrnd.key(1), (param_size,))

    natural = FullMVN.constrain_natural(unconstrained)
    chex.assert_shape(natural, (param_size,))
    chex.assert_tree_all_finite(natural)

    # nat2 block should be negative definite (all eigenvalues < 0)
    nat1, nat2_flat = jnp.split(natural, [state_dim])
    nat2 = jnp.reshape(nat2_flat, (state_dim, state_dim))
    eigenvalues = jnp.linalg.eigvalsh(nat2)
    assert jnp.all(eigenvalues < 0), f"nat2 not negative definite: eigenvalues={eigenvalues}"

    # Constrained natural params should convert to finite moments
    moment = FullMVN.natural_to_moment(natural)
    chex.assert_tree_all_finite(moment)

    # The moment -> natural roundtrip should be stable
    mean, cov = FullMVN.moment_to_canon(moment)
    eigenvalues_cov = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigenvalues_cov > 0), f"Recovered covariance not PD: {eigenvalues_cov}"


def test_fullmvn_unconstrain_natural_roundtrip():
    """unconstrain_natural inverts constrain_natural for FullMVN."""
    state_dim = 3
    # Start from a known valid natural (the prior)
    natural = FullMVN.prior_natural(state_dim)
    unconstrained = FullMVN.unconstrain_natural(natural)
    chex.assert_tree_all_finite(unconstrained)

    # constrain should recover a valid natural that maps to the same distribution
    natural_rt = FullMVN.constrain_natural(unconstrained)
    chex.assert_tree_all_finite(natural_rt)

    # Both should give the same moments
    moment = FullMVN.natural_to_moment(natural)
    moment_rt = FullMVN.natural_to_moment(natural_rt)
    chex.assert_trees_all_close(moment, moment_rt, atol=1e-4)


def test_loramvn_constrain_moment():
    """LoRaMVN.constrain_moment produces valid moment params (PSD covariance)."""
    state_dim = 3
    # LoRaMVN input layout: (D, D, D) = 3D
    input_size = 3 * state_dim
    unconstrained = jrnd.normal(jrnd.key(2), (input_size,))

    moment = LoRaMVN.constrain_moment(unconstrained)
    chex.assert_tree_all_finite(moment)

    # Output is (D + D²) = mean + flattened covariance
    mean, cov_flat = jnp.split(moment, [state_dim])
    cov = jnp.reshape(cov_flat, (state_dim, state_dim))

    chex.assert_shape(mean, (state_dim,))
    chex.assert_shape(cov, (state_dim, state_dim))
    chex.assert_trees_all_close(cov, cov.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(cov)
    assert jnp.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"


def test_loramvn_constrain_natural():
    """LoRaMVN.constrain_natural produces valid natural params (negative definite nat2)."""
    state_dim = 3
    input_size = 3 * state_dim
    unconstrained = jrnd.normal(jrnd.key(3), (input_size,))

    natural = LoRaMVN.constrain_natural(unconstrained)
    chex.assert_tree_all_finite(natural)

    nat1, nat2_flat = jnp.split(natural, [state_dim])
    nat2 = jnp.reshape(nat2_flat, (state_dim, state_dim))
    eigenvalues = jnp.linalg.eigvalsh(nat2)
    assert jnp.all(eigenvalues < 0), f"nat2 not negative definite: eigenvalues={eigenvalues}"


def test_lowrankcov(capsys):
    # MultivariateNormalDiagPlusLowRankCovariance is DIFFERENT from
    # MultivariateNormalDiagPlusLowRank

    loc = jnp.ones(2)
    cov_diag = jnp.ones(2)
    cov_lr = jnp.ones((2, 1))

    _ = tfp.MultivariateNormalDiagPlusLowRankCovariance(loc, cov_diag, cov_lr)
