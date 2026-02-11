from jax import numpy as jnp
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads.distributions import DiagMVN


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

    # nat2 should be bounded (floor of 1e-6 → nat2 ≈ -5e5, not -4e6)
    _, nat2 = jnp.split(natural, 2)
    assert float(jnp.max(jnp.abs(nat2)).item()) < 1e6, (
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


def test_lowrankcov(capsys):
    # MultivariateNormalDiagPlusLowRankCovariance is DIFFERENT from
    # MultivariateNormalDiagPlusLowRank

    loc = jnp.ones(2)
    cov_diag = jnp.ones(2)
    cov_lr = jnp.ones((2, 1))

    _ = tfp.MultivariateNormalDiagPlusLowRankCovariance(loc, cov_diag, cov_lr)
