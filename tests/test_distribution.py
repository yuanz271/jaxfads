from jax import numpy as jnp
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from xfads.distributions import DiagMVN


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


def test_lowrankcov(capsys):
    # MultivariateNormalDiagPlusLowRankCovariance is DIFFERENT from
    # MultivariateNormalDiagPlusLowRank

    loc = jnp.ones(2)
    cov_diag = jnp.ones(2)
    cov_lr = jnp.ones((2, 1))

    mvn = tfp.MultivariateNormalDiagPlusLowRankCovariance(loc, cov_diag, cov_lr)

    with capsys.disabled():
        print(mvn.covariance())
