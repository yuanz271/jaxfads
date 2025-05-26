# from jax import numpy as jnp, random as jrnd, random as jrandom
# from equinox import nn as enn
# import chex

# from xfads.distributions import MVN, DiagMVN
# from xfads.vi import DiagMVNLik, elbo


# def test_poisson(spec):
#     key = jrnd.key(0)
#     state_dim = spec['state_dim']
#     neural_dim = spec['neural_dim']
#     approx = MVN

#     y = jrnd.poisson(key, jnp.ones(neural_dim))
#     chex.assert_shape(y, (neural_dim,))
#     readout = enn.Linear(state_dim, neural_dim, key=key)
#     lik = PoissonLik(readout)

#     m = jnp.ones(state_dim)
#     cov = jnp.eye(state_dim)
#     moment = approx.canon_to_moment(m, cov)
#     chex.assert_shape(moment, (state_dim * state_dim + state_dim,))

#     covariate_predict = jnp.zeros_like(y)

#     ell = lik.eloglik(key, moment, covariate_predict, y, approx, mc_size=10)
#     chex.assert_shape(ell, y.shape)
#     chex.assert_tree_all_finite(ell)


# def test_elbo(spec):
#     key = jrnd.key(0)

#     state_dim = spec['state_dim']
#     neural_dim = spec['neural_dim']

#     m1 = jnp.ones(state_dim)
#     cov1 = jnp.eye(state_dim)
#     m2 = jnp.zeros(state_dim)
#     cov2 = jnp.eye(state_dim)

#     moment1 = MVN.canon_to_moment(m1, cov1)
#     moment2 = MVN.canon_to_moment(m2, cov2)

#     y = jrnd.poisson(key, jnp.ones(neural_dim))
#     covariate_predict = jnp.ones(neural_dim)
#     readout = enn.Linear(state_dim, neural_dim, key=key)
#     lik = PoissonLik(readout)

#     val = elbo(key, moment1, moment2, covariate_predict, y, lik.eloglik, MVN, mc_size=10)
#     chex.assert_tree_all_finite(val)


# def test_batch_elbo(spec, capsys):
#     key = jrandom.key(0)
#     key, rkey, ykey, ykey, ukey, skey, pkey = jrandom.split(key, 7)

#     N: int = 10
#     T: int = 100

#     linear_readout = enn.Linear(spec['state_dim'], spec['neural_dim'], key=rkey)
#     likelihood = DiagMVNLik(cov=jnp.ones(spec['neural_dim']), readout=linear_readout)

#     elbo = make_batch_elbo(likelihood.eloglik, DiagMVN, mc_size=10)
#     ys = jrandom.normal(ykey, (N, T, spec['neural_dim']))
#     covariate_predict = jrandom.normal(ykey, (N, T, spec['neural_dim']))
#     moment_s = jrandom.uniform(skey, (N, T, spec['state_dim'] * 2))
#     moment_p = jrandom.uniform(pkey, (N, T, spec['state_dim'] * 2))

#     with capsys.disabled():
#         elbo(key, moment_s, moment_p, covariate_predict, ys)
