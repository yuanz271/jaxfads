import chex
from jax import numpy as jnp
from jax import random as jrnd

from jaxfads.distributions import MVN
from jaxfads.vi import elbo

approx = MVN(rank=0)


def _eloglik_stub(key, t, moment, y, approx, mc_size):
    del key, t, moment, approx, mc_size
    return jnp.sum(y)


def test_elbo_matches_manual():
    mean = jnp.array([0.2, -0.1])
    cov = jnp.array([1.0, 2.0])
    moment = approx.canon_to_moment(mean, cov)
    moment_p = approx.canon_to_moment(jnp.zeros(2), jnp.ones(2))
    y = jnp.array([1.0, 2.0])

    expected = _eloglik_stub(None, None, None, y, None, None) - approx.kl(
        moment, moment_p
    )
    value = elbo(
        jrnd.key(0),
        jnp.array(0),
        moment,
        moment_p,
        y,
        _eloglik_stub,
        approx,
        mc_size=1,
    )

    chex.assert_trees_all_close(value, expected)
