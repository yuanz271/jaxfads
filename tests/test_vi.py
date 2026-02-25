import chex
from jax import numpy as jnp
from jax import random as jrnd

from jaxfads.vi import elbo


def _eloglik_stub(key, t, mp, y, approx, mc_size):
    del key, t, mp, approx, mc_size
    return jnp.sum(y)


def test_elbo_matches_manual(diag):
    mean = jnp.array([0.2, -0.1])
    cov = jnp.diag(jnp.array([1.0, 2.0]))
    mp = diag.pack(mean, cov)
    mp_p = diag.pack(jnp.zeros(2), jnp.eye(2))
    y = jnp.array([1.0, 2.0])

    expected = _eloglik_stub(None, None, None, y, None, None) - diag.kl(
        mp, mp_p
    )
    value = elbo(
        jrnd.key(0),
        jnp.array(0),
        mp,
        mp_p,
        y,
        _eloglik_stub,
        diag,
        mc_size=1,
    )

    chex.assert_trees_all_close(value, expected)
