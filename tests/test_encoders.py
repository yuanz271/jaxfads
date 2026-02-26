import chex
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

from jaxfads.encoders import AlphaEncoder, BetaEncoder
from conftest import make_approx


def _enc_conf(spec, *, dropout=None):
    # Encoders are Approx-agnostic; they only need the flat parameter length.
    approx = make_approx(spec["state_dim"], spec["approx"], **spec["approx_kwargs"])
    param_size = approx.param_size()

    return OmegaConf.create(
        dict(
            observation_dim=spec["observation_dim"],
            param_size=param_size,
            width=spec["width"],
            depth=spec["depth"],
            dropout=dropout,
        )
    )


def test_alpha_encoder_shape(spec):
    key = jrnd.key(0)
    conf = _enc_conf(spec)
    encoder = AlphaEncoder(conf, key)
    y = jnp.ones((spec["observation_dim"],))

    out = encoder(y)
    chex.assert_shape(out, (int(conf.param_size),))


def test_beta_encoder_shape(spec):
    key = jrnd.key(1)
    conf = _enc_conf(spec)
    encoder = BetaEncoder(conf, key)

    a = jnp.ones((5, int(conf.param_size)))
    out = encoder(a)
    chex.assert_shape(out, (5, int(conf.param_size)))
    chex.assert_tree_all_finite(out)


def test_beta_encoder_dropout_path(spec):
    key = jrnd.key(2)
    conf = _enc_conf(spec, dropout=0.1)
    encoder = BetaEncoder(conf, key)

    a = jnp.ones((6, int(conf.param_size)))
    out = encoder(a, key=jrnd.key(3))
    chex.assert_shape(out, (6, int(conf.param_size)))
    chex.assert_tree_all_finite(out)
