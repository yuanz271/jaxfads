import chex
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

from jaxfads.base import Approx
from jaxfads.encoders import AlphaEncoder, BetaEncoder


def _alpha_conf(spec, *, dropout=None):
    return OmegaConf.create(
        dict(
            observation_dim=spec["observation_dim"],
            state_dim=spec["state_dim"],
            approx=spec["approx"],
            approx_kwargs=spec["approx_kwargs"],
            width=spec["width"],
            depth=spec["depth"],
            dropout=dropout,
        )
    )


def _beta_conf(spec, *, dropout=None):
    return OmegaConf.create(
        dict(
            state_dim=spec["state_dim"],
            approx=spec["approx"],
            approx_kwargs=spec["approx_kwargs"],
            width=spec["width"],
            dropout=dropout,
        )
    )


def _make_approx(conf):
    """Instantiate an Approx from config."""
    cls = Approx.get_subclass(conf.approx)
    return cls(**conf.approx_kwargs)


def test_alpha_encoder_shape(spec):
    key = jrnd.key(0)
    conf = _alpha_conf(spec, dropout=None)
    encoder = AlphaEncoder(conf, key)
    y = jnp.ones((spec["observation_dim"],))

    approx = _make_approx(conf)
    out = encoder(y)
    chex.assert_shape(out, (approx.param_size(spec["state_dim"]),))


def test_beta_encoder_shape(spec):
    key = jrnd.key(1)
    conf = _beta_conf(spec, dropout=None)
    encoder = BetaEncoder(conf, key)

    approx = _make_approx(conf)
    param_size = approx.param_size(spec["state_dim"])
    a = jnp.ones((5, param_size))
    out = encoder(a)
    chex.assert_shape(out, (5, param_size))
    chex.assert_tree_all_finite(out)


def test_beta_encoder_dropout_path(spec):
    key = jrnd.key(2)
    conf = _beta_conf(spec, dropout=0.1)
    encoder = BetaEncoder(conf, key)

    approx = _make_approx(conf)
    param_size = approx.param_size(spec["state_dim"])
    a = jnp.ones((6, param_size))
    out = encoder(a, key=jrnd.key(3))
    chex.assert_shape(out, (6, param_size))
    chex.assert_tree_all_finite(out)
