from dataclasses import dataclass
from enum import auto, StrEnum
from functools import partial
from typing import Type

import jax
from jax import numpy as jnp, random as jrandom
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray

from .dynamics import sample_expected_moment


class Mode(StrEnum):
    PSEUDO = auto()
    BIFILTER = auto()


@dataclass
class Hyperparam:
    approx: Type
    state_dim: int
    input_dim: int
    observation_dim: int
    # covariate_dim: int
    mc_size: int
    fb_penalty: float
    noise_penalty: float
    mode: str


def filter(
    key: PRNGKeyArray,
    t: Array,
    alpha: Array,
    u: Array,
    c: Array,
    model,
) -> tuple[Array, Array, Array]:
    """
    :param alpha:
    """
    approx = model.hyperparam.approx
    nature_p_1 = (
        model.prior_natural()
    )  # TODO: where should prior belongs, approx or dynamics?

    expected_moment_forward = partial(
        sample_expected_moment,
        f=model.forward,
        noise=model.forward,
        approx=approx,
        mc_size=model.hyperparam.mc_size,
    )

    nature_f_1 = nature_p_1 + alpha[0]

    def ff(carry, obs, expected_moment):
        key, nature_tm1 = carry
        key, subkey = jrandom.split(key)
        a_t, u_tm1, c_tm1 = obs
        moment_tm1 = approx.natural_to_moment(nature_tm1)
        moment_p_t = expected_moment(subkey, moment_tm1, u_tm1, c_tm1)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_t = nature_p_t + a_t
        return (key, nature_t), (moment_p_t, nature_p_t, nature_t)

    key, subkey = jrandom.split(key)
    _, (moment_p, _, nature_f) = scan(
        partial(ff, expected_moment=expected_moment_forward),
        init=(subkey, nature_f_1),
        xs=(alpha[1:], u[:-1], c[:-1]),  # t = 2 ... T+1
    )
    nature_f = jnp.vstack((nature_f_1, nature_f))  # 1...T

    moment_f = jax.vmap(approx.natural_to_moment)(nature_f)
    moment_p = jnp.vstack(
        (approx.natural_to_moment(nature_f_1), moment_p)
    )  # prediction of t=1 is the prior

    return nature_f, moment_f, moment_p


def bismooth(
    key: PRNGKeyArray,
    t: Array,
    alpha: Array,
    u: Array,
    c: Array,
    model,
) -> tuple[Array, Array, Array]:
    """
    Bidirectional filtering
    Parameterize inverse dynamics
    q(z[t]|y[1:T]) = q(z[y]|y[1:t])q(z[t]|y[t+1:T])/p(z[t])
    P[t] = Pf[t] + Pb[t] - P0
    Pm = Pfmf + Pbmb
    See Dowling23 Eq.(21-23)

    :param alpha: information update
    """
    hyperparam = model.hyperparam
    approx = hyperparam.approx
    nature_prior = model.prior_natural()

    natural_to_moment = jax.vmap(approx.natural_to_moment)
    expected_moment_forward = partial(
        sample_expected_moment,
        f=model.forward,
        noise=model.forward,
        approx=approx,
        mc_size=hyperparam.mc_size,
    )
    expected_moment_backward = partial(
        sample_expected_moment,
        f=model.backward,
        noise=model.backward,
        approx=approx,
        mc_size=hyperparam.mc_size,
    )

    nature_f_1 = nature_prior + alpha[0]

    def ff(carry, obs, expected_moment):
        key, nature_f_tm1 = carry
        key_tp1, key_t = jrandom.split(key, 2)
        update_obs_t, u, c = obs
        moment_f_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = expected_moment(key_t, moment_f_tm1, u, c)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_f_t = nature_p_t + update_obs_t
        return (key_tp1, nature_f_t), (moment_p_t, nature_p_t, nature_f_t)

    # Forward
    key, forward_key = jrandom.split(key, 2)
    _, (_, _, nature_f) = scan(
        partial(ff, expected_moment=expected_moment_forward),
        init=(forward_key, nature_f_1),
        xs=(alpha[1:], u[:-1], c[:-1]),  # t = 2 ... T+1
    )
    nature_f = jnp.vstack((nature_f_1, nature_f))  # 1...T

    ## Backward
    key, subkey = jrandom.split(key, 2)
    (_, nature_b_Tp1), _ = ff(
        (subkey, nature_f[-1]),
        (jnp.zeros_like(nature_prior), u[-1], c[-1]),
        expected_moment_forward,
    )

    key, backward_key = jrandom.split(key, 2)
    _, (_, nature_p_b, _) = scan(
        partial(ff, expected_moment=expected_moment_backward),
        init=(backward_key, nature_b_Tp1),
        xs=(alpha, u, c),
        reverse=True,
    )

    nature_s = nature_f + nature_p_b - jnp.expand_dims(nature_prior, axis=0)
    moment_s = natural_to_moment(nature_s)

    # expectation should be under smoothing distribution
    keys = jrandom.split(key, jnp.size(moment_s, 0))
    moment_p = jax.vmap(expected_moment_forward)(keys, moment_s, u)
    moment_p = jnp.vstack((moment_s[0], moment_p[:-1]))

    return nature_s, moment_s, moment_p
