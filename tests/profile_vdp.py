"""
Standalone profiling script for the VDP training pipeline.

Usage: uv run python tests/profile_vdp.py
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from omegaconf import OmegaConf

import sys
sys.path.insert(0, "examples")
from vdp_example import simulate_vdp  # noqa: E402

from jaxfads.smoother import XFADS  # noqa: E402
from jaxfads.trainer import batch_loss  # noqa: E402


def timer(label, fn, *args, repeats=5, **kwargs):
    """Time a JIT-compiled function, returning (result, mean_ms)."""
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn(*args, **kwargs)
        jax.block_until_ready(r)
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  {label:<45s}  {mean_ms:8.2f} ± {std_ms:5.2f} ms")
    return result, mean_ms


def make_conf(state_dim, obs_dim, T, sigma_obs, mu, dt, mc_size=1):
    """Build XFADS config matching vdp_example."""
    enc_conf = dict(
        observation_dim=obs_dim, state_dim=state_dim,
        approx="DiagMVN", width=32, depth=2, dropout=None,
    )
    obs_conf = dict(
        model="GLM", likelihood="Poisson",
        observation_dim=obs_dim, state_dim=state_dim,
        cov=[float(sigma_obs ** 2)] * obs_dim,
        norm_readout=False, dropout=0.0,
        readout_init_conf=dict(obs_noise_var=float(sigma_obs ** 2)),
    )
    dyn_conf = dict(
        state_dim=state_dim, input_dim=0, context_dim=0,
        mu=mu, dt=dt, cov=1.0,
    )
    return OmegaConf.create(dict(
        mode="pseudo", observation_dim=obs_dim, state_dim=state_dim,
        approx="DiagMVN", forward="VDPDynamics", seed=0, n_steps=T,
        fb_penalty=0.0, noise_penalty=0.01, dropout=0.0, mc_size=mc_size,
        enc_conf=enc_conf, obs_conf=obs_conf, dyn_conf=dyn_conf,
    ))


def make_data(N, T, obs_dim, state_dim, sigma_obs, mu, dt):
    """Generate synthetic VDP data."""
    key = jr.key(0)
    key, k_lat, k_C, k_b, k_y = jr.split(key, 5)

    latent = simulate_vdp(k_lat, n_trials=N, n_steps=T, dt=dt, mu=mu)
    C_true = 0.7 * jr.normal(k_C, (obs_dim, state_dim))
    b_true = 0.1 * jr.normal(k_b, (obs_dim,))
    observations = latent @ C_true.T + b_true + sigma_obs * jr.normal(k_y, (N, T, obs_dim))

    times = jnp.broadcast_to(jnp.arange(T), (N, T))
    controls = jnp.zeros((N, T, 0))
    covariates = jnp.zeros((N, T, 0))
    return (times, observations, controls, covariates)


def main():
    print(f"JAX devices: {jax.devices()}")

    # --- Setup ---
    N, T, dt, mu = 32, 400, 0.02, 2.0
    obs_dim, state_dim = 10, 2
    sigma_obs = 0.3

    batch = make_data(N, T, obs_dim, state_dim, sigma_obs, mu, dt)

    conf = make_conf(state_dim, obs_dim, T, sigma_obs, mu, dt, mc_size=1)
    model = XFADS(conf, key=jr.key(42))
    model = model.initialize(*batch)

    print(f"\nN={N}, T={T}, state_dim={state_dim}, obs_dim={obs_dim}, "
          f"mc_size={model.conf.mc_size}")
    print()

    key = jr.key(99)

    # ==========================================================
    # 1. Forward pass
    # ==========================================================
    print("=" * 65)
    print("1. Forward pass (inference)")
    print("=" * 65)

    @eqx.filter_jit
    def forward_only(model, batch, key):
        t, y, u, c = batch
        return model(t, y, u, c, key=key)

    key, fwd_key = jr.split(key)

    t0 = time.perf_counter()
    r = forward_only(model, batch, fwd_key)
    jax.block_until_ready(r)
    print(f"  {'JIT compile (forward)':<45s}  {(time.perf_counter() - t0)*1000:8.2f} ms")

    timer("forward (compiled)", forward_only, model, batch, fwd_key)
    print()

    # ==========================================================
    # 2. Gradient step
    # ==========================================================
    print("=" * 65)
    print("2. Gradient computation")
    print("=" * 65)

    @eqx.filter_jit
    def grad_step(model, batch, key):
        return eqx.filter_value_and_grad(batch_loss)(model, batch, key)

    key, gk = jr.split(key)

    t0 = time.perf_counter()
    loss, grads = grad_step(model, batch, gk)
    loss.block_until_ready()
    print(f"  {'JIT compile (grad)':<45s}  {(time.perf_counter() - t0)*1000:8.2f} ms")

    timer("grad step (compiled)", grad_step, model, batch, gk)
    print()

    # ==========================================================
    # 3. Full train step (grad + optimizer)
    # ==========================================================
    print("=" * 65)
    print("3. Full training step (grad + optimizer)")
    print("=" * 65)

    opt = optax.chain(optax.adam(1e-3), optax.clip_by_global_norm(1.0))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, batch, key, opt_state):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(model, batch, key)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    key, sk = jr.split(key)

    t0 = time.perf_counter()
    m2, os2, l2 = train_step(model, batch, sk, opt_state)
    l2.block_until_ready()
    print(f"  {'JIT compile (full step)':<45s}  {(time.perf_counter() - t0)*1000:8.2f} ms")

    timer("full train step (compiled)", train_step, model, batch, sk, opt_state)
    print()

    # ==========================================================
    # 4. Scaling: T (forward)
    # ==========================================================
    print("=" * 65)
    print("4. Scaling: forward time vs sequence length T")
    print("=" * 65)

    for T_test in [50, 100, 200, 400, 800]:
        batch_t = make_data(N, T_test, obs_dim, state_dim, sigma_obs, mu, dt)
        # Need model with matching T for initialization
        conf_t = make_conf(state_dim, obs_dim, T_test, sigma_obs, mu, dt)
        model_t = XFADS(conf_t, key=jr.key(42))
        model_t = model_t.initialize(*batch_t)

        @eqx.filter_jit
        def fwd_t(model, batch, key):
            t, y, u, c = batch
            return model(t, y, u, c, key=key)

        key, k = jr.split(key)
        timer(f"T={T_test:>4d}", fwd_t, model_t, batch_t, k, repeats=3)
    print()

    # ==========================================================
    # 5. Scaling: T (grad)
    # ==========================================================
    print("=" * 65)
    print("5. Scaling: grad time vs sequence length T")
    print("=" * 65)

    for T_test in [50, 100, 200, 400]:
        batch_t = make_data(N, T_test, obs_dim, state_dim, sigma_obs, mu, dt)
        conf_t = make_conf(state_dim, obs_dim, T_test, sigma_obs, mu, dt)
        model_t = XFADS(conf_t, key=jr.key(42))
        model_t = model_t.initialize(*batch_t)

        @eqx.filter_jit
        def grad_t(model, batch, key):
            return eqx.filter_value_and_grad(batch_loss)(model, batch, key)[0]

        key, k = jr.split(key)
        timer(f"T={T_test:>4d}", grad_t, model_t, batch_t, k, repeats=3)
    print()

    # ==========================================================
    # 6. Scaling: mc_size (grad)
    # ==========================================================
    print("=" * 65)
    print("6. Scaling: grad time vs mc_size")
    print("=" * 65)

    for mc in [1, 2, 4, 8]:
        conf_mc = make_conf(state_dim, obs_dim, T, sigma_obs, mu, dt, mc_size=mc)
        model_mc = XFADS(conf_mc, key=jr.key(42))
        model_mc = model_mc.initialize(*batch)

        @eqx.filter_jit
        def grad_mc(model, batch, key):
            return eqx.filter_value_and_grad(batch_loss)(model, batch, key)[0]

        key, gk = jr.split(key)
        timer(f"mc_size={mc}", grad_mc, model_mc, batch, gk, repeats=3)


if __name__ == "__main__":
    main()
