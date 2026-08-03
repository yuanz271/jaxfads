"""Known-z clean baseline for the post-optimizer transform design's Q update.

Tests the joint-MLE-degeneracy claim directly, with z fully known (no
latent inference, no XFADS model at all -- isolates this from the
latent-z entanglement risk, per the doc's Step 1):

- Approach A (naive, current XFADS behavior): jointly gradient-optimize a
  flexible dynamics network f and a free process-noise covariance Q via
  the joint NLL log N(z_t; f(z_{t-1}), Q). Prediction: f overfits (residual
  on TRAIN pairs shrinks below what held-out pairs support), Q chases that
  residual toward collapse, held-out one-step accuracy suffers -- the
  Heywood-style degeneracy this whole plan is about, in its simplest form.
- Approach B (decoupled): fit f via plain MSE only (Q never enters this
  loss at all), then separately estimate Q from the residual with a
  numerical-safety floor only (the historical v1 design: no
  cross-covariance term needed here since z is fully known, so
  Cov(z_{t-1})=Cov(z_t)=0 and the M-step statistic reduces to the plain
  mean squared residual).

Primary metric (per the doc's success-metric framing): f's accuracy
against the TRUE Lorenz vector field on a held-out grid, not Q's value.
Q's recovered value is reported as a diagnostic only.

Run with:
    JAX_PLATFORMS=cpu uv run python benchmarks/mstep_known_z_baseline.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx

from jaxfads.nn import make_mlp


# ---------------------------------------------------------------------------
# Lorenz system + noisy discrete-time simulation
# ---------------------------------------------------------------------------


def lorenz_rhs(state: jax.Array, *, sigma: float = 10.0, rho: float = 28.0,
                beta: float = 8.0 / 3.0) -> jax.Array:
    x, y, z = state[0], state[1], state[2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.stack([dx, dy, dz])


def rk4_step(state: jax.Array, dt: float) -> jax.Array:
    k1 = lorenz_rhs(state)
    k2 = lorenz_rhs(state + 0.5 * dt * k1)
    k3 = lorenz_rhs(state + 0.5 * dt * k2)
    k4 = lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_lorenz_pairs(key, *, n_trials, n_steps, dt, q_true, burn_in=200):
    """Simulate noisy Lorenz trajectories; return flattened (z_prev, z_next)
    pairs with the true process noise Q injected at every discrete step:
    z_next = rk4_step(z_prev, dt) + N(0, q_true)."""
    key_init, key_noise = jr.split(key)
    z0 = 1.0 + jr.normal(key_init, (n_trials, 3))

    def scan_fn(z, key_t):
        z_det = rk4_step(z, dt)
        z_next = z_det + jnp.sqrt(q_true) * jr.normal(key_t, z.shape)
        return z_next, z_next

    def one_trial(z0_trial, key_trial):
        keys = jr.split(key_trial, burn_in + n_steps)
        z_burned, _ = jax.lax.scan(scan_fn, z0_trial, keys[:burn_in])
        _, traj = jax.lax.scan(scan_fn, z_burned, keys[burn_in:])
        return traj

    keys = jr.split(key_noise, n_trials)
    trajs = jax.vmap(one_trial)(z0, keys)  # (n_trials, n_steps, 3)
    z_prev = trajs[:, :-1, :].reshape(-1, 3)
    z_next = trajs[:, 1:, :].reshape(-1, 3)
    return z_prev, z_next


# ---------------------------------------------------------------------------
# Approach A: joint MLE of f and Q
# ---------------------------------------------------------------------------


def make_f(key, width=64, depth=2):
    return make_mlp(3, 3, width, depth, key=key, final_bias=True)


def joint_nll_loss(f, log_q, z_prev, z_next):
    q = jax.nn.softplus(log_q) + 1e-6
    pred = jax.vmap(f)(z_prev)
    resid = z_next - pred
    # diagonal Gaussian NLL per-dim, summed
    nll = 0.5 * jnp.log(q) + 0.5 * (resid**2) / q
    return jnp.mean(jnp.sum(nll, axis=-1))


def train_joint(key, z_prev, z_next, *, epochs, lr, batch_size, log_q0):
    f_key, _ = jr.split(key)
    f = make_f(f_key)
    log_q = jnp.full((3,), log_q0)

    params = (f, log_q)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(params, eqx.is_array))

    n = z_prev.shape[0]
    n_batches = n // batch_size

    @eqx.filter_jit
    def step(params, opt_state, zp, zn):
        def loss_fn(p):
            f, log_q = p
            return joint_nll_loss(f, log_q, zp, zn)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(params, eqx.is_array))
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss

    q_trace = []
    perm_key = key
    for epoch in range(epochs):
        perm_key, sub = jr.split(perm_key)
        perm = jr.permutation(sub, n)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            params, opt_state, loss = step(params, opt_state, z_prev[idx], z_next[idx])
        f, log_q = params
        q_trace.append(jax.nn.softplus(log_q) + 1e-6)

    f, log_q = params
    q_final = jax.nn.softplus(log_q) + 1e-6
    return f, q_final, jnp.stack(q_trace)


# ---------------------------------------------------------------------------
# Approach B: decoupled -- plain MSE for f, then M-step-style Q with a floor
# ---------------------------------------------------------------------------


def mse_loss(f, z_prev, z_next):
    pred = jax.vmap(f)(z_prev)
    return jnp.mean(jnp.sum((z_next - pred) ** 2, axis=-1))


def train_decoupled(key, z_prev, z_next, *, epochs, lr, batch_size):
    f = make_f(key)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(f, eqx.is_array))

    n = z_prev.shape[0]
    n_batches = n // batch_size

    @eqx.filter_jit
    def step(f, opt_state, zp, zn):
        loss, grads = eqx.filter_value_and_grad(mse_loss)(f, zp, zn)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(f, eqx.is_array))
        f = eqx.apply_updates(f, updates)
        return f, opt_state, loss

    perm_key = key
    for epoch in range(epochs):
        perm_key, sub = jr.split(perm_key)
        perm = jr.permutation(sub, n)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            f, opt_state, loss = step(f, opt_state, z_prev[idx], z_next[idx])
    return f


def mstep_q_known_z(f, z_prev, z_next, *, floor):
    """v1 M-step statistic with z fully known: Cov(z_prev)=Cov(z_next)=0,
    so the formula in the Q update reduces to the plain
    mean squared residual (no Jacobian/covariance-propagation term needed
    -- that term only matters under posterior uncertainty, absent here)."""
    pred = jax.vmap(f)(z_prev)
    resid = z_next - pred
    raw_stat = jnp.mean(resid**2, axis=0)
    return jnp.maximum(raw_stat, floor)


def mstep_shrink(raw_stat, prior, prior_dof, n):
    """MAP shrinkage toward an explicit, informative prior (unlike the
    doc's final design, which uses a floor-only, non-informative prior --
    this is deliberately the OTHER version, to test whether genuine
    shrinkage changes the picture vs. a floor-only clip)."""
    return (n * raw_stat + prior_dof * prior) / (n + prior_dof)


# ---------------------------------------------------------------------------
# Approach C: alternating EM -- Q stays IN the joint-NLL loss (so f's
# gradient is still scaled by 1/Q, same as Approach A), but Q itself is
# updated via MAP-shrunk M-step rounds instead of free gradient descent.
# Tests both "does shrinkage help" and (indirectly) the "free Q benefits
# SGD as an extra parameter" conjecture, by keeping Q in the loss while
# controlling its value differently.
# ---------------------------------------------------------------------------


def train_alternating_em(key, z_prev, z_next, *, n_rounds, epochs_per_round, lr,
                          batch_size, q_init, prior, prior_dof_frac):
    f = make_f(key)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(f, eqx.is_array))
    q = jnp.full((3,), q_init)

    n = z_prev.shape[0]
    n_batches = n // batch_size
    prior_dof = prior_dof_frac * n

    @eqx.filter_jit
    def step(f, opt_state, q, zp, zn):
        def loss_fn(f):
            pred = jax.vmap(f)(zp)
            resid = zn - pred
            return jnp.mean(jnp.sum(0.5 * jnp.log(q) + 0.5 * resid**2 / q, axis=-1))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(f)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(f, eqx.is_array))
        f = eqx.apply_updates(f, updates)
        return f, opt_state, loss

    q_trace = [q]
    perm_key = key
    for round_idx in range(n_rounds):
        for epoch in range(epochs_per_round):
            perm_key, sub = jr.split(perm_key)
            perm = jr.permutation(sub, n)
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                f, opt_state, loss = step(f, opt_state, q, z_prev[idx], z_next[idx])
        raw_stat = mstep_q_known_z(f, z_prev, z_next, floor=1e-6)
        q = mstep_shrink(raw_stat, prior, prior_dof, n)
        q_trace.append(q)

    return f, q, jnp.stack(q_trace)


# ---------------------------------------------------------------------------
# Evaluation: f's accuracy against the TRUE Lorenz vector field (primary
# metric, per the doc's success-metric framing) -- not Q's value.
# ---------------------------------------------------------------------------


def vector_field_error(f, eval_points):
    """RMSE between f(z) and rk4_step(z, dt)'s deterministic prediction
    (the true one-step map, no noise) evaluated ON the actual attractor --
    using real (held-out) trajectory points, not uniform-random points over
    a bounding box. Lorenz's attractor occupies a narrow, specific region of
    state space; evaluating off-attractor is pure extrapolation and doesn't
    discriminate between a well- and poorly-fit f."""
    true_next = jax.vmap(lambda z: rk4_step(z, dt=DT))(eval_points)
    pred_next = jax.vmap(f)(eval_points)
    return float(jnp.sqrt(jnp.mean((true_next - pred_next) ** 2)))


DT = 0.01


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--q-true", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-q0", type=float, default=0.0,
                   help="Initial log_q for Approach A (0.0 -> Q~0.69; "
                        "-4.6 -> Q~0.01, matching q-true default).")
    args = p.parse_args()

    print(f"JAX devices: {jax.devices()}")
    key = jr.key(args.seed)
    key_data, key_a, key_b = jr.split(key, 3)

    z_prev, z_next = simulate_lorenz_pairs(
        key_data, n_trials=args.n_trials, n_steps=args.n_steps, dt=DT,
        q_true=args.q_true,
    )
    n = z_prev.shape[0]
    n_train = int(0.8 * n)
    perm = jr.permutation(jr.key(999), n)
    train_idx, held_idx = perm[:n_train], perm[n_train:]
    zp_train, zn_train = z_prev[train_idx], z_next[train_idx]
    zp_held, zn_held = z_prev[held_idx], z_next[held_idx]

    print(f"n_pairs={n} (train={n_train}, held_out={n - n_train}), Q_true={args.q_true}")

    print("\n=== Approach A: joint MLE of f and Q ===")
    f_a, q_a, q_trace = train_joint(
        key_a, zp_train, zn_train, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, log_q0=args.log_q0,
    )
    held_mse_a = float(mse_loss(f_a, zp_held, zn_held))
    train_mse_a = float(mse_loss(f_a, zp_train, zn_train))
    vf_err_a = vector_field_error(f_a, zp_held)
    print(f"Q trace (per-epoch, mean over dims): {[float(q.mean()) for q in q_trace]}")
    print(f"Q_final: {q_a}, mean={float(q_a.mean()):.5f} (true={args.q_true})")
    print(f"train MSE={train_mse_a:.5f}  held-out MSE={held_mse_a:.5f}  "
          f"(overfit gap={held_mse_a - train_mse_a:.5f})")
    print(f"vector-field RMSE vs true Lorenz RHS: {vf_err_a:.5f}")

    print("\n=== Approach B: decoupled (plain MSE for f, then M-step Q) ===")
    f_b = train_decoupled(
        key_b, zp_train, zn_train, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size,
    )
    held_mse_b = float(mse_loss(f_b, zp_held, zn_held))
    train_mse_b = float(mse_loss(f_b, zp_train, zn_train))
    vf_err_b = vector_field_error(f_b, zp_held)
    q_b = mstep_q_known_z(f_b, zp_train, zn_train, floor=1e-6)
    print(f"Q_estimate (M-step, held-out-independent floor only): {q_b}, "
          f"mean={float(q_b.mean()):.5f} (true={args.q_true})")
    print(f"train MSE={train_mse_b:.5f}  held-out MSE={held_mse_b:.5f}  "
          f"(overfit gap={held_mse_b - train_mse_b:.5f})")
    print(f"vector-field RMSE vs true Lorenz RHS: {vf_err_b:.5f}")

    print("\n=== Approach C: alternating EM (Q in the loss, MAP-shrunk rounds) ===")
    n_rounds = 5
    epochs_per_round = max(1, args.epochs // n_rounds)
    f_c, q_c, q_trace_c = train_alternating_em(
        key_b, zp_train, zn_train, n_rounds=n_rounds,
        epochs_per_round=epochs_per_round, lr=args.lr, batch_size=args.batch_size,
        q_init=1.0, prior=1.0, prior_dof_frac=0.1,
    )
    held_mse_c = float(mse_loss(f_c, zp_held, zn_held))
    train_mse_c = float(mse_loss(f_c, zp_train, zn_train))
    vf_err_c = vector_field_error(f_c, zp_held)
    print(f"Q trace (per-round, mean over dims): {[float(q.mean()) for q in q_trace_c]}")
    print(f"Q_final: {q_c}, mean={float(q_c.mean()):.5f} (true={args.q_true})")
    print(f"train MSE={train_mse_c:.5f}  held-out MSE={held_mse_c:.5f}  "
          f"(overfit gap={held_mse_c - train_mse_c:.5f})")
    print(f"vector-field RMSE vs true Lorenz RHS: {vf_err_c:.5f}")

    print("\n=== Summary ===")
    print(f"{'metric':<30}{'A (joint MLE)':<20}{'B (decoupled)':<20}{'C (alt. EM+shrink)':<20}")
    print(f"{'vector-field RMSE':<30}{vf_err_a:<20.5f}{vf_err_b:<20.5f}{vf_err_c:<20.5f}")
    print(f"{'held-out MSE':<30}{held_mse_a:<20.5f}{held_mse_b:<20.5f}{held_mse_c:<20.5f}")
    print(f"{'overfit gap':<30}{held_mse_a - train_mse_a:<20.5f}{held_mse_b - train_mse_b:<20.5f}{held_mse_c - train_mse_c:<20.5f}")
    print(f"{'Q mean':<30}{float(q_a.mean()):<20.5f}{float(q_b.mean()):<20.5f}{float(q_c.mean()):<20.5f}")


if __name__ == "__main__":
    main()
