"""Latent-z case for the post-optimizer transform design's Q update -- a real XFADS
model doing posterior inference, not the known-z clean baseline
(benchmarks/q_update_known_z.py).

Simulates synthetic Lorenz trajectories with a KNOWN Q_true injected into
the latent dynamics and a linear-Gaussian observation model, trains XFADS
normally (current behavior: Q gradient-trained jointly via the ELBO),
and reports:
- Q's trajectory over training (diagnostic).
- The learned dynamics network's flow-field accuracy against the TRUE
  Lorenz RHS, after Procrustes-aligning the inferred latent space to the
  true one (the PRIMARY metric, per the doc's success-metric framing) --
  evaluated at the actual (aligned) data points, not a regular grid
  (avoids curse-of-dimensionality issues a 2D-style grid would have in 3D).

Run with:
    JAX_PLATFORMS=cpu uv run python benchmarks/q_update_lorenz_latent.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import equinox as eqx
import equinox.nn as enn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _utils import DT, rk4_step
from omegaconf import OmegaConf

from jaxfads import XFADS, configure_logging
from jaxfads.base import Dynamics
from jaxfads.nn import make_mlp
from jaxfads.observations import GLM  # noqa: F401 -- registers GLM
from jaxfads.training import EpochHandler, train, train_test_split

# ---------------------------------------------------------------------------
# Lorenz simulation with known process noise, plus a linear-Gaussian
# observation model
# ---------------------------------------------------------------------------


def simulate_lorenz_trajectories(key, *, n_trials, n_steps, dt, q_true, burn_in=200):
    key_init, key_noise = jr.split(key)
    z0 = 1.0 + jr.normal(key_init, (n_trials, 3))

    def scan_fn(z, key_t):
        z_next = rk4_step(z, dt) + jnp.sqrt(q_true) * jr.normal(key_t, z.shape)
        return z_next, z_next

    def one_trial(z0_trial, key_trial):
        keys = jr.split(key_trial, burn_in + n_steps)
        z_burned, _ = jax.lax.scan(scan_fn, z0_trial, keys[:burn_in])
        _, traj = jax.lax.scan(scan_fn, z_burned, keys[burn_in:])
        return traj

    keys = jr.split(key_noise, n_trials)
    return jax.vmap(one_trial)(z0, keys)  # (n_trials, n_steps, 3)


def build_dataset(key, *, n_trials, n_steps, dt, q_true, obs_dim, sigma_obs):
    key_traj, key_c, key_b, key_obs = jr.split(key, 4)
    latent = simulate_lorenz_trajectories(
        key_traj, n_trials=n_trials, n_steps=n_steps, dt=dt, q_true=q_true
    )
    c_true = 0.5 * jr.normal(key_c, (obs_dim, 3))
    b_true = 0.1 * jr.normal(key_b, (obs_dim,))
    obs = (
        latent @ c_true.T
        + b_true
        + sigma_obs * jr.normal(key_obs, (n_trials, n_steps, obs_dim))
    )
    times = jnp.broadcast_to(jnp.arange(n_steps), (n_trials, n_steps))
    controls = jnp.zeros((n_trials, n_steps, 0))
    contexts = jnp.zeros((n_trials, n_steps, 0))
    return (times, obs, controls, contexts), latent, c_true, b_true


# ---------------------------------------------------------------------------
# Trainable MLP dynamics (mirrors examples/vdp_example.py's MLPDynamics)
# ---------------------------------------------------------------------------


class MLPDynamics(Dynamics):
    """Trainable MLP dynamics with Euler integration; the MLP learns the
    continuous-time derivative f(z), eval() returns that derivative."""

    net: enn.Sequential

    def __init__(self, conf, key):
        self.conf = conf
        self.net = make_mlp(
            conf.state_dim, conf.state_dim, width=conf.width, depth=conf.depth, key=key
        )

    def eval(self, z, u, c, *, key=None):
        del u, c, key
        return self.net(z)


# ---------------------------------------------------------------------------
# Procrustes alignment + flow-field accuracy against the true Lorenz RHS
# ---------------------------------------------------------------------------


@dataclass
class AffineAlignment:
    A: jax.Array
    d: jax.Array


def procrustes_affine(z_true, z_inferred) -> AffineAlignment:
    mu_inf, mu_true = z_inferred.mean(0), z_true.mean(0)
    A_T, *_ = jnp.linalg.lstsq(z_inferred - mu_inf, z_true - mu_true)
    A = A_T.T
    return AffineAlignment(A=A, d=mu_true - A @ mu_inf)


def align(aff: AffineAlignment, z):
    return z @ aff.A.T + aff.d


def flow_field_rmse(model, aff: AffineAlignment, eval_points_true_coords):
    """RMSE between the model's learned Euler-integration one-step-ahead
    map (mapped back into true coordinates via the alignment) and the true
    Lorenz RK4 one-step-ahead map, evaluated at real (aligned) data points
    -- not a regular grid (avoids curse-of-dimensionality in 3D)."""
    A_inv = jnp.linalg.inv(aff.A)
    pts_model_coords = (eval_points_true_coords - aff.d) @ A_inv.T

    def model_rhs(z):
        return model.dynamics.eval(z, jnp.zeros((0,)), jnp.zeros((0,)), key=None)

    rhs_model_coords = jax.vmap(model_rhs)(pts_model_coords)
    # dynamics.eval returns the continuous-time derivative; approximate the
    # one-step map consistently with how it's used inside the model (Euler,
    # matching MLPDynamics + the built-in Euler integrator convention).
    dt = DT
    pred_next_true_coords = eval_points_true_coords + dt * (rhs_model_coords @ aff.A.T)
    true_next = jax.vmap(lambda z: rk4_step(z, dt))(eval_points_true_coords)
    return float(jnp.sqrt(jnp.mean((pred_next_true_coords - true_next) ** 2)))


# ---------------------------------------------------------------------------
# Historical transition-stat experiment, applied against a
# real XFADS model's own smoothed moments -- decoupled from
# Approx.transition_points entirely.
# ---------------------------------------------------------------------------


def mstep_transition_diag(model, data, approx, *, floor):
    """v1 M-step statistic (no cross-covariance term), computed from this
    model's own smoothed moments. Returns a per-dimension diagonal Q
    estimate, floored for numerical safety only (per the doc's design --
    not a meaningful Bayesian prior)."""
    t, y, u, c = data
    _, moment, _ = model(t, y, u, c, key=jr.key(0))  # smoothed moments
    moment_tm1 = moment[:, :-1, :].reshape(-1, moment.shape[-1])
    moment_t = moment[:, 1:, :].reshape(-1, moment.shape[-1])

    def f(z):
        return model.transition(z, jnp.zeros(0), jnp.zeros(0), key=None)

    def per_pair_stat(m_tm1, m_t):
        mean_tm1, cov_tm1 = approx.unpack(m_tm1)
        mean_t, cov_t = approx.unpack(m_t)
        r = mean_t - f(mean_tm1)
        J = jax.jacrev(f)(mean_tm1)
        return jnp.outer(r, r) + cov_t + J @ cov_tm1 @ J.T

    stats = jax.vmap(per_pair_stat)(moment_tm1, moment_t)  # (n_pairs, D, D)
    raw_stat_diag = jnp.diag(jnp.mean(stats, axis=0))
    return jnp.maximum(raw_stat_diag, floor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--q-true", type=float, default=0.01)
    p.add_argument("--obs-dim", type=int, default=10)
    p.add_argument("--sigma-obs", type=float, default=0.3)
    p.add_argument("--max-epoch", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    configure_logging()
    print(f"JAX devices: {jax.devices()}")
    key = jr.key(args.seed)
    key_data, key_model = jr.split(key)

    data, latent, c_true, b_true = build_dataset(
        key_data,
        n_trials=args.n_trials,
        n_steps=args.n_steps,
        dt=DT,
        q_true=args.q_true,
        obs_dim=args.obs_dim,
        sigma_obs=args.sigma_obs,
    )
    train_data, valid_data = train_test_split(
        data, rng=np.random.default_rng(0), test_size=args.batch_size
    )

    conf = OmegaConf.create({
        "mode": "smooth",
        "observation_dim": args.obs_dim,
        "state_dim": 3,
        "dynamics": "MLPDynamics",
        "integrator": "Euler",
        "approx": "MVN",
        "approx_kwargs": {},  # MVN defaults to use_sigma_points=True
        "mc_size": 4,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "dropout": 0.0,
        "dyn_conf": {
            "width": 64,
            "depth": 2,
            "input_dim": 0,
            "context_dim": 0,
            "dt": DT,
        },
        "enc_conf": {"width": 64, "depth": 2, "dropout": None},
        "obs_conf": {
            "model": "GLM",
            "likelihood": "Gaussian",
            "cov": [float(args.sigma_obs**2)] * args.obs_dim,
            "norm_readout": False,
            "dropout": 0.0,
        },
    })

    def run(name, *, freeze_q, use_mstep):
        model = XFADS(conf, key_model).initialize(*train_data)
        approx = model.approx
        q_trace = []

        def on_epoch_end(m, info):
            _, Q = approx.unpack(approx.canon_to_moment(approx.free_to_canon(m.noise)))
            q_trace.append(np.asarray(jnp.diag(Q)))
            return False

        handler = EpochHandler(valid_data=valid_data)

        def combined_cb(m, info):
            on_epoch_end(m, info)
            return handler(m, info)

        trainer_conf = OmegaConf.create({
            "seed": args.seed,
            "learning_rate": 1e-3,
            "max_epoch": args.max_epoch,
            "batch_size": args.batch_size,
        })
        if freeze_q:
            trainer_conf.freeze_paths = ["noise"]

        # Q frozen means it stays at its CONSTANT initial value throughout
        # training -- f's gradient is then scaled by a fixed 1/Q, not a
        # shrinking one, so this isolates "does f learn better without a
        # moving, damping Q" from the accumulated epoch-local trainer path
        # used by the automatic MAP-Q comparison below.
        trained = train(model, train_data, conf=trainer_conf, on_epoch_end=combined_cb)
        if use_mstep:
            q_diag_final = mstep_transition_diag(
                trained, train_data, approx, floor=1e-6
            )
            trained = eqx.tree_at(
                lambda mm: mm.noise, trained, approx.free_from_kw(scale=q_diag_final)
            )

        print(f"\n=== {name} ===")
        print(
            "Q diag trace (per epoch, from training-time free param, "
            "NOT the post-hoc mstep estimate):"
        )
        for i, q in enumerate(q_trace):
            print(f"  epoch {i}: {q}")
        _, Q_final = approx.unpack(
            approx.canon_to_moment(approx.free_to_canon(trained.noise))
        )
        print(
            f"Q_final diag (applied to model): {jnp.diag(Q_final)} (true={args.q_true})"
        )

        t, y, u, c = data
        _, means, _ = trained(t, y, u, c, key=jr.key(123))
        means, _ = jax.vmap(jax.vmap(approx.unpack))(means)
        aff = procrustes_affine(latent.reshape(-1, 3), means.reshape(-1, 3))

        eval_pts = latent.reshape(-1, 3)[
            :: max(1, latent.reshape(-1, 3).shape[0] // 2000)
        ]
        rmse = flow_field_rmse(trained, aff, eval_pts)
        print(
            f"flow-field RMSE vs true Lorenz one-step map (aligned, "
            f"n_eval_pts={eval_pts.shape[0]}): {rmse:.5f}"
        )

        aligned_means = align(aff, means)
        post_rmse = float(jnp.sqrt(jnp.mean((aligned_means - latent) ** 2)))
        print(f"posterior-mean RMSE (aligned) vs true latent: {post_rmse:.5f}")
        return dict(
            name=name,
            q_final=np.asarray(jnp.diag(Q_final)),
            flow_rmse=rmse,
            post_rmse=post_rmse,
        )

    def run_alternating_em(name, *, n_rounds, epochs_per_round, prior, prior_dof_frac):
        """Q stays IN the ELBO/KL loss throughout (never frozen -- unlike
        condition B), but is updated between rounds via a MAP-shrunk
        mstep_transition_stat estimate instead of free gradient descent.
        Matches Approach C's design in benchmarks/q_update_known_z.py."""
        model = XFADS(conf, key_model).initialize(*train_data)
        approx = model.approx
        n_pairs = train_data[0].shape[0] * (train_data[0].shape[1] - 1)
        prior_dof = prior_dof_frac * n_pairs
        q_trace = []

        for round_idx in range(n_rounds):
            trainer_conf = OmegaConf.create({
                "seed": args.seed,
                "learning_rate": 1e-3,
                "max_epoch": epochs_per_round,
                "batch_size": args.batch_size,
                "freeze_paths": ["noise"],
            })
            model = train(model, train_data, conf=trainer_conf)
            raw_stat = mstep_transition_diag(model, train_data, approx, floor=1e-6)
            q_shrunk = (n_pairs * raw_stat + prior_dof * prior) / (n_pairs + prior_dof)
            model = eqx.tree_at(
                lambda m: m.noise, model, approx.free_from_kw(scale=q_shrunk)
            )
            _, Q = approx.unpack(
                approx.canon_to_moment(approx.free_to_canon(model.noise))
            )
            q_trace.append(np.asarray(jnp.diag(Q)))

        print(f"\n=== {name} ===")
        print("Q diag trace (per round):")
        for i, q in enumerate(q_trace):
            print(f"  round {i}: {q}")
        print(f"Q_final diag: {q_trace[-1]} (true={args.q_true})")

        t, y, u, c = data
        _, means, _ = model(t, y, u, c, key=jr.key(123))
        means, _ = jax.vmap(jax.vmap(approx.unpack))(means)
        aff = procrustes_affine(latent.reshape(-1, 3), means.reshape(-1, 3))

        eval_pts = latent.reshape(-1, 3)[
            :: max(1, latent.reshape(-1, 3).shape[0] // 2000)
        ]
        rmse = flow_field_rmse(model, aff, eval_pts)
        print(
            f"flow-field RMSE vs true Lorenz one-step map (aligned, "
            f"n_eval_pts={eval_pts.shape[0]}): {rmse:.5f}"
        )

        aligned_means = align(aff, means)
        post_rmse = float(jnp.sqrt(jnp.mean((aligned_means - latent) ** 2)))
        print(f"posterior-mean RMSE (aligned) vs true latent: {post_rmse:.5f}")
        return dict(name=name, q_final=q_trace[-1], flow_rmse=rmse, post_rmse=post_rmse)

    def run_train_integrated(name, *, q_scale):
        """Fully automatic epoch-level MAP-Q path through train()."""
        model = XFADS(conf, key_model).initialize(*train_data)
        approx = model.approx

        trainer_conf = OmegaConf.create({
            "seed": args.seed,
            "learning_rate": 1e-3,
            "max_epoch": args.max_epoch,
            "batch_size": args.batch_size,
            "post_optimizer_transforms": [
                {"name": "gaussian_observation"},
                {"name": "mvn_noise", "q_scale": q_scale, "q_prior_fraction": 0.1},
            ],
        })
        model = train(
            model,
            train_data,
            conf=trainer_conf,
            post_optimizer_transforms=(),
        )

        _, Q_final = approx.unpack(
            approx.canon_to_moment(approx.free_to_canon(model.noise))
        )
        print(f"\n=== {name} ===")
        print(f"Q_final diag: {jnp.diag(Q_final)} (true={args.q_true})")

        t, y, u, c = data
        _, means, _ = model(t, y, u, c, key=jr.key(123))
        means, _ = jax.vmap(jax.vmap(approx.unpack))(means)
        aff = procrustes_affine(latent.reshape(-1, 3), means.reshape(-1, 3))

        eval_pts = latent.reshape(-1, 3)[
            :: max(1, latent.reshape(-1, 3).shape[0] // 2000)
        ]
        rmse = flow_field_rmse(model, aff, eval_pts)
        print(
            f"flow-field RMSE vs true Lorenz one-step map (aligned, "
            f"n_eval_pts={eval_pts.shape[0]}): {rmse:.5f}"
        )

        aligned_means = align(aff, means)
        post_rmse = float(jnp.sqrt(jnp.mean((aligned_means - latent) ** 2)))
        print(f"posterior-mean RMSE (aligned) vs true latent: {post_rmse:.5f}")
        return dict(
            name=name,
            q_final=np.asarray(jnp.diag(Q_final)),
            flow_rmse=rmse,
            post_rmse=post_rmse,
        )

    result_a = run(
        "A: joint gradient training (baseline)", freeze_q=False, use_mstep=False
    )
    result_b = run(
        "B: Q frozen + mstep_transition_stat (decoupled)", freeze_q=True, use_mstep=True
    )
    result_c = run_alternating_em(
        "C: alternating EM (Q in loss, MAP-shrunk rounds) -- prototype math",
        n_rounds=5,
        epochs_per_round=max(1, args.max_epoch // 5),
        prior=1.0,
        prior_dof_frac=0.1,
    )
    result_d = run_train_integrated(
        "D: automatic accumulated epoch-local train() MAP-Q",
        q_scale=1.0,
    )

    print("\n=== Summary ===")
    header = (
        f"{'metric':<20}{'A (joint)':<20}{'B (mstep)':<20}"
        f"{'C (prototype alt. EM)':<25}{'D (epoch MAP-Q)':<22}"
    )
    print(header)
    print(
        f"{'Q_final mean':<20}{result_a['q_final'].mean():<20.5f}{result_b['q_final'].mean():<20.5f}"
        f"{result_c['q_final'].mean():<25.5f}{result_d['q_final'].mean():<22.5f}"
    )
    print(
        f"{'flow RMSE':<20}{result_a['flow_rmse']:<20.5f}{result_b['flow_rmse']:<20.5f}"
        f"{result_c['flow_rmse']:<25.5f}{result_d['flow_rmse']:<22.5f}"
    )
    print(
        f"{'post RMSE':<20}{result_a['post_rmse']:<20.5f}{result_b['post_rmse']:<20.5f}"
        f"{result_c['post_rmse']:<25.5f}{result_d['post_rmse']:<22.5f}"
    )


if __name__ == "__main__":
    main()
