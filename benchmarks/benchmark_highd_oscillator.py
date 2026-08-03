"""High-D oscillator benchmark for MVN rank variants.

This benchmark scales latent dimension and compares:
- MVN(dim, rank=0)        — diagonal
- MVN(dim, rank=r)        — low-rank + diagonal
- MVN(dim, rank=dim)      — full rank

Outputs raw/summary JSON + CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import OmegaConf

from jaxfads import XFADS, configure_logging
from jaxfads.training import GaussianObservationMstep, MVNNoiseMstep
from jaxfads.observations import GLM  # noqa: F401 (register GLM)
from jaxfads.training import EpochHandler, train, train_test_split


def oscillator_bank_dynamics(z, u, c, *, omega=1.2, gamma=0.15, beta=0.02):
    """Continuous-time oscillator bank RHS in R^(2K).

    For each pair (x, v):
        dx/dt = v
        dv/dt = -omega^2 x - gamma v - beta x^3
    """
    del u, c
    pairs = z.reshape(-1, 2)
    x = pairs[:, 0]
    v = pairs[:, 1]

    dx = v
    dv = -(omega**2) * x - gamma * v - beta * (x**3)
    return jnp.stack([dx, dv], axis=-1).reshape(z.shape)


def rk4_step(z, dt):
    k1 = oscillator_bank_dynamics(z, None, None)
    k2 = oscillator_bank_dynamics(z + 0.5 * dt * k1, None, None)
    k3 = oscillator_bank_dynamics(z + 0.5 * dt * k2, None, None)
    k4 = oscillator_bank_dynamics(z + dt * k3, None, None)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_oscillator_bank(
    key, *, n_trials: int, n_steps: int, state_dim: int, dt: float
):
    if state_dim % 2 != 0:
        raise ValueError("state_dim must be even for oscillator-bank construction")

    # Randomized initial phases/radii per 2D block.
    n_blocks = state_dim // 2
    phi = jr.uniform(key, (n_trials, n_blocks), minval=0.0, maxval=2.0 * jnp.pi)
    rad = 1.5 + 0.25 * jr.normal(jr.fold_in(key, 1), (n_trials, n_blocks))
    x0 = rad * jnp.cos(phi)
    v0 = rad * jnp.sin(phi)
    z0 = jnp.stack([x0, v0], axis=-1).reshape(n_trials, state_dim)

    def scan_fn(z, _):
        z_next = rk4_step(z, dt)
        return z_next, z_next

    _, zs = jax.vmap(lambda z: jax.lax.scan(scan_fn, z, None, length=n_steps))(z0)
    return zs


def affine_fit(z_true: jax.Array, z_inf: jax.Array):
    mu_i = z_inf.mean(0)
    mu_t = z_true.mean(0)
    a_t, *_ = jnp.linalg.lstsq(z_inf - mu_i, z_true - mu_t)
    a = a_t.T
    d = mu_t - a @ mu_i
    return a, d


def evaluate_model(trained, data, latent_true):
    t, y, u, c = data
    _, moments, _ = trained(t, y, u, c, key=jr.key(999))
    means, _covs = jax.vmap(jax.vmap(trained.approx.unpack))(moments)

    d = latent_true.shape[-1]
    a, b = affine_fit(latent_true.reshape(-1, d), means.reshape(-1, d))
    aligned = means @ a.T + b
    post_rmse = float(jnp.sqrt(jnp.mean((aligned - latent_true) ** 2)))

    # Observation reconstruction RMSE in model observation space.
    c_l = trained.observation.readout.weight
    layer = trained.observation.readout.layer
    b_l = layer.layer.bias if hasattr(layer, "layer") else layer.bias
    y_hat = means @ c_l.T + b_l
    obs_rmse = float(jnp.sqrt(jnp.mean((y_hat - y) ** 2)))

    return dict(post_rmse=post_rmse, obs_rmse=obs_rmse)


def mean_std(rows: list[dict], key: str):
    vals = np.asarray([r[key] for r in rows], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=str, default="8,16,32")
    p.add_argument("--lora-ranks", type=str, default="1,2,4")
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--n-trials", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--max-epoch", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=str, default="benchmarks/results/highd_oscillator")
    p.add_argument("--q-mstep", action="store_true")
    args = p.parse_args()

    configure_logging("INFO")
    print("JAX devices:", jax.devices())

    dims = [int(s) for s in args.dims.split(",") if s.strip()]
    ranks = [int(s) for s in args.lora_ranks.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []

    for dim in dims:
        data_key = jr.key(1000 + dim)
        latent = simulate_oscillator_bank(
            data_key,
            n_trials=args.n_trials,
            n_steps=args.n_steps,
            state_dim=dim,
            dt=args.dt,
        )

        k_c, k_b, k_y = jr.split(jr.fold_in(data_key, 1), 3)
        c_true = 0.6 * jr.normal(k_c, (args.obs_dim, dim))
        b_true = 0.1 * jr.normal(k_b, (args.obs_dim,))
        sigma_obs = 0.3
        obs = (
            latent @ c_true.T
            + b_true
            + sigma_obs * jr.normal(k_y, (args.n_trials, args.n_steps, args.obs_dim))
        )

        times = jnp.broadcast_to(
            jnp.arange(args.n_steps), (args.n_trials, args.n_steps)
        )
        controls = jnp.zeros((args.n_trials, args.n_steps, 0))
        covariates = jnp.zeros((args.n_trials, args.n_steps, 0))
        data = (times, obs, controls, covariates)

        # MVN defaults to use_sigma_points=True; every "plain MC" variant
        # below pins use_sigma_points=False explicitly so this comparison
        # stays meaningful regardless of MVN's own default.
        variants = (
            [
                ("DiagMVN", "MVN", {"rank": 0, "use_sigma_points": False}, None),
                ("FullMVN", "MVN", {"use_sigma_points": False}, None),
            ]
            + [
                (f"LoRaMVN-r{r}", "MVN", {"rank": r, "use_sigma_points": False}, None)
                for r in ranks
                if r <= dim
            ]
            + [
                # Deliberately below the mc_size >= dim+1 safety threshold
                # (transition_points' rank-deficiency warning) -- an
                # explicit comparison point, not relying on the sweep's
                # own default (now fixed to a safe dim+1) to demonstrate
                # the pathology by accident.
                ("FullMVN-mc4-unsafe", "MVN", {"use_sigma_points": False}, 4),
                # Deterministic unscented-transform sigma points instead of
                # MC; mc_size is ignored (always 2*dim+1 points).
                ("FullMVN-UT", "MVN", {"use_sigma_points": True}, None),
            ]
            + [
                # use_sigma_points only affects transition_points, which
                # branches on approx._layout.is_diag (True for rank=0,
                # False for any rank>0) -- never on the specific rank
                # value, so it composes with LoRa's low-rank encoder
                # parameterization exactly like plain MC does.
                (f"LoRaMVN-r{r}-UT", "MVN", {"rank": r, "use_sigma_points": True}, None)
                for r in ranks
                if r <= dim
            ]
        )

        for variant_name, approx_name, approx_kwargs, mc_size_override in variants:
            for seed in seeds:
                conf = OmegaConf.create(
                    {
                        "mode": "smooth",
                        "observation_dim": args.obs_dim,
                        "state_dim": dim,
                        "dynamics": "Functional",
                        "integrator": "RK4",
                        "approx": approx_name,
                        "approx_kwargs": approx_kwargs,
                        "seed": seed,
                        "n_steps": args.n_steps,
                        "fb_penalty": 0.0,
                        "noise_penalty": 0.01,
                        "dropout": 0.0,
                        # Safe margin against the transition_points
                        # rank-deficiency warning (mc_size <= state_dim):
                        # the MC spread term needs mc_size >= dim + 1 to be
                        # full rank, scaled per dim rather than one fixed
                        # value across the sweep. mc_size_override lets
                        # specific variants (the deliberate "unsafe" MC
                        # comparison point) opt out of this safety margin.
                        "mc_size": (
                            mc_size_override
                            if mc_size_override is not None
                            else dim + 1
                        ),
                        "dyn_conf": {
                            "input_dim": 0,
                            "context_dim": 0,
                            "fn_path": "benchmark_highd_oscillator:oscillator_bank_dynamics",
                            "fn_kwargs": {},
                            "dt": args.dt,
                        },
                        "enc_conf": {
                            "width": 64,
                            "depth": 2,
                            "dropout": None,
                        },
                        "obs_conf": {
                            "model": "GLM",
                            "likelihood": "Gaussian",
                            "cov": [float(sigma_obs**2)] * args.obs_dim,
                            "norm_readout": False,
                            "dropout": 0.0,
                            "readout_init_conf": {"obs_noise_var": float(sigma_obs**2)},
                        },
                    }
                )

                trainer_conf = OmegaConf.create(
                    {
                        "seed": seed,
                        "learning_rate": 1e-3,
                        "max_epoch": args.max_epoch,
                        "batch_size": args.batch_size,
                    }
                )

                train_data, valid_data = train_test_split(
                    data, rng=np.random.default_rng(0), test_size=args.batch_size
                )
                model = XFADS(conf, jr.key(seed)).initialize(*train_data)

                t0 = time.perf_counter()
                handler = EpochHandler(valid_data=valid_data)
                train(
                    model,
                    train_data,
                    conf=trainer_conf,
                    on_epoch_end=handler,
                    post_optimizer_transforms=(
                        GaussianObservationMstep(),
                        *(
                            (MVNNoiseMstep(q_scale=0.1),)
                            if args.q_mstep
                            else ()
                        ),
                    ),
                )
                trained = handler.best_model
                train_s = time.perf_counter() - t0

                metrics = evaluate_model(trained, data, latent)
                approx = trained.approx

                raw_rows.append(
                    {
                        "state_dim": dim,
                        "variant": variant_name,
                        "seed": seed,
                        "train_seconds": float(train_s),
                        "natural_size": int(approx.param_size()),
                        "encoder_free_size": int(approx.free_size()),
                        "post_rmse": float(metrics["post_rmse"]),
                        "obs_rmse": float(metrics["obs_rmse"]),
                    }
                )

    # aggregate
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in raw_rows:
        groups.setdefault((row["state_dim"], row["variant"]), []).append(row)

    summary_rows: list[dict] = []
    for (dim, variant), rows in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        summary_rows.append(
            {
                "state_dim": dim,
                "variant": variant,
                "n_runs": len(rows),
                "natural_size": rows[0]["natural_size"],
                "encoder_free_size": rows[0]["encoder_free_size"],
                "train_seconds_mean": mean_std(rows, "train_seconds")[0],
                "train_seconds_std": mean_std(rows, "train_seconds")[1],
                "post_rmse_mean": mean_std(rows, "post_rmse")[0],
                "post_rmse_std": mean_std(rows, "post_rmse")[1],
                "obs_rmse_mean": mean_std(rows, "obs_rmse")[0],
                "obs_rmse_std": mean_std(rows, "obs_rmse")[1],
            }
        )

    raw_path = out_dir / "highd_oscillator_raw.json"
    summary_path = out_dir / "highd_oscillator_summary.json"
    csv_path = out_dir / "highd_oscillator_summary.csv"

    raw_path.write_text(json.dumps(raw_rows, indent=2))
    summary_path.write_text(json.dumps(summary_rows, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSummary (mean±std):")
    for s in summary_rows:
        print(
            f"D={s['state_dim']:<3} {s['variant']:<12} "
            f"time={s['train_seconds_mean']:.2f}±{s['train_seconds_std']:.2f}s "
            f"post={s['post_rmse_mean']:.4f}±{s['post_rmse_std']:.4f} "
            f"obs={s['obs_rmse_mean']:.4f}±{s['obs_rmse_std']:.4f} "
            f"free={s['encoder_free_size']} nat={s['natural_size']}"
        )

    print(f"\nWrote: {raw_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
