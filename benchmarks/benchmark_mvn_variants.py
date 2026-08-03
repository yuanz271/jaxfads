"""Benchmark MVN rank variants on the VDP synthetic task.

This script runs a small controlled benchmark comparing MVN(rank=0),
MVN(rank=r), and MVN(rank=dim) and writes raw + aggregated results to
JSON/CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import operator
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import OmegaConf

from jaxfads import XFADS, configure_logging
from jaxfads.observations import GLM  # noqa: F401 (register GLM)
from jaxfads.training import (
    EpochHandler,
    GaussianObservationMstep,
    MVNNoiseMstep,
    train,
    train_test_split,
)

# Import helpers from the VDP example (sibling directory).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from vdp_example import evaluate, simulate_vdp


def _build_data(*, n_trials: int, n_steps: int, dt: float, mu: float, obs_dim: int):
    key, k_lat, k_C, k_b, k_y = jr.split(jr.key(0), 5)
    latent = simulate_vdp(k_lat, n_trials=n_trials, n_steps=n_steps, dt=dt, mu=mu)

    c_true = 0.7 * jr.normal(k_C, (obs_dim, latent.shape[-1]))
    b_true = 0.1 * jr.normal(k_b, (obs_dim,))
    sigma_obs = 0.3
    obs = (
        latent @ c_true.T
        + b_true
        + sigma_obs * jr.normal(k_y, (n_trials, n_steps, obs_dim))
    )

    times = jnp.broadcast_to(jnp.arange(n_steps), (n_trials, n_steps))
    controls = jnp.zeros((n_trials, n_steps, 0))
    covariates = jnp.zeros((n_trials, n_steps, 0))
    return (times, obs, controls, covariates), latent, obs, c_true, b_true, sigma_obs


def _variant_rows(rank_list: list[int]):
    # MVN defaults to use_sigma_points=True; pin False so this rank
    # comparison isn't silently confounded by also switching propagation
    # method (that's a different question, covered by
    # benchmarks/benchmark_highd_oscillator.py).
    rows = [
        dict(name="DiagMVN", approx="MVN", approx_kwargs={"rank": 0, "use_sigma_points": False}),
        dict(name="FullMVN", approx="MVN", approx_kwargs={"use_sigma_points": False}),
    ]
    for r in rank_list:
        rows.append(
            dict(
                name=f"LoRaMVN-r{r}",
                approx="MVN",
                approx_kwargs={"rank": r, "use_sigma_points": False},
            )
        )
    return rows


def _mean_std(rows: list[dict], key: str):
    vals = np.asarray([r[key] for r in rows], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--obs-dim", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--max-epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--lora-ranks", type=str, default="1")
    parser.add_argument("--out-dir", type=str, default="benchmarks/results/vdp_smoke")
    parser.add_argument("--q-mstep", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    print("JAX devices:", jax.devices())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    lora_ranks = [int(s) for s in args.lora_ranks.split(",") if s.strip()]
    if any(rank < 0 or rank > 2 for rank in lora_ranks):
        parser.error("--lora-ranks values must satisfy 0 <= rank <= 2")

    data, latent, observations, c_true, b_true, sigma_obs = _build_data(
        n_trials=args.n_trials,
        n_steps=args.n_steps,
        dt=args.dt,
        mu=args.mu,
        obs_dim=args.obs_dim,
    )

    xlim, vlim, grid = (-3.0, 3.0), (-3.0, 3.0), 25

    base_conf = dict(
        mode="smooth",
        observation_dim=args.obs_dim,
        state_dim=2,
        dynamics="Functional",
        integrator="RK4",
        seed=0,
        n_steps=args.n_steps,
        dropout=0.0,
        mc_size=4,
        dyn_conf=dict(
            input_dim=0,
            context_dim=0,
            fn_path="vdp_example:vdp_dynamics",
            fn_kwargs={"mu": args.mu},
            dt=args.dt,
        ),
        enc_conf=dict(width=32, depth=2, dropout=None),
        obs_conf=dict(
            model="GLM",
            likelihood="Gaussian",
            cov=[float(sigma_obs**2)] * args.obs_dim,
            norm_readout=False,
            dropout=0.0,
            readout_init_conf=dict(obs_noise_var=float(sigma_obs**2)),
        ),
    )

    trainer_conf = dict(
        seed=0,
        learning_rate=1e-3,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
    )

    train_data, valid_data = train_test_split(
        data, rng=np.random.default_rng(0), test_size=args.batch_size
    )

    rows: list[dict] = []
    variants = _variant_rows(lora_ranks)

    for variant in variants:
        for seed in seeds:
            conf = OmegaConf.create(
                {
                    **base_conf,
                    "seed": seed,
                    "approx": variant["approx"],
                    "approx_kwargs": variant["approx_kwargs"],
                }
            )
            model = XFADS(conf, jr.key(seed)).initialize(*train_data)

            t0 = time.perf_counter()
            handler = EpochHandler(valid_data=valid_data)
            train(
                model,
                train_data,
                conf=OmegaConf.create({**trainer_conf, "seed": seed}),
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
            dt_train = time.perf_counter() - t0

            _, eval_key = jr.split(jr.key(seed))
            metrics = evaluate(
                variant["name"],
                trained,
                latent,
                observations,
                data,
                c_true,
                b_true,
                key=eval_key,
                approx=trained.approx,
                mu=args.mu,
                xlim=xlim,
                vlim=vlim,
                grid=grid,
                align_flow=False,
            )

            approx = trained.approx
            rows.append(
                dict(
                    variant=variant["name"],
                    seed=seed,
                    train_seconds=float(dt_train),
                    natural_size=int(approx.param_size()),
                    encoder_free_size=int(approx.free_size()),
                    post_rmse=float(metrics["post_rmse"]),
                    c_rmse=float(metrics["C_rmse"]),
                    b_rmse=float(metrics["b_rmse"]),
                    obs_rmse=float(metrics["obs_rmse"]),
                    flow_nrmse=float(metrics["flow_nrmse"]),
                    flow_angle=float(metrics["flow_angle"]),
                )
            )

    # aggregate
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    summary = []
    for variant, rs in grouped.items():
        summary.append(
            dict(
                variant=variant,
                n_runs=len(rs),
                natural_size=int(rs[0]["natural_size"]),
                encoder_free_size=int(rs[0]["encoder_free_size"]),
                train_seconds_mean=_mean_std(rs, "train_seconds")[0],
                train_seconds_std=_mean_std(rs, "train_seconds")[1],
                post_rmse_mean=_mean_std(rs, "post_rmse")[0],
                post_rmse_std=_mean_std(rs, "post_rmse")[1],
                obs_rmse_mean=_mean_std(rs, "obs_rmse")[0],
                obs_rmse_std=_mean_std(rs, "obs_rmse")[1],
                flow_nrmse_mean=_mean_std(rs, "flow_nrmse")[0],
                flow_nrmse_std=_mean_std(rs, "flow_nrmse")[1],
            )
        )

    summary.sort(key=operator.itemgetter("variant"))

    raw_path = out_dir / "mvn_benchmark_raw.json"
    summary_path = out_dir / "mvn_benchmark_summary.json"
    csv_path = out_dir / "mvn_benchmark_summary.csv"

    raw_path.write_text(json.dumps(rows, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nBenchmark summary:")
    for s in summary:
        print(
            f"{s['variant']:<12} "
            f"time={s['train_seconds_mean']:.2f}±{s['train_seconds_std']:.2f}s "
            f"post_rmse={s['post_rmse_mean']:.4f}±{s['post_rmse_std']:.4f} "
            f"obs_rmse={s['obs_rmse_mean']:.4f}±{s['obs_rmse_std']:.4f} "
            f"free={s['encoder_free_size']} nat={s['natural_size']}"
        )

    print(f"\nWrote: {raw_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
