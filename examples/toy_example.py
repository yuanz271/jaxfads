"""Toy example: XFADS with synthetic Van der Pol oscillator data.

Demonstrates variational Bayesian state-space modeling on a 2D Van der Pol
latent with 10D Gaussian observations.  Uses a ``ToyDynamics`` module that
implements the *exact* Van der Pol RK4 step (no model mismatch) so that all
learning is concentrated in the encoder and observation readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from jaxfads import XFADS, configure_logging
from jaxfads.base import Dynamics
from jaxfads.dynamics import DiagGaussian
from jaxfads.trainer import train


# ---------------------------------------------------------------------------
# Van der Pol dynamics
# ---------------------------------------------------------------------------


def vdp_rhs(state: jax.Array, mu: float) -> jax.Array:
    """Right-hand side of the Van der Pol oscillator."""
    x, v = state[0], state[1]
    dx = v
    dv = mu * (1.0 - x * x) * v - x
    return jnp.stack([dx, dv], axis=0)


def rk4_step(state: jax.Array, dt: float, *, mu: float) -> jax.Array:
    """Single RK4 integration step for the Van der Pol oscillator."""
    k1 = vdp_rhs(state, mu)
    k2 = vdp_rhs(state + 0.5 * dt * k1, mu)
    k3 = vdp_rhs(state + 0.5 * dt * k2, mu)
    k4 = vdp_rhs(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def simulate_vdp(
    key: jax.Array,
    *,
    n_trials: int,
    n_steps: int,
    dt: float,
    mu: float,
    init_radius: float = 2.0,
    init_radius_jitter: float = 0.2,
    latent_noise: float = 0.0,
) -> jax.Array:
    """Generate ``n_trials`` Van der Pol trajectories via RK4.

    Returns
    -------
    jax.Array
        Latent states with shape ``(n_trials, n_steps, 2)``.
    """
    key, k_phi, k_r, k_eps = jr.split(key, 4)

    phi = jr.uniform(k_phi, (n_trials,), minval=0.0, maxval=2.0 * jnp.pi)
    r = init_radius + init_radius_jitter * jr.normal(k_r, (n_trials,))

    x0 = r * jnp.cos(phi)
    v0 = r * jnp.sin(phi)
    z0 = jnp.stack([x0, v0], axis=-1)

    eps = latent_noise * jr.normal(k_eps, (n_trials, n_steps, 2))

    def one_trial(z0_single, eps_single):
        def step(z, e):
            z_next = rk4_step(z, dt, mu=mu) + e
            return z_next, z_next

        _, zs = jax.lax.scan(step, z0_single, eps_single)
        return zs

    return jax.vmap(one_trial)(z0, eps)


# ---------------------------------------------------------------------------
# Flow-field metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowMetrics:
    nrmse: float
    mean_angle_rad: float


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    max_abs_err: float


def _max_abs(x) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(x))).item())


def _assert_close(a, b, *, atol=1e-6, rtol=1e-6, name="") -> CheckResult:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    err = _max_abs(a - b)
    ok = bool(jnp.allclose(a, b, atol=atol, rtol=rtol).item())
    if not ok:
        raise AssertionError(f"{name} failed: max_abs_err={err:.3e}")
    return CheckResult(name=name, ok=True, max_abs_err=err)


def _reconstruct_z0_and_eps(
    key,
    *,
    n_trials: int,
    n_steps: int,
    init_radius: float,
    init_radius_jitter: float,
    latent_noise: float,
) -> tuple[jax.Array, jax.Array]:
    key, k_phi, k_r, k_eps = jr.split(key, 4)
    phi = jr.uniform(k_phi, (n_trials,), minval=0.0, maxval=2.0 * jnp.pi)
    r = init_radius + init_radius_jitter * jr.normal(k_r, (n_trials,))
    x0 = r * jnp.cos(phi)
    v0 = r * jnp.sin(phi)
    z0 = jnp.stack([x0, v0], axis=-1)
    eps = latent_noise * jr.normal(k_eps, (n_trials, n_steps, 2))
    return z0, eps


def _check_synthesis_step_consistency(
    *,
    n_trials: int,
    n_steps: int,
    dt: float,
    mu: float,
    init_radius: float,
    init_radius_jitter: float,
    latent_noise: float,
) -> list[CheckResult]:
    key = jr.key(0)

    zs = simulate_vdp(
        key,
        n_trials=n_trials,
        n_steps=n_steps,
        dt=dt,
        mu=mu,
        init_radius=init_radius,
        init_radius_jitter=init_radius_jitter,
        latent_noise=latent_noise,
    )

    z0, eps = _reconstruct_z0_and_eps(
        key,
        n_trials=n_trials,
        n_steps=n_steps,
        init_radius=init_radius,
        init_radius_jitter=init_radius_jitter,
        latent_noise=latent_noise,
    )

    results: list[CheckResult] = []
    z = z0
    for t in range(n_steps):
        z_next = jax.vmap(lambda s, e: rk4_step(s, dt, mu=mu) + e)(z, eps[:, t])
        results.append(_assert_close(zs[:, t], z_next, name=f"synthesis_step[t={t}]"))
        z = z_next

    return results


def _check_toydynamics_matches_rk4(
    *,
    mu: float,
    dt: float,
    cov: float,
) -> list[CheckResult]:
    class Cfg:
        def __init__(self, mu, dt, cov, state_dim=2, input_dim=0, context_dim=0):
            self.mu = mu
            self.dt = dt
            self.cov = cov
            self.state_dim = state_dim
            self.input_dim = input_dim
            self.context_dim = context_dim

    dyn = ToyDynamics(Cfg(mu=mu, dt=dt, cov=cov), key=jr.key(1))

    key = jr.key(2)
    z = jr.normal(key, (32, 2))
    u0 = jnp.zeros((0,))
    c0 = jnp.zeros((0,))

    a = jax.vmap(lambda s: dyn.forward(s, u0, c0))(z)
    b = jax.vmap(lambda s: rk4_step(s, dt, mu=mu))(z)
    return [_assert_close(a, b, name="toydynamics_vs_rk4")]


def run_consistency_checks(
    *,
    n_trials: int,
    n_steps: int,
    dt: float,
    mu: float,
    init_radius: float,
    init_radius_jitter: float,
    latent_noise: float,
) -> list[CheckResult]:
    results = []
    results.extend(
        _check_synthesis_step_consistency(
            n_trials=n_trials,
            n_steps=n_steps,
            dt=dt,
            mu=mu,
            init_radius=init_radius,
            init_radius_jitter=init_radius_jitter,
            latent_noise=latent_noise,
        )
    )
    results.extend(_check_toydynamics_matches_rk4(mu=mu, dt=dt, cov=0.0))
    return results


def flow_metrics_vdp_vs_model(
    flow_model, *, mu: float, dt: float, xlim, vlim, grid: int
) -> FlowMetrics:
    """Compare model flow field against true Van der Pol on a grid."""
    xs = jnp.linspace(xlim[0], xlim[1], grid)
    vs = jnp.linspace(vlim[0], vlim[1], grid)
    X, V = jnp.meshgrid(xs, vs, indexing="xy")
    flat = jnp.stack([X, V], axis=-1).reshape(-1, 2)

    true_rhs = jax.vmap(lambda s: vdp_rhs(s, mu))(flat)

    u0 = jnp.zeros((0,))
    c0 = jnp.zeros((0,))

    def model_step(z):
        return flow_model.forward(z, u0, c0)

    pred = jax.vmap(model_step)(flat)
    inferred_rhs = (pred - flat) / dt

    mse = jnp.mean(jnp.sum((inferred_rhs - true_rhs) ** 2, axis=-1))
    norm = jnp.mean(jnp.sum(true_rhs**2, axis=-1)) + 1e-8
    nrmse = jnp.sqrt(mse / norm)

    dot = jnp.sum(inferred_rhs * true_rhs, axis=-1)
    denom = (
        jnp.linalg.norm(inferred_rhs, axis=-1) * jnp.linalg.norm(true_rhs, axis=-1)
        + 1e-8
    )
    angle = jnp.arccos(jnp.clip(dot / denom, -1.0, 1.0))
    mean_angle = jnp.mean(angle)

    return FlowMetrics(
        nrmse=float(nrmse.item()), mean_angle_rad=float(mean_angle.item())
    )


# ---------------------------------------------------------------------------
# Dynamics module (exact Van der Pol — no model mismatch)
# ---------------------------------------------------------------------------


class ToyDynamics(Dynamics):
    noise: DiagGaussian
    mu: float
    dt: float

    def __init__(self, conf, key):
        self.conf = conf
        self.noise = DiagGaussian(jnp.array(conf.cov), conf.state_dim)
        self.mu = float(conf.mu)
        self.dt = float(conf.dt)

    def forward(self, z, u, c, *, key=None):
        return rk4_step(z, self.dt, mu=self.mu)

    def loss(self):
        return jnp.mean(self.cov())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging("INFO")
    print("JAX devices:", jax.devices())

    # ----- Synthesis parameters -----
    N = 128  # trials
    T = 400  # time steps
    dt = 0.02  # integration step
    mu = 2.0  # Van der Pol nonlinearity

    obs_dim = 10
    state_dim = 2
    sigma_obs = 0.3

    init_radius = 2.0
    init_radius_jitter = 0.2
    latent_noise = 0.0

    key = jr.key(0)
    key, k_lat, k_C, k_b, k_y = jr.split(key, 5)

    latent_states = simulate_vdp(
        k_lat,
        n_trials=N,
        n_steps=T,
        dt=dt,
        mu=mu,
        init_radius=init_radius,
        init_radius_jitter=init_radius_jitter,
        latent_noise=latent_noise,
    )

    C = 0.7 * jr.normal(k_C, (obs_dim, state_dim))
    b = 0.1 * jr.normal(k_b, (obs_dim,))

    obs_mean = latent_states @ C.T + b
    observations = obs_mean + sigma_obs * jr.normal(k_y, obs_mean.shape)

    times = jnp.broadcast_to(jnp.arange(T), (N, T))
    inputs = jnp.zeros((N, T, 0))
    covariates = jnp.zeros((N, T, 0))

    print("latent_states:", latent_states.shape)
    print("observations:", observations.shape)

    if "--check-consistency" in sys.argv:
        results = run_consistency_checks(
            n_trials=8,
            n_steps=5,
            dt=dt,
            mu=mu,
            init_radius=init_radius,
            init_radius_jitter=init_radius_jitter,
            latent_noise=latent_noise,
        )
        print("OK: consistency checks passed")
        for r in results:
            print(f"- {r.name}: max_abs_err={r.max_abs_err:.3e}")

    # ----- Plot synthetic data -----
    out_dir = Path(__file__).resolve().parent
    trial = 0

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(latent_states[trial, :, 0], label="x")
    ax[0].plot(latent_states[trial, :, 1], label="v")
    ax[0].set_title("Van der Pol latent (one trial)")
    ax[0].legend()

    for d in range(min(4, obs_dim)):
        ax[1].plot(observations[trial, :, d], label=f"y[{d}]")
    ax[1].set_title("Gaussian observations (subset of dims)")
    ax[1].legend(ncol=4)

    plt.tight_layout()
    fig.savefig(out_dir / "toy_example_data.pdf")
    plt.close(fig)

    # ----- Model configuration -----
    xlim = (-3.0, 3.0)
    vlim = (-3.0, 3.0)
    grid = 25

    model_conf = OmegaConf.create(
        {
            "mode": "pseudo",
            "observation_dim": obs_dim,
            "state_dim": state_dim,
            "forward": "ToyDynamics",
            "approx": "DiagMVN",
            "mc_size": 1,
            "seed": 0,
            "n_steps": T,
            "fb_penalty": 0.0,
            "noise_penalty": 0.01,
            "dropout": 0.0,
            "dyn_conf": {
                "state_dim": state_dim,
                "input_dim": 0,
                "context_dim": 0,
                "mu": mu,
                "dt": dt,
                "cov": 0.0,
            },
            "enc_conf": {
                "observation_dim": obs_dim,
                "state_dim": state_dim,
                "approx": "DiagMVN",
                "width": 32,
                "depth": 2,
                "dropout": None,
            },
            "obs_conf": {
                "model": "GLM",
                "likelihood": "DiagGaussian",
                "observation_dim": obs_dim,
                "state_dim": state_dim,
                "cov": [float(sigma_obs**2)] * obs_dim,
                "norm_readout": False,
                "dropout": 0.0,
            },
        }
    )

    model = XFADS(model_conf, jr.key(123))

    # ----- Pre-training inference -----
    key, infer_key = jr.split(key)
    (
        posterior_natural_params,
        posterior_moment_params,
        predicted_moment_params,
    ) = model(times, observations, inputs, covariates, key=infer_key)

    print("posterior_moment_params:", posterior_moment_params.shape)

    def moment_to_mean_and_cov(moment_vec):
        mean, cov = model.approx.moment_to_canon(moment_vec)
        return mean, cov

    mean_and_cov_vmap = jax.vmap(jax.vmap(moment_to_mean_and_cov, in_axes=0), in_axes=0)
    posterior_means, posterior_covs = mean_and_cov_vmap(posterior_moment_params)

    posterior_rmse = jnp.sqrt(jnp.mean((posterior_means - latent_states) ** 2))
    print(f"posterior rmse before training: {float(posterior_rmse):.6f}")

    pre_flow = flow_metrics_vdp_vs_model(
        model, mu=mu, dt=dt, xlim=xlim, vlim=vlim, grid=grid
    )
    print(
        "flow metrics before training: "
        f"nrmse={pre_flow.nrmse:.4f}, mean_angle={pre_flow.mean_angle_rad:.4f} rad"
    )

    # ----- Training -----
    n_devices = len(jax.devices())
    max_batch = N // 2
    batch_size = max(n_devices, (min(32, max_batch) // n_devices) * n_devices)
    batch_size = max(n_devices, batch_size)

    validation_size = batch_size
    if validation_size >= N:
        validation_size = max(batch_size, (N // 4 // batch_size) * batch_size)
    validation_size = int(max(batch_size, min(validation_size, N - batch_size)))

    trainer_conf = OmegaConf.create(
        {
            "seed": 0,
            "learning_rate": 1e-3,
            "clip_norm": 5.0,
            "weight_decay": 1e-3,
            "noise_eta": 0.0,
            "noise_gamma": 0.8,
            "min_epoch": 0,
            "max_epoch": 100,
            "batch_size": int(batch_size),
            "validation_size": int(validation_size),
            "valid_ratio": 0.2,
        }
    )

    trained_model = train(
        model,
        (times, observations, inputs, covariates),
        conf=trainer_conf,
    )

    # ----- Post-training evaluation -----
    post_flow = flow_metrics_vdp_vs_model(
        trained_model, mu=mu, dt=dt, xlim=xlim, vlim=vlim, grid=grid
    )
    print(
        "flow metrics after training: "
        f"nrmse={post_flow.nrmse:.4f}, mean_angle={post_flow.mean_angle_rad:.4f} rad"
    )

    key, infer_key = jr.split(key)
    _, trained_posterior_moments, _ = trained_model(
        times, observations, inputs, covariates, key=infer_key
    )
    trained_posterior_means, trained_posterior_covs = mean_and_cov_vmap(
        trained_posterior_moments
    )
    trained_posterior_rmse = jnp.sqrt(
        jnp.mean((trained_posterior_means - latent_states) ** 2)
    )
    print(f"posterior rmse after training: {float(trained_posterior_rmse):.6f}")

    # ----- Posterior plot -----
    t = jnp.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, latent_states[trial, :, 0], label="true x", color="C0", linewidth=2)
    ax[0].plot(t, latent_states[trial, :, 1], label="true v", color="C1", linewidth=2)
    ax[0].plot(
        t,
        trained_posterior_means[trial, :, 0],
        "--",
        label="posterior mean x",
        color="C0",
    )
    ax[0].plot(
        t,
        trained_posterior_means[trial, :, 1],
        "--",
        label="posterior mean v",
        color="C1",
    )
    ax[0].set_title("Posterior mean vs true latent (one trial)")
    ax[0].legend(ncol=2)

    cov = trained_posterior_covs[trial]
    if cov.ndim == 3:
        cov = jax.vmap(jnp.diag)(cov)
    std = jnp.sqrt(jnp.clip(cov, a_min=0.0))

    ax[1].plot(t, trained_posterior_means[trial, :, 0], label="x mean", color="C0")
    ax[1].fill_between(
        t,
        trained_posterior_means[trial, :, 0] - 2 * std[:, 0],
        trained_posterior_means[trial, :, 0] + 2 * std[:, 0],
        color="C0",
        alpha=0.2,
        label="x +- 2sigma",
    )
    ax[1].plot(t, trained_posterior_means[trial, :, 1], label="v mean", color="C1")
    ax[1].fill_between(
        t,
        trained_posterior_means[trial, :, 1] - 2 * std[:, 1],
        trained_posterior_means[trial, :, 1] + 2 * std[:, 1],
        color="C1",
        alpha=0.2,
        label="v +- 2sigma",
    )
    ax[1].set_title("Posterior uncertainty (approx; DiagMVN)")
    ax[1].legend(ncol=2)

    plt.tight_layout()
    fig.savefig(out_dir / "toy_example_posterior.pdf")
    plt.close(fig)

    # ----- Flow-field plot -----
    xs = jnp.linspace(xlim[0], xlim[1], grid)
    vs = jnp.linspace(vlim[0], vlim[1], grid)
    X, V = jnp.meshgrid(xs, vs, indexing="xy")
    flat = jnp.stack([X, V], axis=-1).reshape(-1, 2)

    true_rhs = jax.vmap(lambda s: vdp_rhs(s, mu))(flat)
    true_dx = true_rhs[:, 0].reshape(grid, grid)
    true_dv = true_rhs[:, 1].reshape(grid, grid)

    u0 = jnp.zeros((0,))
    c0 = jnp.zeros((0,))

    def model_step(z):
        return trained_model.forward(z, u0, c0)

    pred = jax.vmap(model_step)(flat)
    inferred_rhs = (pred - flat) / dt
    inf_dx = inferred_rhs[:, 0].reshape(grid, grid)
    inf_dv = inferred_rhs[:, 1].reshape(grid, grid)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    ax[0].streamplot(
        np.asarray(X),
        np.asarray(V),
        np.asarray(true_dx),
        np.asarray(true_dv),
        density=1.2,
        linewidth=0.8,
        color="C0",
    )
    ax[0].plot(
        np.asarray(latent_states[trial, :, 0]),
        np.asarray(latent_states[trial, :, 1]),
        color="k",
        linewidth=1.2,
        label="true trajectory",
    )
    ax[0].set_title("True Van der Pol flow field")
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(vlim)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("v")
    ax[0].legend(loc="upper right")

    ax[1].streamplot(
        np.asarray(X),
        np.asarray(V),
        np.asarray(inf_dx),
        np.asarray(inf_dv),
        density=1.2,
        linewidth=0.8,
        color="C1",
    )
    ax[1].plot(
        np.asarray(trained_posterior_means[trial, :, 0]),
        np.asarray(trained_posterior_means[trial, :, 1]),
        color="k",
        linewidth=1.2,
        label="posterior mean",
    )
    ax[1].set_title("Inferred flow field (model)")
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(vlim)
    ax[1].set_xlabel("x")
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / "toy_example_flow.pdf")
    plt.close(fig)

    # ----- Save / load roundtrip -----
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy_model.zip"
        XFADS.save(trained_model, path)
        reloaded_model = XFADS.load(path)

        key, reload_key = jr.split(key)
        _ = reloaded_model(
            times[:1], observations[:1], inputs[:1], covariates[:1], key=reload_key
        )
        print("Save/load roundtrip: OK")


if __name__ == "__main__":
    main()
