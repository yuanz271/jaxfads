"""Toy example: XFADS with synthetic Van der Pol oscillator data.

Demonstrates variational Bayesian state-space modeling on a 2D Van der Pol
latent with 10D Gaussian observations.  Two cases are shown:

1. **ToyDynamics** — exact Van der Pol RK4 step (no model mismatch).
   The readout is initialised to the true observation matrix, so learning
   is concentrated in the encoder.

2. **MLPDynamics** — a trainable residual MLP that must learn the dynamics
   from data.  Because the readout and dynamics are both free, the latent
   space is only determined up to an affine transformation.  Evaluation
   uses Procrustes alignment to map inferred states back to the true
   coordinate system before computing RMSE and flow-field metrics.
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

from equinox import nn as enn

from jaxfads import XFADS, configure_logging
from jaxfads.base import Dynamics
from jaxfads.dynamics import DiagGaussian
from jaxfads.nn import make_mlp
from jaxfads.observations import GLM  # noqa: F401 — registers GLM with SubclassRegistry
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


def _model_rhs(flow_model, flat, dt, *, alignment=None):
    """Compute model-predicted RHS on grid points.

    When *alignment* is given, grid points in true coordinates are mapped
    to the model's latent space, the dynamics are evaluated there, and
    the result is mapped back:

        f_aligned(z) = A @ (f_model(A⁻¹(z − d)) − A⁻¹(z − d)) / dt

    Parameters
    ----------
    flow_model : XFADS or Dynamics
        Model with a `forward(z, u, c)` method.
    flat : Array, shape (M, D)
        Grid points (in true coordinates).
    dt : float
        Integration time-step.
    alignment : AffineAlignment or None
        If given, transform through the affine map.

    Returns
    -------
    Array, shape (M, D)
        Estimated RHS at each grid point (in true coordinates).
    """
    u0 = jnp.zeros((0,))
    c0 = jnp.zeros((0,))

    if alignment is not None:
        A_inv = jnp.linalg.inv(alignment.A)
        flat_model = (flat - alignment.d) @ A_inv.T  # true → model coords

        def model_step(z):
            return flow_model.forward(z, u0, c0)

        pred_model = jax.vmap(model_step)(flat_model)
        # Map the displacement back to true coords
        return (pred_model - flat_model) @ alignment.A.T / dt
    else:

        def model_step(z):
            return flow_model.forward(z, u0, c0)

        pred = jax.vmap(model_step)(flat)
        return (pred - flat) / dt


def flow_metrics_vdp_vs_model(
    flow_model,
    *,
    mu: float,
    dt: float,
    xlim,
    vlim,
    grid: int,
    alignment: AffineAlignment | None = None,
) -> FlowMetrics:
    """Compare model flow field against true Van der Pol on a grid.

    Parameters
    ----------
    flow_model : XFADS or Dynamics
        Model with a ``forward(z, u, c)`` method.
    mu : float
        Van der Pol nonlinearity parameter.
    dt : float
        Integration time-step.
    xlim, vlim : tuple
        Grid extent.
    grid : int
        Number of grid points per axis.
    alignment : AffineAlignment or None
        If given, model dynamics are evaluated in the model's latent
        space and mapped back to true coordinates before comparison.
    """
    xs = jnp.linspace(xlim[0], xlim[1], grid)
    vs = jnp.linspace(vlim[0], vlim[1], grid)
    X, V = jnp.meshgrid(xs, vs, indexing="xy")
    flat = jnp.stack([X, V], axis=-1).reshape(-1, 2)

    true_rhs = jax.vmap(lambda s: vdp_rhs(s, mu))(flat)
    inferred_rhs = _model_rhs(flow_model, flat, dt, alignment=alignment)

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


class MLPDynamics(Dynamics):
    """Trainable MLP dynamics with residual connection.

    Learns z_{t+1} = z_t + net(z_t).  At initialisation the MLP output is
    near-zero so the dynamics approximate the identity map, which keeps
    filtering stable before any training.

    Note: ``u`` and ``c`` are ignored — they are zero-dimensional in this
    toy example.
    """

    noise: DiagGaussian
    net: enn.Sequential

    def __init__(self, conf, key):
        self.conf = conf
        self.noise = DiagGaussian(jnp.array(conf.cov), conf.state_dim)
        self.net = make_mlp(
            conf.state_dim,
            conf.state_dim,
            width=conf.width,
            depth=conf.depth,
            key=key,
        )

    def forward(self, z, u, c, *, key=None):
        return z + self.net(z)

    def loss(self):
        return jnp.mean(self.cov())


# ---------------------------------------------------------------------------
# Procrustes (affine) alignment
# ---------------------------------------------------------------------------


@dataclass
class AffineAlignment:
    """Affine map z_true ≈ A @ z_inferred + d.

    Attributes
    ----------
    A : Array, shape (state_dim, state_dim)
        Linear part (rotation + scaling).
    d : Array, shape (state_dim,)
        Translation (bias).
    """

    A: jax.Array
    d: jax.Array


def procrustes_affine(
    z_true: jax.Array, z_inferred: jax.Array
) -> AffineAlignment:
    """Fit the best affine map  z_true ≈ A @ z_inferred + d  (least-squares).

    Parameters
    ----------
    z_true : Array, shape (N, D)
        Ground-truth latent states (flattened across trials/time).
    z_inferred : Array, shape (N, D)
        Inferred latent states (same shape).

    Returns
    -------
    AffineAlignment
        The fitted affine map.
    """
    mu_inf = jnp.mean(z_inferred, axis=0)
    mu_true = jnp.mean(z_true, axis=0)
    z_c = z_inferred - mu_inf
    t_c = z_true - mu_true
    # Solve  t_c ≈ z_c @ A^T  ⟹  A^T = pinv(z_c) @ t_c
    A_T, _, _, _ = jnp.linalg.lstsq(z_c, t_c)
    A = A_T.T
    d = mu_true - A @ mu_inf
    return AffineAlignment(A=A, d=d)


def align(alignment: AffineAlignment, z: jax.Array) -> jax.Array:
    """Apply affine alignment:  z_aligned = z @ A^T + d.

    Parameters
    ----------
    alignment : AffineAlignment
        Fitted alignment.
    z : Array, shape (..., D)
        Inferred latent states.

    Returns
    -------
    Array
        Aligned latent states in the true coordinate system.
    """
    return z @ alignment.A.T + alignment.d


# ---------------------------------------------------------------------------
# Readout initialisation helpers
# ---------------------------------------------------------------------------


def init_readout(model, C: jax.Array, b: jax.Array):
    """Set the observation readout weight and bias to known values.

    For synthetic experiments where the true readout is known, this
    removes the latent-space rotation ambiguity and focuses learning
    on the dynamics and encoder.

    Parameters
    ----------
    model : XFADS
        Model whose readout will be replaced.
    C : Array, shape (obs_dim, state_dim)
        Readout weight matrix.
    b : Array, shape (obs_dim,)
        Readout bias vector.
    """
    import equinox as eqx

    readout = model.observation.readout
    if hasattr(readout.layer, "layer"):
        # WeightNorm wrapper
        new_readout = eqx.tree_at(
            lambda r: r.layer.layer.weight, readout, C
        )
    else:
        new_readout = eqx.tree_at(lambda r: r.layer.weight, readout, C)
    new_readout = new_readout.initialize(b)

    return eqx.tree_at(lambda m: m.observation.readout, model, new_readout)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_posterior(
    case_name: str,
    means: jax.Array,
    covs: jax.Array,
    latent_states: jax.Array,
    trial: int,
    T: int,
    out_dir: Path,
) -> None:
    """Plot posterior mean vs truth and uncertainty bands for one trial."""
    t = jnp.arange(T)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, latent_states[trial, :, 0], label="true x", color="C0", linewidth=2)
    ax[0].plot(t, latent_states[trial, :, 1], label="true v", color="C1", linewidth=2)
    ax[0].plot(
        t, means[trial, :, 0], "--", label="posterior mean x", color="C0",
    )
    ax[0].plot(
        t, means[trial, :, 1], "--", label="posterior mean v", color="C1",
    )
    ax[0].set_title(f"Posterior mean vs true latent — {case_name} (one trial)")
    ax[0].legend(ncol=2)

    cov = covs[trial]
    if cov.ndim == 3:
        cov = jax.vmap(jnp.diag)(cov)
    std = jnp.sqrt(jnp.clip(cov, a_min=0.0))

    ax[1].plot(t, means[trial, :, 0], label="x mean", color="C0")
    ax[1].fill_between(
        t,
        means[trial, :, 0] - 2 * std[:, 0],
        means[trial, :, 0] + 2 * std[:, 0],
        color="C0",
        alpha=0.2,
        label="x +- 2sigma",
    )
    ax[1].plot(t, means[trial, :, 1], label="v mean", color="C1")
    ax[1].fill_between(
        t,
        means[trial, :, 1] - 2 * std[:, 1],
        means[trial, :, 1] + 2 * std[:, 1],
        color="C1",
        alpha=0.2,
        label="v +- 2sigma",
    )
    ax[1].set_title(f"Posterior uncertainty (approx; DiagMVN) — {case_name}")
    ax[1].legend(ncol=2)

    plt.tight_layout()
    fig.savefig(out_dir / f"toy_example_posterior_{case_name}.pdf")
    plt.close(fig)


def plot_flow_field(
    case_name: str,
    model,
    true_trajectory: jax.Array,
    inferred_trajectory: jax.Array,
    mu: float,
    dt: float,
    xlim: tuple[float, float],
    vlim: tuple[float, float],
    grid: int,
    out_dir: Path,
    *,
    alignment: AffineAlignment | None = None,
) -> None:
    """Plot true vs inferred flow field as side-by-side streamplots.

    Parameters
    ----------
    alignment : AffineAlignment or None
        When given, the model's dynamics are evaluated in its own latent
        space and transformed back to the true coordinate system via
        the affine map before plotting.
    """
    xs = jnp.linspace(xlim[0], xlim[1], grid)
    vs = jnp.linspace(vlim[0], vlim[1], grid)
    X, V = jnp.meshgrid(xs, vs, indexing="xy")
    flat = jnp.stack([X, V], axis=-1).reshape(-1, 2)

    true_rhs = jax.vmap(lambda s: vdp_rhs(s, mu))(flat)
    true_dx = true_rhs[:, 0].reshape(grid, grid)
    true_dv = true_rhs[:, 1].reshape(grid, grid)

    inferred_rhs = _model_rhs(model, flat, dt, alignment=alignment)
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
        np.asarray(true_trajectory[:, 0]),
        np.asarray(true_trajectory[:, 1]),
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
        np.asarray(inferred_trajectory[:, 0]),
        np.asarray(inferred_trajectory[:, 1]),
        color="k",
        linewidth=1.2,
        label="posterior mean",
    )
    ax[1].set_title(f"Inferred flow field — {case_name}")
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(vlim)
    ax[1].set_xlabel("x")
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / f"toy_example_flow_{case_name}.pdf")
    plt.close(fig)


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

    C = np.asarray(0.7 * jr.normal(k_C, (obs_dim, state_dim)))
    b = np.asarray(0.1 * jr.normal(k_b, (obs_dim,)))

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

    # ----- Shared configs -----
    enc_conf = {
        "observation_dim": obs_dim,
        "state_dim": state_dim,
        "approx": "DiagMVN",
        "width": 32,
        "depth": 2,
        "dropout": None,
    }
    obs_conf = {
        "model": "GLM",
        "likelihood": "DiagGaussian",
        "observation_dim": obs_dim,
        "state_dim": state_dim,
        "cov": [float(sigma_obs**2)] * obs_dim,
        "norm_readout": False,
        "dropout": 0.0,
    }

    n_devices = len(jax.devices())
    max_batch = N // 2
    batch_size = max(n_devices, (min(32, max_batch) // n_devices) * n_devices)
    batch_size = max(n_devices, batch_size)

    validation_size = batch_size
    if validation_size >= N:
        validation_size = max(batch_size, (N // 4 // batch_size) * batch_size)
    validation_size = int(max(batch_size, min(validation_size, N - batch_size)))

    base_trainer_conf = {
        "seed": 0,
        "learning_rate": 1e-3,
        "clip_norm": 5.0,
        "weight_decay": 1e-3,
        "noise_eta": 0.0,
        "noise_gamma": 0.8,
        "min_epoch": 0,
        "batch_size": int(batch_size),
        "validation_size": int(validation_size),
        "valid_ratio": 0.2,
    }

    data = (times, observations, inputs, covariates)

    # ===================================================================
    # Case 1: ToyDynamics (exact Van der Pol — no model mismatch)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Case 1: ToyDynamics (exact Van der Pol)")
    print("=" * 60)

    toy_model_conf = OmegaConf.create(
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
            "enc_conf": enc_conf,
            "obs_conf": obs_conf,
        }
    )

    toy_model = XFADS(toy_model_conf, jr.key(123))
    toy_model = init_readout(toy_model, C, b)

    def moment_to_mean_and_cov(moment_vec):
        mean, cov = toy_model.approx.moment_to_canon(moment_vec)
        return mean, cov

    mean_and_cov_vmap = jax.vmap(jax.vmap(moment_to_mean_and_cov, in_axes=0), in_axes=0)

    # Pre-training inference
    key, infer_key = jr.split(key)
    _, posterior_moment_params, _ = toy_model(
        times, observations, inputs, covariates, key=infer_key
    )
    posterior_means, _ = mean_and_cov_vmap(posterior_moment_params)
    posterior_rmse = jnp.sqrt(jnp.mean((posterior_means - latent_states) ** 2))
    print(f"posterior rmse before training: {float(posterior_rmse):.6f}")

    pre_flow = flow_metrics_vdp_vs_model(
        toy_model, mu=mu, dt=dt, xlim=xlim, vlim=vlim, grid=grid
    )
    print(
        "flow metrics before training: "
        f"nrmse={pre_flow.nrmse:.4f}, mean_angle={pre_flow.mean_angle_rad:.4f} rad"
    )

    # Training
    toy_trainer_conf = OmegaConf.create(
        {**base_trainer_conf, "max_epoch": 100}
    )
    trained_toy = train(toy_model, data, conf=toy_trainer_conf)

    # Post-training evaluation
    post_flow = flow_metrics_vdp_vs_model(
        trained_toy, mu=mu, dt=dt, xlim=xlim, vlim=vlim, grid=grid
    )
    print(
        "flow metrics after training: "
        f"nrmse={post_flow.nrmse:.4f}, mean_angle={post_flow.mean_angle_rad:.4f} rad"
    )

    key, infer_key = jr.split(key)
    _, trained_moments, _ = trained_toy(
        times, observations, inputs, covariates, key=infer_key
    )
    trained_means, trained_covs = mean_and_cov_vmap(trained_moments)
    trained_rmse = jnp.sqrt(jnp.mean((trained_means - latent_states) ** 2))
    print(f"posterior rmse after training: {float(trained_rmse):.6f}")

    plot_posterior("toy", trained_means, trained_covs, latent_states, trial, T, out_dir)
    plot_flow_field(
        "toy", trained_toy, latent_states[trial], trained_means[trial],
        mu, dt, xlim, vlim, grid, out_dir,
    )

    # Save / load roundtrip
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy_model.zip"
        XFADS.save(trained_toy, path)
        reloaded_model = XFADS.load(path)

        key, reload_key = jr.split(key)
        _ = reloaded_model(
            times[:1], observations[:1], inputs[:1], covariates[:1], key=reload_key
        )
        print("Save/load roundtrip: OK")

    # ===================================================================
    # Case 2: MLPDynamics (learned dynamics — model mismatch)
    # ===================================================================
    #
    # With learned dynamics and a learned readout the latent space is
    # only determined up to an affine transformation.  All evaluation
    # uses Procrustes alignment to map inferred states back to the true
    # coordinate system before computing RMSE and flow-field metrics.
    print("\n" + "=" * 60)
    print("Case 2: MLPDynamics (learned dynamics)")
    print("=" * 60)

    mlp_model_conf = OmegaConf.create(
        {
            "mode": "pseudo",
            "observation_dim": obs_dim,
            "state_dim": state_dim,
            "forward": "MLPDynamics",
            "approx": "DiagMVN",
            "mc_size": 4,
            "seed": 0,
            "n_steps": T,
            "fb_penalty": 0.0,
            "noise_penalty": 0.01,
            "dropout": 0.0,
            "dyn_conf": {
                "state_dim": state_dim,
                "input_dim": 0,
                "context_dim": 0,
                "cov": 1.0,
                "width": 32,
                "depth": 1,
            },
            "enc_conf": enc_conf,
            "obs_conf": obs_conf,
        }
    )

    mlp_model = XFADS(mlp_model_conf, jr.key(456))

    # Training
    mlp_trainer_conf = OmegaConf.create(
        {**base_trainer_conf, "max_epoch": 300}
    )
    trained_mlp = train(mlp_model, data, conf=mlp_trainer_conf)

    # Post-training inference
    key, infer_key = jr.split(key)
    _, trained_moments, _ = trained_mlp(
        times, observations, inputs, covariates, key=infer_key
    )
    trained_means, trained_covs = mean_and_cov_vmap(trained_moments)

    # --- Procrustes alignment ---
    # Flatten (N, T, D) → (N*T, D) and fit z_true ≈ A @ z_inferred + d
    flat_true = latent_states.reshape(-1, state_dim)
    flat_inf = trained_means.reshape(-1, state_dim)
    aff = procrustes_affine(flat_true, flat_inf)
    print(f"alignment matrix A:\n{np.asarray(aff.A)}")
    print(f"alignment offset d: {np.asarray(aff.d)}")

    aligned_means = align(aff, trained_means)
    aligned_rmse = jnp.sqrt(jnp.mean((aligned_means - latent_states) ** 2))
    print(f"posterior rmse (aligned): {float(aligned_rmse):.6f}")

    post_flow = flow_metrics_vdp_vs_model(
        trained_mlp, mu=mu, dt=dt, xlim=xlim, vlim=vlim, grid=grid,
        alignment=aff,
    )
    print(
        "flow metrics (aligned): "
        f"nrmse={post_flow.nrmse:.4f}, mean_angle={post_flow.mean_angle_rad:.4f} rad"
    )

    # Plots use aligned posterior means so they share the true scale
    plot_posterior(
        "mlp", aligned_means, trained_covs, latent_states, trial, T, out_dir,
    )
    plot_flow_field(
        "mlp", trained_mlp, latent_states[trial], aligned_means[trial],
        mu, dt, xlim, vlim, grid, out_dir,
        alignment=aff,
    )

    # Save/load roundtrip skipped for MLPDynamics: the class is defined in this
    # example file, so a saved model cannot be loaded from a different script
    # without also defining/importing MLPDynamics.


if __name__ == "__main__":
    main()
