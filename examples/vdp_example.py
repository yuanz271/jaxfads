"""Van der Pol example: XFADS with synthetic data.

Three cases:

1. **MLPStateMap** — trainable MLP learns continuous-time dynamics, stepped with RK4.
2. **OUStateMap** — built-in tracking prior drift.
3. **LoRaMVN + FunctionStateMap** — low-rank pseudo-observation encoder updates
   (paper Eq. 19–21), using a declarative function-backed Van der Pol map.

All use Factor Analysis readout initialisation. Evaluation uses Procrustes
alignment to compare inferred latents against ground truth.
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from equinox import nn as enn
from omegaconf import OmegaConf

from jaxfads import XFADS, configure_logging
from jaxfads.base import StateMap
from jaxfads.nn import make_mlp
from jaxfads.observations import GLM  # noqa: F401 — registers GLM
from jaxfads.trainer import train


# ---------------------------------------------------------------------------
# Van der Pol dynamics
# ---------------------------------------------------------------------------


def vdp_rhs(state: jax.Array, mu: float) -> jax.Array:
    """Right-hand side of the Van der Pol oscillator."""
    x, v = state[0], state[1]
    return jnp.stack([v, mu * (1.0 - x * x) * v - x])


def rk4_step(state: jax.Array, dt: float, *, mu: float) -> jax.Array:
    """Single RK4 integration step."""
    k1 = vdp_rhs(state, mu)
    k2 = vdp_rhs(state + 0.5 * dt * k1, mu)
    k3 = vdp_rhs(state + 0.5 * dt * k2, mu)
    k4 = vdp_rhs(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_vdp(
    key: jax.Array,
    *,
    n_trials: int,
    n_steps: int,
    dt: float,
    mu: float,
    init_radius: float = 2.0,
) -> jax.Array:
    """Generate Van der Pol trajectories via RK4.

    Returns
    -------
    jax.Array, shape (n_trials, n_steps, 2)
    """
    phi = jr.uniform(key, (n_trials,), maxval=2.0 * jnp.pi)
    z0 = init_radius * jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)

    def scan_fn(z, _):
        z_next = rk4_step(z, dt, mu=mu)
        return z_next, z_next

    _, zs = jax.vmap(lambda z: jax.lax.scan(scan_fn, z, None, length=n_steps))(z0)
    return zs


# ---------------------------------------------------------------------------
# State-map modules
# ---------------------------------------------------------------------------


def vdp_state_map(
    z: jax.Array,
    u: jax.Array,
    c: jax.Array,
    *,
    mu: float,
    key: jax.Array | None = None,
) -> jax.Array:
    """FunctionStateMap-compatible Van der Pol continuous-time map."""
    del u, c, key
    return vdp_rhs(z, mu)


class MLPStateMap(StateMap):
    """Trainable MLP dynamics with Euler integration.

    The MLP learns the continuous-time derivative ``f(z)``,
    and ``eval`` returns that derivative ``f(z)``.
    Noise is auto-initialised by XFADS via ``state_noise`` in dyn_conf.
    """

    net: enn.Sequential
    def __init__(self, conf, key):
        self.conf = conf
        self.net = make_mlp(
            conf.state_dim,
            conf.state_dim,
            width=conf.width,
            depth=conf.depth,
            key=key,
        )

    def rhs(self, z):
        """Continuous-time derivative f(z)."""
        return self.net(z)

    def eval(self, z, u, c, *, key=None):
        del u, c, key
        return self.rhs(z)


# ---------------------------------------------------------------------------
# Procrustes alignment: z_true ≈ A @ z_inferred + d
# ---------------------------------------------------------------------------


@dataclass
class AffineAlignment:
    """Affine map z_true ≈ A @ z_inferred + d."""

    A: jax.Array  # (D, D) linear part
    d: jax.Array  # (D,)   translation


def procrustes_affine(z_true: jax.Array, z_inferred: jax.Array) -> AffineAlignment:
    """Least-squares affine fit."""
    mu_inf, mu_true = z_inferred.mean(0), z_true.mean(0)
    A_T, *_ = jnp.linalg.lstsq(z_inferred - mu_inf, z_true - mu_true)
    A = A_T.T
    return AffineAlignment(A=A, d=mu_true - A @ mu_inf)


def align(aff: AffineAlignment, z: jax.Array) -> jax.Array:
    """Apply alignment: z @ A^T + d."""
    return z @ aff.A.T + aff.d


# ---------------------------------------------------------------------------
# Flow-field evaluation
# ---------------------------------------------------------------------------


def _model_rhs(model, grid_pts, alignment=None):
    """Evaluate the model's continuous-time derivative on a grid.

    Parameters
    ----------
    model : XFADS
        Trained model whose ``state_map`` implements ``eval(z, u, c, key=...)``.
    grid_pts : Array, shape (M, D)
        Points at which to evaluate the derivative (in true coordinates).
    alignment : AffineAlignment, optional
        If provided, grid points are mapped into latent space before
        evaluation, and the resulting derivatives are mapped back.
    """
    input_dim = int(model.state_map.conf.input_dim)
    context_dim = int(model.state_map.conf.context_dim)
    u0 = jnp.zeros((input_dim,), dtype=grid_pts.dtype)
    c0 = jnp.zeros((context_dim,), dtype=grid_pts.dtype)

    def rhs_fn(z):
        return model.state_map.eval(z, u0, c0, key=None)

    if alignment is not None:
        A_inv = jnp.linalg.inv(alignment.A)
        pts = (grid_pts - alignment.d) @ A_inv.T
        rhs = jax.vmap(rhs_fn)(pts)
        return rhs @ alignment.A.T

    return jax.vmap(rhs_fn)(grid_pts)


def flow_metrics(model, *, mu, xlim, vlim, data_pts, grid=25, alignment=None):
    """Density-weighted NRMSE and mean angle between true and model flow fields.

    Parameters
    ----------
    model : XFADS
        Trained model whose state map supports ``eval()``.
    mu : float
        Van der Pol parameter for the ground-truth RHS.
    xlim, vlim : tuple[float, float]
        Axis limits for the evaluation grid.
    data_pts : Array, shape (M, D)
        Flattened latent-state samples (in true coordinates) that define
        the data region.  Grid points are weighted by a Gaussian kernel
        density estimated from these points, so metrics emphasise the
        region where the model has seen training data.
    grid : int, optional
        Number of grid points per axis.  Default is ``25``.
    alignment : AffineAlignment, optional
        Procrustes alignment to map grid points into model latent space.

    Returns
    -------
    nrmse : float
        Density-weighted normalised RMSE of the flow field.
    angle : float
        Density-weighted mean angle (rad) between true and inferred flow.
    """
    xs = jnp.linspace(*xlim, grid)
    vs = jnp.linspace(*vlim, grid)
    X, V = jnp.meshgrid(xs, vs, indexing="xy")
    pts = jnp.stack([X, V], axis=-1).reshape(-1, 2)

    true_rhs = jax.vmap(lambda s: vdp_rhs(s, mu))(pts)
    inf_rhs = _model_rhs(model, pts, alignment)

    # Density weights: Gaussian kernel on min distance to data points
    # bandwidth = median nearest-neighbour distance in data_pts
    dists = jnp.linalg.norm(
        pts[:, None, :] - data_pts[None, :, :], axis=-1
    )  # (grid², M)
    min_dists = jnp.min(dists, axis=1)  # (grid²,)
    bandwidth = jnp.median(min_dists)
    weights = jnp.exp(-0.5 * (min_dists / (bandwidth + 1e-8)) ** 2)
    weights = weights / (jnp.sum(weights) + 1e-8)

    # Weighted NRMSE
    sq_err = jnp.sum((inf_rhs - true_rhs) ** 2, -1)
    sq_true = jnp.sum(true_rhs**2, -1)
    nrmse = float(
        jnp.sqrt(jnp.sum(weights * sq_err) / (jnp.sum(weights * sq_true) + 1e-8))
    )

    # Weighted mean angle
    dot = jnp.sum(inf_rhs * true_rhs, -1)
    norms = (
        jnp.linalg.norm(inf_rhs, axis=-1) * jnp.linalg.norm(true_rhs, axis=-1) + 1e-8
    )
    angles = jnp.arccos(jnp.clip(dot / norms, -1, 1))
    angle = float(jnp.sum(weights * angles))

    return nrmse, angle


# ---------------------------------------------------------------------------
# Evaluation (shared by both cases)
# ---------------------------------------------------------------------------


def _readout_bias(model):
    """Extract readout bias (handles NormalizedLinear wrapper)."""
    layer = model.observation.readout.layer
    return layer.layer.bias if hasattr(layer, "layer") else layer.bias


def evaluate(
    name,
    trained,
    latent_states,
    observations,
    data,
    C_true,
    b_true,
    *,
    key,
    approx,
    mu,
    xlim,
    vlim,
    grid,
    align_flow=True,
):
    """Infer, align, compute metrics, and print results.

    Parameters
    ----------
    align_flow : bool, default=True
        Whether to pass the Procrustes alignment to flow-field
        evaluation.  Set ``False`` when the dynamics operate in the
        true coordinate system (e.g. ``vdp_state_map``).

    Returns dict with metric values, aligned means, covariances, and
    the Procrustes alignment.
    """
    times, obs, inputs, covs = data
    D = latent_states.shape[-1]

    # Inference
    _, means, _ = trained(times, obs, inputs, covs, key=key)
    means, post_covs = jax.vmap(jax.vmap(approx.unpack))(means)

    # Procrustes alignment
    aff = procrustes_affine(latent_states.reshape(-1, D), means.reshape(-1, D))
    aligned = align(aff, means)

    # Metrics
    post_rmse = float(jnp.sqrt(jnp.mean((aligned - latent_states) ** 2)))

    C_l, b_l = trained.observation.readout.weight, _readout_bias(trained)
    A_inv = jnp.linalg.inv(aff.A)
    C_eff = C_l @ A_inv
    b_eff = b_l + C_l @ (-A_inv @ aff.d)
    C_rmse = float(jnp.sqrt(jnp.mean((C_eff - C_true) ** 2)))
    b_rmse = float(jnp.sqrt(jnp.mean((b_eff - b_true) ** 2)))

    y_hat = means @ C_l.T + b_l
    obs_rmse = float(jnp.sqrt(jnp.mean((y_hat - observations) ** 2)))

    flow_aff = aff if align_flow else None
    # Use true latent states as the data region for density weighting
    data_pts_flat = latent_states.reshape(-1, D)
    nrmse, angle = flow_metrics(
        trained,
        mu=mu,
        xlim=xlim,
        vlim=vlim,
        data_pts=data_pts_flat,
        grid=grid,
        alignment=flow_aff,
    )

    print(f"\n--- {name} ---")
    print(f"  posterior RMSE (aligned): {post_rmse:.4f}")
    print(f"  readout C RMSE: {C_rmse:.4f},  b RMSE: {b_rmse:.4f}")
    print(f"  obs recon RMSE: {obs_rmse:.4f}")
    print(f"  flow NRMSE: {nrmse:.4f},  angle: {angle:.4f} rad")

    return dict(
        post_rmse=post_rmse,
        C_rmse=C_rmse,
        b_rmse=b_rmse,
        obs_rmse=obs_rmse,
        flow_nrmse=nrmse,
        flow_angle=angle,
        aligned_means=aligned,
        covs=post_covs,
        alignment=aff,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_posterior(name, means, covs, truth, trial, T, out_dir):
    """Posterior mean vs truth and ±2σ uncertainty bands."""
    t = jnp.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i, label in enumerate(("x", "v")):
        ax[0].plot(t, truth[trial, :, i], label=f"true {label}", color=f"C{i}", lw=2)
        ax[0].plot(
            t,
            means[trial, :, i],
            "--",
            label=f"inferred {label}",
            color=f"C{i}",
        )
    ax[0].set_title(f"Posterior mean vs truth — {name}")
    ax[0].legend(ncol=2)

    cov = covs[trial]
    if cov.ndim == 3:
        cov = jax.vmap(jnp.diag)(cov)
    std = jnp.sqrt(jnp.clip(cov, a_min=0.0))

    for i, label in enumerate(("x", "v")):
        ax[1].plot(t, means[trial, :, i], color=f"C{i}")
        ax[1].fill_between(
            t,
            means[trial, :, i] - 2 * std[:, i],
            means[trial, :, i] + 2 * std[:, i],
            color=f"C{i}",
            alpha=0.2,
            label=f"{label} ±2σ",
        )
    ax[1].set_title(f"Posterior uncertainty — {name}")
    ax[1].legend(ncol=2)

    plt.tight_layout()
    fig.savefig(out_dir / f"vdp_example_posterior_{name}.pdf")
    plt.close(fig)


def plot_flow_field(
    name,
    model,
    true_traj,
    inferred_traj,
    mu,
    xlim,
    vlim,
    grid,
    out_dir,
    *,
    alignment=None,
):
    """Side-by-side streamplots: true vs inferred flow field."""
    xs = jnp.linspace(*xlim, grid)
    vs = jnp.linspace(*vlim, grid)
    X, V = jnp.meshgrid(xs, vs, indexing="xy")
    pts = jnp.stack([X, V], axis=-1).reshape(-1, 2)

    true_rhs = jax.vmap(lambda s: vdp_rhs(s, mu))(pts)
    inf_rhs = _model_rhs(model, pts, alignment)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for a, rhs, traj, title, color in [
        (ax[0], true_rhs, true_traj, "True VdP", "C0"),
        (ax[1], inf_rhs, inferred_traj, f"Inferred — {name}", "C1"),
    ]:
        dx = rhs[:, 0].reshape(grid, grid)
        dv = rhs[:, 1].reshape(grid, grid)
        a.streamplot(
            np.asarray(X),
            np.asarray(V),
            np.asarray(dx),
            np.asarray(dv),
            density=1.2,
            linewidth=0.8,
            color=color,
        )
        a.plot(np.asarray(traj[:, 0]), np.asarray(traj[:, 1]), color="k", lw=1.2)
        a.set(title=title, xlim=xlim, ylim=vlim, xlabel="x")
    ax[0].set_ylabel("v")

    plt.tight_layout()
    fig.savefig(out_dir / f"vdp_example_flow_{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging("INFO")
    print("JAX devices:", jax.devices())

    # ----- Synthesis -----
    N, T, dt, mu = 128, 500, 0.04, 2.0
    obs_dim, state_dim = 10, 2
    sigma_obs = 0.3

    key, k_lat, k_C, k_b, k_y = jr.split(jr.key(0), 5)

    latent_states = simulate_vdp(k_lat, n_trials=N, n_steps=T, dt=dt, mu=mu)

    C_true = 0.7 * jr.normal(k_C, (obs_dim, state_dim))
    b_true = 0.1 * jr.normal(k_b, (obs_dim,))
    observations = (
        latent_states @ C_true.T + b_true + sigma_obs * jr.normal(k_y, (N, T, obs_dim))
    )

    times = jnp.broadcast_to(jnp.arange(T), (N, T))
    inputs = jnp.zeros((N, T, 0))
    covariates = jnp.zeros((N, T, 0))
    data = (times, observations, inputs, covariates)
    print(f"latent: {latent_states.shape}, observations: {observations.shape}")

    # ----- Plot synthetic data -----
    out_dir = Path(__file__).resolve().parent

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(latent_states[0, :, 0], label="x")
    ax[0].plot(latent_states[0, :, 1], label="v")
    ax[0].set_title("Van der Pol latent (trial 0)")
    ax[0].legend()
    for d in range(min(4, obs_dim)):
        ax[1].plot(observations[0, :, d], label=f"y[{d}]")
    ax[1].set_title("Observations (first 4 dims)")
    ax[1].legend(ncol=4)
    plt.tight_layout()
    fig.savefig(out_dir / "vdp_example_data.pdf")
    plt.close(fig)

    # ----- Shared configuration -----
    xlim, vlim, grid = (-3.0, 3.0), (-3.0, 3.0), 25

    enc_conf = dict(
        width=32,
        depth=2,
        dropout=None,
    )
    obs_conf = dict(
        model="GLM",
        likelihood="Gaussian",
        cov=[float(sigma_obs**2)] * obs_dim,
        norm_readout=False,
        dropout=0.0,
        readout_init_conf=dict(obs_noise_var=float(sigma_obs**2)),
    )
    shared_conf = dict(
        mode="smooth",
        observation_dim=obs_dim,
        state_dim=state_dim,
        approx="MVN",
        # Explicit for tutorial purposes. Use {"structure": "diag"} for a
        # diagonal exponential-family Gaussian approximation.
        approx_kwargs={"structure": "full"},
        seed=0,
        n_steps=T,
        fb_penalty=0.0,
        noise_penalty=0.01,
        dropout=0.0,
        enc_conf=enc_conf,
        obs_conf=obs_conf,
    )

    n_devices = len(jax.devices())
    batch_size = max(n_devices, (32 // n_devices) * n_devices)

    base_trainer_conf = dict(
        seed=0,
        learning_rate=1e-3,
        clip_norm=5.0,
        weight_decay=1e-3,
        noise_eta=0.0,
        noise_gamma=0.8,
        min_epoch=0,
        batch_size=batch_size,
        validation_size=batch_size,
        valid_ratio=0.2,
    )

    trainer_conf_mlp = OmegaConf.create({**base_trainer_conf, "max_epoch": 500})
    trainer_conf_ou = OmegaConf.create({**base_trainer_conf, "max_epoch": 100})
    trainer_conf_lora = OmegaConf.create(
        {
            **base_trainer_conf,
            "max_epoch": 100,
            # Keep transition noise fixed for this low-rank encoder demo.
            "freeze_paths": ["noise_free"],
        }
    )

    eval_kw = dict(mu=mu, xlim=xlim, vlim=vlim, grid=grid)

    # ===================================================================
    # Case 1: MLPStateMap (learned dynamics)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Case 1: MLPStateMap (learned dynamics)")
    print("=" * 60)

    conf1 = OmegaConf.create(
        {
            **shared_conf,
            "state_map": "MLPStateMap",
            "stepper": "RK4Stepper",
            "mc_size": 4,
            "dyn_conf": dict(
                input_dim=0,
                context_dim=0,
                state_noise=1.0,
                width=32,
                depth=1,
                dt=dt,
                system_type="continuous",
            ),
        }
    )
    model1 = XFADS(conf1, jr.key(456))
    model1 = model1.initialize(*data)

    trained1 = train(model1, data, conf=trainer_conf_mlp)

    key, k = jr.split(key)
    r1 = evaluate(
        "MLPStateMap",
        trained1,
        latent_states,
        observations,
        data,
        C_true,
        b_true,
        key=k,
        approx=trained1.approx,
        **eval_kw,
    )
    plot_posterior(
        "mlp",
        r1["aligned_means"],
        r1["covs"],
        latent_states,
        0,
        T,
        out_dir,
    )
    plot_flow_field(
        "mlp",
        trained1,
        latent_states[0],
        r1["aligned_means"][0],
        mu,
        xlim,
        vlim,
        grid,
        out_dir,
        alignment=r1["alignment"],
    )

    # ===================================================================
    # Case 2: OUStateMap (diffusion-style tracking prior)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Case 2: OUStateMap (diffusion-style tracking prior)")
    print("=" * 60)

    conf2 = OmegaConf.create(
        {
            **shared_conf,
            "state_map": "OUStateMap",
            "stepper": "EulerStepper",
            "mc_size": 4,
            "dyn_conf": dict(
                input_dim=0,
                context_dim=0,
                theta=2.0,
                dt=dt,
                system_type="continuous",
                state_noise=1.0,
            ),
        }
    )
    model2 = XFADS(conf2, jr.key(789))
    model2 = model2.initialize(*data)

    trained2 = train(model2, data, conf=trainer_conf_ou)

    key, k = jr.split(key)
    r2 = evaluate(
        "OUStateMap",
        trained2,
        latent_states,
        observations,
        data,
        C_true,
        b_true,
        key=k,
        approx=trained2.approx,
        **eval_kw,
    )
    plot_posterior(
        "ou",
        r2["aligned_means"],
        r2["covs"],
        latent_states,
        0,
        T,
        out_dir,
    )
    plot_flow_field(
        "ou",
        trained2,
        latent_states[0],
        r2["aligned_means"][0],
        mu,
        xlim,
        vlim,
        grid,
        out_dir,
        alignment=r2["alignment"],
    )

    # ===================================================================
    # Case 3: LoRaMVN encoder updates + declarative FunctionStateMap
    # ===================================================================
    print("\n" + "=" * 60)
    print("Case 3: LoRaMVN + FunctionStateMap")
    print("=" * 60)

    conf3 = OmegaConf.create(
        {
            **shared_conf,
            "approx": "LoRaMVN",
            "approx_kwargs": {"rank": 1},
            "state_map": "FunctionStateMap",
            "stepper": "RK4Stepper",
            "mc_size": 4,
            "dyn_conf": dict(
                input_dim=0,
                context_dim=0,
                fn_path="__main__:vdp_state_map",
                fn_kwargs={"mu": mu},
                dt=dt,
                system_type="continuous",
                state_noise=1.0,
            ),
        }
    )
    model3 = XFADS(conf3, jr.key(321))
    model3 = model3.initialize(*data)

    trained3 = train(model3, data, conf=trainer_conf_lora)

    key, k = jr.split(key)
    r3 = evaluate(
        "LoRaMVN",
        trained3,
        latent_states,
        observations,
        data,
        C_true,
        b_true,
        key=k,
        align_flow=False,
        approx=trained3.approx,
        **eval_kw,
    )
    plot_posterior(
        "lora",
        r3["aligned_means"],
        r3["covs"],
        latent_states,
        0,
        T,
        out_dir,
    )
    plot_flow_field(
        "lora",
        trained3,
        latent_states[0],
        r3["aligned_means"][0],
        mu,
        xlim,
        vlim,
        grid,
        out_dir,
    )

    # Save / load roundtrip (covers LoRaMVN + injected encoder free size)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.zip"
        XFADS.save(trained3, path)
        reloaded = XFADS.load(path)
        reloaded(
            times[:1], observations[:1], inputs[:1], covariates[:1], key=jr.key(99)
        )
        print("Save/load roundtrip: OK")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 74)
    print(f"Summary  (Procrustes-aligned; obs noise σ = {sigma_obs})")
    print("=" * 74)
    header = (
        f"{'Metric':<30s} {'MLPStateMap':>14s} {'OUStateMap':>14s} {'LoRaMVN':>14s}"
    )
    print(header)
    print("-" * len(header))
    for metric, label in [
        ("post_rmse", "Posterior RMSE"),
        ("C_rmse", "Readout C RMSE"),
        ("b_rmse", "Readout b RMSE"),
        ("obs_rmse", "Obs recon RMSE"),
        ("flow_nrmse", "Flow NRMSE"),
        ("flow_angle", "Flow angle (rad)"),
    ]:
        print(
            f"{label:<30s} {r1[metric]:>14.4f} {r2[metric]:>14.4f} {r3[metric]:>14.4f}"
        )
    print("=" * len(header))


if __name__ == "__main__":
    main()
