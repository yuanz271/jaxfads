"""PCA dynamics verification.

Section 1: direct gradient equivalence check (KL vs MSE gradients).
Section 2: NOFILT end-to-end training with fixed PCA encoder/observation.

Run with:
    uv run python benchmarks/pca_dynamics_verification.py
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import OmegaConf
import tensorflow_probability.substrates.jax.distributions as tfp

from jaxfads import XFADS
from jaxfads.base import Encoder, Observation, Approx, StateMap
from jaxfads.distributions import MVN
from jaxfads.trainer import train


class PCAEncoder(Encoder):
    """Fixed PCA projection: z = C^T(y - b). No trainable parameters."""

    weight: jax.Array
    bias: jax.Array

    def __init__(self, conf, key=None):
        del key
        self.conf = conf
        self.weight = jnp.zeros((int(conf.observation_dim), int(conf.state_dim)))
        self.bias = jnp.zeros((int(conf.observation_dim),))

    def __call__(self, y, *, key=None):
        del key
        return (y - self.bias) @ self.weight


class PCAObservation(Observation):
    """Fixed Gaussian observation with PCA readout. No trainable parameters."""

    weight: jax.Array
    bias: jax.Array
    obs_var: float

    def __init__(self, conf, key=None):
        del key
        self.conf = conf
        d_obs = int(conf.observation_dim)
        d_state = int(conf.state_dim)
        self.weight = jnp.zeros((d_obs, d_state))
        self.bias = jnp.zeros((d_obs,))
        self.obs_var = float(conf.get("obs_var", 1e-4))

    def eloglik(self, key, t, moment, y, approx: Approx, mc_size):
        del key, t, mc_size
        mean_z, cov_z = approx.unpack(moment)
        c = self.weight
        b = self.bias
        mean_y = c @ mean_z + b
        cov_y = c @ cov_z @ c.T + self.obs_var * jnp.eye(c.shape[0], dtype=cov_z.dtype)
        return tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)

    def initialize(self, t, y, u, c):
        del t, y, u, c
        return self


class LinearStateMap(StateMap):
    """Learnable linear discrete map: z_t = W @ z_{t-1}."""

    W: jax.Array

    def __init__(self, conf, key):
        self.conf = conf
        d = int(conf.state_dim)
        self.W = 0.9 * jnp.eye(d) + 0.01 * jr.normal(key, (d, d))

    def eval(self, z, u, c, *, key=None):
        del u, c, key
        return self.W @ z


def make_stable_A(r=0.95, angle=math.pi / 6):
    c, s = math.cos(angle), math.sin(angle)
    return r * np.array([[c, -s], [s, c]], dtype=np.float32)


def generate_pca_data(seed=0, n_trials=64, T=50, state_dim=2, obs_dim=10):
    """Generate synthetic data, then PCA-project observations."""
    rng = np.random.default_rng(seed)
    a_true = make_stable_A()
    c_true = rng.normal(size=(obs_dim, state_dim)).astype(np.float32)
    b_true = rng.normal(size=(obs_dim,)).astype(np.float32)

    z = np.zeros((n_trials, T, state_dim), dtype=np.float32)
    y = np.zeros((n_trials, T, obs_dim), dtype=np.float32)
    for n in range(n_trials):
        z_prev = rng.normal(size=(state_dim,)).astype(np.float32)
        for t in range(T):
            if t > 0:
                z_prev = (a_true @ z_prev + 0.1 * rng.normal(size=(state_dim,))).astype(
                    np.float32
                )
            z[n, t] = z_prev
            y[n, t] = (
                c_true @ z_prev + b_true + 0.5 * rng.normal(size=(obs_dim,))
            ).astype(np.float32)

    y_flat = y.reshape(-1, obs_dim)
    b_pca = y_flat.mean(0)
    y_centered = y_flat - b_pca
    _, _, vt = np.linalg.svd(y_centered, full_matrices=False)
    c_pca = vt[:state_dim].T.astype(np.float32)
    z_pca = ((y.reshape(-1, obs_dim) - b_pca) @ c_pca).reshape(
        n_trials, T, state_dim
    ).astype(np.float32)

    return z_pca, y, c_pca, b_pca.astype(np.float32)


def gradient_comparison():
    z_pca, _, _, _ = generate_pca_data()
    z_pca = jnp.asarray(z_pca)
    _, _, d = z_pca.shape

    z_prev = z_pca[:, :-1].reshape(-1, d)
    z_next = z_pca[:, 1:].reshape(-1, d)
    a_ols = (z_next.T @ z_prev) @ jnp.linalg.inv(z_prev.T @ z_prev)

    w = a_ols
    q = jnp.eye(d)
    approx = MVN(dim=d, rank=d)

    residuals = z_next - z_prev @ w.T
    grad_mse_w = -(residuals.T @ z_prev) / z_prev.shape[0]

    epsilons = [1.0, 0.1, 0.01, 0.001, 1e-4, 1e-5]

    print("eps      | cos_sim  | norm_ratio | angle_deg")
    print("---------|----------|------------|----------")

    for eps in epsilons:

        def kl_loss(w_param):
            cov_q = eps * jnp.eye(d)
            moment_q = jax.vmap(lambda m: approx.pack(m, cov_q))(z_next)
            m_p = z_prev @ w_param.T
            moment_p = jax.vmap(lambda m: approx.pack(m, q))(m_p)
            kl_vals = jax.vmap(approx.kl)(moment_q, moment_p)
            return jnp.mean(kl_vals)

        grad_kl_w = jax.grad(kl_loss)(w)

        g1 = grad_kl_w.ravel()
        g2 = grad_mse_w.ravel()
        cos_sim = float(jnp.dot(g1, g2) / (jnp.linalg.norm(g1) * jnp.linalg.norm(g2)))
        norm_ratio = float(jnp.linalg.norm(g1) / jnp.linalg.norm(g2))
        angle = float(jnp.degrees(jnp.arccos(jnp.clip(cos_sim, -1, 1))))

        print(f"{eps:<8g} | {cos_sim:>8.5f} | {norm_ratio:>10.4f} | {angle:>9.3f}")


def nofilt_training_comparison():
    """Section 2: NOFILT end-to-end — train dynamics, compare to OLS."""
    print("\n=== Section 2: NOFILT end-to-end training ===")

    n_trials, T, state_dim, obs_dim = 256, 50, 2, 10
    z_pca, y, c_pca, b_pca = generate_pca_data(
        seed=0, n_trials=n_trials, T=T, state_dim=state_dim, obs_dim=obs_dim
    )

    z_prev = jnp.asarray(z_pca[:, :-1].reshape(-1, state_dim))
    z_next = jnp.asarray(z_pca[:, 1:].reshape(-1, state_dim))
    a_ols = (z_next.T @ z_prev) @ jnp.linalg.inv(z_prev.T @ z_prev)

    conf = OmegaConf.create(
        {
            "mode": "nofilt",
            "state_dim": state_dim,
            "observation_dim": obs_dim,
            "seed": 0,
            "mc_size": 8,
            "approx": "MVN",
            "approx_kwargs": {},
            "state_map": "LinearStateMap",
            "stepper": "DiscreteStepper",
            "dropout": 0.0,
            "nofilt_eps": 1e-6,
            "dyn_conf": {
                "system_type": "discrete",
                "state_noise": 0.1,
                "input_dim": 0,
                "context_dim": 0,
            },
            "enc_conf": {
                "alpha_encoder": "PCAEncoder",
            },
            "obs_conf": {
                "model": "PCAObservation",
                "obs_var": 1e-4,
            },
        }
    )

    model = XFADS(conf, jr.key(0))

    c_enc = jnp.asarray(np.array(c_pca, copy=True))
    b_enc = jnp.asarray(np.array(b_pca, copy=True))
    c_obs = jnp.asarray(np.array(c_pca, copy=True))
    b_obs = jnp.asarray(np.array(b_pca, copy=True))
    model = eqx.tree_at(
        lambda m: (
            m.alpha_encoder.weight,
            m.alpha_encoder.bias,
            m.observation.weight,
            m.observation.bias,
        ),
        model,
        (c_enc, b_enc, c_obs, b_obs),
    )

    times = jnp.tile(jnp.arange(T)[None, :], (n_trials, 1))
    observations = jnp.asarray(y)
    controls = jnp.zeros((n_trials, T, 0))
    covariates = jnp.zeros((n_trials, T, 0))
    data = (times, observations, controls, covariates)

    trainer_conf = OmegaConf.create(
        {
            "seed": 0,
            "learning_rate": 1e-2,
            "min_epoch": 300,
            "max_epoch": 300,
            "patience": 100,
            "batch_size": 64,
            "validation_size": 32,
            "weight_decay": 0.0,
            "freeze_paths": ["alpha_encoder", "observation", "noise_free"],
        }
    )

    trained = train(model, data, conf=trainer_conf)

    w_learned = np.asarray(trained.state_map.W)
    w_dist = float(np.linalg.norm(w_learned - np.asarray(a_ols), ord="fro"))

    z_prev_np = np.asarray(z_pca[:, :-1])
    z_next_np = np.asarray(z_pca[:, 1:])

    pred_ols = np.einsum("ij,ntj->nti", np.asarray(a_ols), z_prev_np)
    rmse_ols = float(np.sqrt(np.mean(np.sum((z_next_np - pred_ols) ** 2, axis=-1))))

    pred_xfads = np.einsum("ij,ntj->nti", w_learned, z_prev_np)
    rmse_xfads = float(
        np.sqrt(np.mean(np.sum((z_next_np - pred_xfads) ** 2, axis=-1)))
    )

    print(f"OLS pred RMSE:    {rmse_ols:.6f}")
    print(f"XFADS pred RMSE:  {rmse_xfads:.6f}")
    print(f"||W_xfads - A_ols||_F: {w_dist:.6f}")
    print(f"\nA_ols:\n{np.asarray(a_ols)}")
    print(f"W_learned:\n{w_learned}")


def main():
    print("=== Section 1: Gradient equivalence ===")
    gradient_comparison()
    print()
    nofilt_training_comparison()


if __name__ == "__main__":
    main()
