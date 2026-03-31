"""Direct verification of PCA dynamics gradient equivalence.

Tests the mathematical claim: KL gradient w.r.t. dynamics W converges
to MSE gradient as posterior variance -> 0.

No encoder, no XFADS pipeline - pure math verification.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxfads.distributions import MVN


def make_stable_A(r=0.95, angle=math.pi / 6):
    c, s = math.cos(angle), math.sin(angle)
    return r * np.array([[c, -s], [s, c]], dtype=np.float32)


def generate_pca_coords(seed=0, n_trials=64, T=50, state_dim=2, obs_dim=10):
    """Generate synthetic data and return PCA coordinates."""
    rng = np.random.default_rng(seed)
    A_true = make_stable_A()
    C_true = rng.normal(size=(obs_dim, state_dim)).astype(np.float32)
    b_true = rng.normal(size=(obs_dim,)).astype(np.float32)

    z = np.zeros((n_trials, T, state_dim), dtype=np.float32)
    y = np.zeros((n_trials, T, obs_dim), dtype=np.float32)
    for n in range(n_trials):
        z_prev = rng.normal(size=(state_dim,)).astype(np.float32)
        for t in range(T):
            if t > 0:
                z_prev = (A_true @ z_prev + 0.1 * rng.normal(size=(state_dim,))).astype(
                    np.float32
                )
            z[n, t] = z_prev
            y[n, t] = (
                C_true @ z_prev + b_true + 0.5 * rng.normal(size=(obs_dim,))
            ).astype(np.float32)

    y_flat = y.reshape(-1, obs_dim)
    b_pca = y_flat.mean(0)
    y_c = y_flat - b_pca
    _, _, vt = np.linalg.svd(y_c, full_matrices=False)
    C_pca = vt[:state_dim].T
    z_pca = ((y.reshape(-1, obs_dim) - b_pca) @ C_pca).reshape(
        n_trials, T, state_dim
    ).astype(np.float32)
    return z_pca


def main():
    z_pca = jnp.asarray(generate_pca_coords())
    _, T, D = z_pca.shape

    z_prev = z_pca[:, :-1].reshape(-1, D)
    z_next = z_pca[:, 1:].reshape(-1, D)
    A_ols = (z_next.T @ z_prev) @ jnp.linalg.inv(z_prev.T @ z_prev)

    W = A_ols
    Q = jnp.eye(D)
    approx = MVN(dim=D, rank=D)
    N_pairs = z_prev.shape[0]

    residuals = z_next - z_prev @ W.T
    grad_mse_W = -(residuals.T @ z_prev) / N_pairs

    epsilons = [1.0, 0.1, 0.01, 0.001, 1e-4, 1e-5]

    print("eps      | cos_sim  | norm_ratio | angle_deg")
    print("---------|----------|------------|----------")

    for eps in epsilons:

        def kl_loss(W_param):
            cov_q = eps * jnp.eye(D)
            moment_q = jax.vmap(lambda mq: approx.pack(mq, cov_q))(z_next)
            m_p = z_prev @ W_param.T
            moment_p = jax.vmap(lambda mp: approx.pack(mp, Q))(m_p)
            kl_vals = jax.vmap(approx.kl)(moment_q, moment_p)
            return jnp.mean(kl_vals)

        grad_kl_W = jax.grad(kl_loss)(W)

        g1 = grad_kl_W.ravel()
        g2 = grad_mse_W.ravel()
        cos_sim = float(jnp.dot(g1, g2) / (jnp.linalg.norm(g1) * jnp.linalg.norm(g2)))
        norm_ratio = float(jnp.linalg.norm(g1) / jnp.linalg.norm(g2))
        angle = float(jnp.degrees(jnp.arccos(jnp.clip(cos_sim, -1, 1))))

        print(f"{eps:<8g} | {cos_sim:>8.5f} | {norm_ratio:>10.4f} | {angle:>9.3f}")


if __name__ == "__main__":
    main()
