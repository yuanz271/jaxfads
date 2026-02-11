"""Verify that synthesis and ToyDynamics produce identical trajectories."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Make examples/ importable so we can ``from toy_example import ...``.
_examples_dir = str(Path(__file__).resolve().parent)
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from toy_example import ToyDynamics, rk4_step, simulate_vdp  # noqa: E402


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    max_abs_err: float


def max_abs(x) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(x))).item())


def assert_close(a, b, *, atol=1e-6, rtol=1e-6, name="") -> CheckResult:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    err = max_abs(a - b)
    ok = bool(jnp.allclose(a, b, atol=atol, rtol=rtol).item())
    if not ok:
        raise AssertionError(f"{name} failed: max_abs_err={err:.3e}")
    return CheckResult(name=name, ok=True, max_abs_err=err)


def reconstruct_z0_and_eps(
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


def check_synthesis_step_consistency() -> list[CheckResult]:
    key = jr.key(0)
    n_trials = 8
    n_steps = 5

    mu = 2.0
    dt = 0.02
    init_radius = 2.0
    init_radius_jitter = 0.2
    latent_noise = 0.0

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

    z0, eps = reconstruct_z0_and_eps(
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
        results.append(assert_close(zs[:, t], z_next, name=f"synthesis_step[t={t}]"))
        z = z_next

    return results


def check_toydynamics_matches_rk4() -> list[CheckResult]:
    class Cfg:
        def __init__(self, mu, dt, cov, state_dim=2, input_dim=0, context_dim=0):
            self.mu = mu
            self.dt = dt
            self.cov = cov
            self.state_dim = state_dim
            self.input_dim = input_dim
            self.context_dim = context_dim

    mu = 2.0
    dt = 0.02
    cov = 0.0

    dyn = ToyDynamics(Cfg(mu=mu, dt=dt, cov=cov), key=jr.key(1))

    key = jr.key(2)
    z = jr.normal(key, (32, 2))
    u0 = jnp.zeros((0,))
    c0 = jnp.zeros((0,))

    a = jax.vmap(lambda s: dyn.forward(s, u0, c0))(z)
    b = jax.vmap(lambda s: rk4_step(s, dt, mu=mu))(z)
    return [assert_close(a, b, name="toydynamics_vs_rk4")]


def main() -> None:
    results = []
    results.extend(check_synthesis_step_consistency())
    results.extend(check_toydynamics_matches_rk4())

    print("OK: consistency checks passed")
    for r in results:
        print(f"- {r.name}: max_abs_err={r.max_abs_err:.3e}")


if __name__ == "__main__":
    main()
