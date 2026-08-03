"""Shared utilities for standalone research benchmarks."""

from __future__ import annotations

import jax
from jax import numpy as jnp

DT = 0.01


def lorenz_rhs(
    state: jax.Array,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> jax.Array:
    """Continuous-time Lorenz vector field."""
    x, y, z = state[0], state[1], state[2]
    return jnp.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def rk4_step(state: jax.Array, dt: float) -> jax.Array:
    """One fourth-order Runge–Kutta step for the Lorenz vector field."""
    k1 = lorenz_rhs(state)
    k2 = lorenz_rhs(state + 0.5 * dt * k1)
    k3 = lorenz_rhs(state + 0.5 * dt * k2)
    k4 = lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
