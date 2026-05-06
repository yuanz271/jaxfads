"""State-evolution integrators for XFADS."""

from __future__ import annotations

from jax import Array

from .base import Dynamics, Integrator


def _has_attr(conf, name: str) -> bool:
    if hasattr(conf, name):
        return True
    try:
        return name in conf
    except TypeError:
        return False


class Euler(Integrator):
    """Forward-Euler integrator."""

    dt: float

    def __init__(self, conf):
        if not _has_attr(conf, "dt"):
            raise ValueError("Euler requires dyn_conf.dt.")
        self.dt = float(conf.dt)

    def step(
        self,
        z: Array,
        u: Array,
        c: Array,
        dynamics: Dynamics,
        *,
        key=None,
    ) -> Array:
        return z + self.dt * dynamics.eval(z, u, c, key=key)


class RK4(Integrator):
    """Classical fourth-order Runge-Kutta integrator."""

    dt: float

    def __init__(self, conf):
        if not _has_attr(conf, "dt"):
            raise ValueError("RK4 requires dyn_conf.dt.")
        self.dt = float(conf.dt)

    def step(
        self,
        z: Array,
        u: Array,
        c: Array,
        dynamics: Dynamics,
        *,
        key=None,
    ) -> Array:
        dt = self.dt
        k1 = dynamics.eval(z, u, c, key=key)
        k2 = dynamics.eval(z + 0.5 * dt * k1, u, c, key=key)
        k3 = dynamics.eval(z + 0.5 * dt * k2, u, c, key=key)
        k4 = dynamics.eval(z + dt * k3, u, c, key=key)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class Identity(Integrator):
    """Pass-through integrator."""

    def __init__(self, conf):
        if _has_attr(conf, "dt"):
            raise ValueError("Identity must not receive dyn_conf.dt.")

    def step(
        self,
        z: Array,
        u: Array,
        c: Array,
        dynamics: Dynamics,
        *,
        key=None,
    ) -> Array:
        return dynamics.eval(z, u, c, key=key)


__all__ = ["Euler", "RK4", "Identity"]
