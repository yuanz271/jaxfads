"""State-evolution steppers for XFADS."""

from __future__ import annotations

from jax import Array

from .base import StateMap, Stepper


def _has_attr(conf, name: str) -> bool:
    if hasattr(conf, name):
        return True
    try:
        return name in conf
    except TypeError:
        return False


class EulerStepper(Stepper):
    """Forward-Euler stepper for continuous-time state maps."""

    dt: float

    def __init__(self, conf, key: Array):
        del key
        self.conf = conf
        if str(conf.system_type) != "continuous":
            raise ValueError("EulerStepper requires dyn_conf.system_type='continuous'.")
        if not _has_attr(conf, "dt"):
            raise ValueError("EulerStepper requires dyn_conf.dt.")
        self.dt = float(conf.dt)

    def step(
        self,
        z: Array,
        u: Array,
        c: Array,
        state_map: StateMap,
        *,
        key=None,
    ) -> Array:
        return z + self.dt * state_map.eval(z, u, c, key=key)


class RK4Stepper(Stepper):
    """Classical fourth-order Runge-Kutta stepper for continuous-time maps."""

    dt: float

    def __init__(self, conf, key: Array):
        del key
        self.conf = conf
        if str(conf.system_type) != "continuous":
            raise ValueError("RK4Stepper requires dyn_conf.system_type='continuous'.")
        if not _has_attr(conf, "dt"):
            raise ValueError("RK4Stepper requires dyn_conf.dt.")
        self.dt = float(conf.dt)

    def step(
        self,
        z: Array,
        u: Array,
        c: Array,
        state_map: StateMap,
        *,
        key=None,
    ) -> Array:
        dt = self.dt
        k1 = state_map.eval(z, u, c, key=key)
        k2 = state_map.eval(z + 0.5 * dt * k1, u, c, key=key)
        k3 = state_map.eval(z + 0.5 * dt * k2, u, c, key=key)
        k4 = state_map.eval(z + dt * k3, u, c, key=key)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class DiscreteStepper(Stepper):
    """Pass-through stepper for discrete-time state maps."""

    def __init__(self, conf, key: Array):
        del key
        self.conf = conf
        if str(conf.system_type) != "discrete":
            raise ValueError(
                "DiscreteStepper requires dyn_conf.system_type='discrete'."
            )
        if _has_attr(conf, "dt"):
            raise ValueError("DiscreteStepper must not receive dyn_conf.dt.")

    def step(
        self,
        z: Array,
        u: Array,
        c: Array,
        state_map: StateMap,
        *,
        key=None,
    ) -> Array:
        return state_map.eval(z, u, c, key=key)


__all__ = ["EulerStepper", "RK4Stepper", "DiscreteStepper"]
