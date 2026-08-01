"""Generic transition-noise component and optional Approx M-step strategies."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
from jax import Array

from .base import Approx


class Noise(eqx.Module):
    """Transition-noise state composed with one static Approx configuration.

    ``approx`` is array-free structural distribution configuration. ``free``
    is the sole trainable transition-noise representation. Optional MAP
    behavior is supplied by an exact concrete-Approx-class strategy
    registration; without one, Noise remains fully usable for prediction and
    SGD-managed Q but contributes no M-step statistics.
    """

    approx: Approx = eqx.field(static=True)
    q_scale: float = eqx.field(static=True)
    q_prior_fraction: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    mstep_enabled: bool = eqx.field(static=True)
    free: Array

    _mstep_strategies: ClassVar[dict[type[Approx], Any]] = {}

    @classmethod
    def register_mstep(cls, approx_cls: type[Approx], strategy: Any) -> None:
        """Register one optional Q-M-step strategy for an exact Approx class."""
        cls._mstep_strategies[approx_cls] = strategy

    @property
    def mstep_strategy(self) -> Any | None:
        """Strategy for the exact concrete Approx class, without MRO fallback."""
        return self._mstep_strategies.get(type(self.approx))

    @property
    def supports_mstep(self) -> bool:
        return self.mstep_strategy is not None

    @property
    def mstep_active(self) -> bool:
        return self.mstep_enabled and self.supports_mstep

    def moment(self) -> Array:
        """Decode the free Q representation into Approx moment parameters."""
        return self.approx.canon_to_moment(self.approx.free_to_canon(self.free))

    def predictive_moment(self, z: Array) -> Array:
        """Delegate transition prediction under this component's noise state."""
        return self.approx.predictive_moment(z, self.moment())

    def batch_stat(self, t, y, u, c, moment, transition_stat, approx) -> Any:
        """Return one additive Q statistic, or ``None`` when inactive."""
        if not self.mstep_active:
            return None
        return self.mstep_strategy.batch_stat(
            self, t, y, u, c, moment, transition_stat, approx
        )

    def frozen_paths(self) -> list[str]:
        return ["free"] if self.mstep_active else []

    def accumulate_stat(self, total: Any, delta: Any) -> Any:
        """Add fixed-shape Q-statistic pytrees while preserving no-op ``None``."""
        if total is None:
            return delta
        if delta is None:
            return total
        return jax.tree.map(lambda left, right: left + right, total, delta)

    def mstep(self, stat: Any) -> "Noise":
        """Return an updated component from accumulated Q statistics."""
        if not self.mstep_active or stat is None:
            return self
        free = self.mstep_strategy.mstep(self, stat)
        return eqx.tree_at(lambda noise: noise.free, self, free)
