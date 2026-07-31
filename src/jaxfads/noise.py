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

    def moment(self) -> Array:
        """Decode the free Q representation into Approx moment parameters."""
        return self.approx.canon_to_moment(self.approx.free_to_canon(self.free))

    def predictive_moment(self, z: Array) -> Array:
        """Delegate transition prediction under this component's noise state."""
        return self.approx.predictive_moment(z, self.moment())

    def collect_minibatch_stat(self, moment: Array, transition_stat: Any) -> Any:
        """Return one additive Q-statistic delta, or ``None`` if unsupported."""
        strategy = self.mstep_strategy
        if strategy is None:
            return None
        return strategy.collect_minibatch_stat(self, moment, transition_stat)

    def accumulate_minibatch_stat(self, total: Any, delta: Any) -> Any:
        """Add fixed-shape Q-statistic pytrees while preserving no-op ``None``."""
        if total is None:
            return delta
        if delta is None:
            return total
        return jax.tree.map(lambda left, right: left + right, total, delta)

    def mstep(self, epoch_stat: Any, *, prior: Any) -> "Noise":
        """Return an updated component from accumulated Q statistics."""
        strategy = self.mstep_strategy
        if strategy is None or epoch_stat is None:
            return self
        free = strategy.mstep(self, epoch_stat, prior=prior)
        return eqx.tree_at(lambda noise: noise.free, self, free)
