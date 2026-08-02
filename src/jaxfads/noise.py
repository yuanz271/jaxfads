"""Generic transition-noise component."""

from __future__ import annotations

import equinox as eqx
from jax import Array

from .base import Approx


class Noise(eqx.Module):
    """Transition-noise state composed with one static Approx configuration.

    ``approx`` is array-free structural distribution configuration and ``free``
    is the sole trainable transition-noise representation. Closed-form Q
    updates are trainer policies, not Noise behavior.
    """

    approx: Approx = eqx.field(static=True)
    free: Array

    def moment(self) -> Array:
        """Decode the free Q representation into Approx moment parameters."""
        return self.approx.canon_to_moment(self.approx.free_to_canon(self.free))

    def predictive_moment(self, z: Array) -> Array:
        """Delegate transition prediction under this component's noise state."""
        return self.approx.predictive_moment(z, self.moment())
