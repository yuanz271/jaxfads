"""
Abstract base interfaces for XFADS.

This module centralizes abstract base classes for dynamics and observation
components to keep concrete implementations in their respective modules.
"""

from __future__ import annotations

from abc import abstractmethod

from jax import Array

from gearax.mixin import SubclassRegistryMixin
from gearax.modules import ConfModule

from .distributions import Approx


class Dynamics(SubclassRegistryMixin, ConfModule):  # pyright: ignore[reportImplicitAbstractClass]
    """
    Abstract base class for dynamics models in XFADS.

    Defines the interface for state transition models that describe how
    the latent state evolves over time.  Concrete subclasses implement
    the deterministic transition ``forward(z, u, c)``.

    Process noise is **not** stored here — it is owned by the
    orchestrating model (``XFADS``) so that ``Dynamics`` stays a pure
    transition function.
    """

    @abstractmethod
    def forward(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        """
        Compute the deterministic part of state transition.

        Parameters
        ----------
        z : Array, shape (state_dim,)
            Current state vector.
        u : Array, shape (input_dim,)
            Control/input vector.
        c : Array, shape (covariate_dim,)
            Covariate vector.
        key : PRNGKeyArray, optional
            Random key for stochastic dynamics (e.g., dropout).

        Returns
        -------
        Array, shape (state_dim,)
            Predicted next state mean (before adding noise).
        """
        ...

    def __call__(self, *args, **kwargs) -> Array:
        """
        Convenience method to call forward().

        Returns
        -------
        Array
            Result of forward(*args, **kwargs).
        """
        return self.forward(*args, **kwargs)


class ObservationModel(SubclassRegistryMixin, ConfModule):
    """
    Abstract observation model interface.

    Observation models combine readouts and likelihoods to compute expected
    log-likelihoods and initialize observation parameters.
    """

    @abstractmethod
    def eloglik(
        self,
        key: Array,
        t: Array,
        moment: Array,
        y: Array,
        approx: type[Approx],
        mc_size: int,
    ) -> Array:
        """
        Compute expected log-likelihood for observations.
        """
        ...

    @abstractmethod
    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "ObservationModel":
        """
        Initialize observation parameters from data statistics.
        """
        ...


__all__ = ["Dynamics", "ObservationModel"]
