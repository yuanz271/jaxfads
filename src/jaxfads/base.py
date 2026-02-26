"""
Abstract base interfaces for XFADS.

This module centralizes abstract base classes for dynamics, observation,
and approximate-distribution components to keep concrete implementations
in their respective modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jax import Array
from jaxtyping import PyTree

from gearax.mixin import SubclassRegistryMixin
from gearax.modules import ConfModule


class Approx(SubclassRegistryMixin, ABC):
    """
    Abstract base class for exponential family approximations in XFADS.

    This class defines the interface for exponential family distributions
    used in variational inference, providing conversions between natural
    and moment parameterizations, sampling methods, and other utilities.

    Implementations are instantiated with family-specific parameters.
    All methods are instance methods so that the distribution
    configuration is carried by the instance.
    """

    @abstractmethod
    def natural_to_moment(self, natural: Array) -> Array:
        """
        Convert natural parameters to moment parameters.

        Parameters
        ----------
        natural : Array
            Natural parameter vector of the exponential-family distribution.

        Returns
        -------
        Array
            Corresponding moment parameter vector.

        Notes
        -----
        For exponential families, the moment parameters are the expected
        values of the sufficient statistics under the distribution.
        """
        ...

    @abstractmethod
    def moment_to_natural(self, moment: Array) -> Array:
        """
        Convert moment parameters to natural parameters.

        Parameters
        ----------
        moment : Array
            Moment parameter vector of the exponential-family distribution.

        Returns
        -------
        Array
            Corresponding natural parameter vector.

        Notes
        -----
        The natural parameters are the canonical parameterization for
        exponential families, often providing better numerical properties
        for optimization and inference.
        """
        ...

    @abstractmethod
    def sample_by_moment(self, key: Array, moment: Array, mc_size: int) -> Array:
        """
        Generate samples from the distribution using moment parameters.

        Parameters
        ----------
        key : Array
            JAX PRNG key for randomness.
        moment : Array
            Moment parameter vector defining the distribution.
        mc_size : int
            Number of Monte Carlo samples to draw.

        Returns
        -------
        Array, shape (mc_size, D)
            Samples drawn from the distribution.

        Notes
        -----
        Uses reparameterization trick when possible for gradient estimation
        compatibility in variational inference.
        """
        ...

    @abstractmethod
    def param_size(self) -> int:
        """Return the flat natural-parameter size for this instance."""
        ...

    @abstractmethod
    def kl(self, moment1: Array, moment2: Array) -> Array:
        """
        Compute KL divergence between two distributions.

        Parameters
        ----------
        moment1 : Array
            Moment parameters of the first distribution.
        moment2 : Array
            Moment parameters of the second distribution.

        Returns
        -------
        Array
            KL divergence KL(p1 || p2) where p1 and p2 are parameterized
            by moment1 and moment2 respectively.
        """
        ...

    @abstractmethod
    def free_to_canon(self, free: Any) -> Any:
        """
        Transform free-form pytree to valid canon pytree.

        Parameters
        ----------
        free : pytree
            Free-form (unconstrained) parameter pytree from optimization.

        Returns
        -------
        pytree
            Valid canon parameters with constraints satisfied
            (e.g., positive covariance components).
        """
        ...

    @abstractmethod
    def canon_to_moment(self, canon: Any) -> Array:
        """
        Convert canon pytree to flat moment parameters.

        Direct path avoiding the roundtrip through natural parameters,
        which may be numerically unstable (e.g. matrix inversion).

        Parameters
        ----------
        canon : pytree
            Valid canon parameter pytree (output of
            :meth:`free_to_canon`).

        Returns
        -------
        Array
            Flat moment parameters (expected sufficient statistics).
        """
        ...

    @abstractmethod
    def moment_to_canon(self, moment: Array) -> Any:
        """
        Convert flat moment parameters to canon pytree.

        Inverse of :meth:`canon_to_moment`.

        Parameters
        ----------
        moment : Array
            Flat moment parameters (expected sufficient statistics).

        Returns
        -------
        pytree
            Valid canon parameter pytree.
        """
        ...

    @abstractmethod
    def canon_to_free(self, canon: Any) -> Any:
        """
        Transform valid canon pytree to free-form pytree.

        Inverse of :meth:`free_to_canon`.

        Parameters
        ----------
        canon : pytree
            Valid canon parameter pytree.

        Returns
        -------
        pytree
            Free-form (unconstrained) parameter pytree suitable for
            optimization.
        """
        ...

    @abstractmethod
    def free_from_kw(self, **kwargs) -> PyTree:
        """
        Create free-form parameters from a serializable spec.

        Each implementation defines which keyword arguments it accepts.
        The returned pytree is in free-form (unconstrained), suitable
        for storage on ``XFADS`` and optimization by optax.

        Parameters
        ----------
        **kwargs
            Family-specific keyword arguments.

        Returns
        -------
        pytree
            Free-form parameter pytree.
        """
        ...

    @abstractmethod
    def predictive_moment(self, z: Array, noise: Array) -> Array:
        """
        Moment parameters for a single state realization.

        Computes ``E[T(z)]`` for one dynamics output and transition
        noise parameters. The output is a flat moment-parameter vector
        in the sufficient-statistic layout ``E[T(z)]``.

        Parameters
        ----------
        z : Array, shape (state_dim,)
            Single state realization (dynamics output).
        noise : Array
            Additional transition parameters (e.g. dispersion).
            Pass ``jnp.array([])`` for families without separate
            dispersion.

        Returns
        -------
        Array
            Expanded sufficient statistics (flat).
        """
        ...


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


class Observation(SubclassRegistryMixin, ConfModule):
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
        approx: Approx,
        mc_size: int,
    ) -> Array:
        """Compute expected log-likelihood for observations.

        Parameters
        ----------
        moment : Array
            Exponential-family moment parameters of the latent state
            approximation ``q(z_t)`` (i.e. a packed vector ``μ = E[T(z_t)]``).
        """
        ...

    @abstractmethod
    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "Observation":
        """
        Initialize observation parameters from data statistics.
        """
        ...


__all__ = ["Approx", "Dynamics", "Observation"]
