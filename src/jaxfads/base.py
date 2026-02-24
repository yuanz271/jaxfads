"""
Abstract base interfaces for XFADS.

This module centralizes abstract base classes for dynamics, observation,
and approximate-distribution components to keep concrete implementations
in their respective modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from jax import Array

from gearax.mixin import SubclassRegistryMixin
from gearax.modules import ConfModule


class Approx(SubclassRegistryMixin, ABC):
    """
    Abstract base class for exponential family approximations in XFADS.

    This class defines the interface for exponential family distributions
    used in variational inference, providing conversions between natural
    and mean parameterizations, sampling methods, and other utilities.

    Concrete subclasses are instantiated with family-specific parameters
    (e.g., ``MVN(dim=3, rank=0)``).  All methods are instance methods so
    that the distribution configuration is carried by the instance.
    """

    @abstractmethod
    def natural_to_mean(self, natural: Array) -> Array:
        """
        Convert natural parameters to mean parameters.

        Parameters
        ----------
        natural : Array
            Natural parameter vector of the exponential-family distribution.

        Returns
        -------
        Array
            Corresponding mean parameter vector.

        Notes
        -----
        For exponential families, the mean parameters are the expected
        values of the sufficient statistics under the distribution.
        """
        ...

    @abstractmethod
    def mean_to_natural(self, mean: Array) -> Array:
        """
        Convert mean parameters to natural parameters.

        Parameters
        ----------
        mean : Array
            Mean parameter vector of the exponential-family distribution.

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
    def sample_by_mean(self, key: Array, mean: Array, mc_size: int) -> Array:
        """
        Generate samples from the distribution using mean parameters.

        Parameters
        ----------
        key : Array
            JAX PRNG key for randomness.
        mean : Array
            Mean parameter vector defining the distribution.
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
    def param_size(self, state_dim: int) -> int:
        """
        Get the natural parameter size for given state dimension.

        Parameters
        ----------
        state_dim : int
            Dimensionality of the state space.

        Returns
        -------
        int
            Total number of natural parameters.
        """
        ...

    @abstractmethod
    def kl(self, mean1: Array, mean2: Array) -> Array:
        """
        Compute KL divergence between two distributions.

        Parameters
        ----------
        mean1 : Array
            Mean parameters of the first distribution.
        mean2 : Array
            Mean parameters of the second distribution.

        Returns
        -------
        Array
            KL divergence KL(p1 || p2) where p1 and p2 are parameterized
            by mean1 and mean2 respectively.
        """
        ...

    @abstractmethod
    def constrain_mean(self, unconstrained: Array) -> Array:
        """
        Transform unconstrained parameters to valid mean parameters.

        Parameters
        ----------
        unconstrained : Array
            Unconstrained parameter vector from optimization.

        Returns
        -------
        Array
            Valid mean parameters satisfying distribution constraints
            (e.g., positive definite covariance).
        """
        ...

    @abstractmethod
    def constrain_natural(self, unconstrained: Array) -> Array:
        """
        Transform unconstrained parameters to valid natural parameters.

        Parameters
        ----------
        unconstrained : Array
            Unconstrained parameter vector from optimization.

        Returns
        -------
        Array
            Valid natural parameters satisfying distribution constraints
            (e.g., negative definite precision for Gaussians).
        """
        ...

    @abstractmethod
    def unconstrain_natural(self, natural: Array) -> Array:
        """
        Transform natural parameters to unconstrained space.

        Parameters
        ----------
        natural : Array
            Valid natural parameter vector.

        Returns
        -------
        Array
            Unconstrained parameters suitable for optimization.
        """
        ...

    @abstractmethod
    def unconstrain_mean(self, mean: Array) -> Array:
        """
        Transform valid mean parameters to unconstrained space.

        Inverse of :meth:`constrain_mean`.

        Parameters
        ----------
        mean : Array
            Valid mean parameter vector.

        Returns
        -------
        Array
            Unconstrained parameters suitable for optimization.
        """
        ...

    @abstractmethod
    def prior_natural(self, state_dim: int) -> Array:
        """
        Get natural parameters for the standard normal prior.

        Parameters
        ----------
        state_dim : int
            Dimensionality of the state space.

        Returns
        -------
        Array
            Natural parameters for N(0, I) prior distribution.
        """
        ...

    @abstractmethod
    def init_noise(self, scale: float, state_dim: int) -> Array:
        """
        Create unconstrained noise mean parameters.

        Each exponential-family subclass decides how to interpret
        *scale* for its own parameterisation and returns an array
        suitable for storage in
        ``XFADS.unconstrained_noise_mean``.

        Parameters
        ----------
        scale : float
            Noise scale (interpretation is family-specific,
            e.g. variance for Gaussians).
        state_dim : int
            Dimensionality of the latent state.

        Returns
        -------
        Array
            Unconstrained mean parameters.
        """
        ...

    @abstractmethod
    def predict_mean(self, locs: Array, noise_mean: Array) -> Array:
        """
        Predict structured mean from a batch of dynamics locations.

        Computes the expected sufficient statistics ``E[T(z)]`` for each
        location, averages in the appropriate mean-parameter space, and
        returns the result in the structured mean format used by the rest
        of the codebase.

        Samples containing non-finite values are masked out.  If every
        sample is non-finite the result may itself be non-finite; the
        caller is responsible for fallback logic.

        Parameters
        ----------
        locs : Array, shape (N, state_dim)
            Dynamics output locations for N Monte Carlo samples.
        noise_mean : Array
            Noise parameters in the mean parameter format.  May be
            empty (``jnp.array([])``) for families without a separate
            dispersion parameter.

        Returns
        -------
        Array
            Structured mean parameter vector of the predictive
            distribution.
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
        mean: Array,
        y: Array,
        approx: Approx,
        mc_size: int,
    ) -> Array:
        """
        Compute expected log-likelihood for observations.
        """
        ...

    @abstractmethod
    def initialize(self, t: Array, y: Array, u: Array, c: Array) -> "Observation":
        """
        Initialize observation parameters from data statistics.
        """
        ...


__all__ = ["Approx", "Dynamics", "Observation"]
