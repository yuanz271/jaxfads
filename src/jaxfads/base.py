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
    and moment parameterizations, sampling methods, and other utilities.

    Concrete subclasses are instantiated with family-specific parameters
    (e.g., ``MVN(rank=0)``).  All methods are instance methods so that
    the distribution configuration is carried by the instance.
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
    def moment_size(self, state_dim: int) -> int:
        """
        Get the moment parameter size for given state dimension.

        Parameters
        ----------
        state_dim : int
            Dimensionality of the state space.

        Returns
        -------
        int
            Total number of moment parameters.
        """
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
    def moment_to_canon(self, moment: Array) -> tuple[Array, Array]:
        """
        Convert moment parameters to canonical mean and covariance.

        Parameters
        ----------
        moment : Array
            Moment parameter vector.

        Returns
        -------
        mean : Array, shape (D,)
            Mean vector.
        cov : Array
            Covariance — vector (D,) for diagonal, matrix (D, D) otherwise.
        """
        ...

    @abstractmethod
    def canon_to_moment(self, mean: Array, cov: Array) -> Array:
        """
        Convert canonical mean and covariance to moment parameters.

        Parameters
        ----------
        mean : Array, shape (D,)
            Mean vector.
        cov : Array
            Covariance matrix (D, D) or diagonal vector (D,).

        Returns
        -------
        Array
            Moment parameter vector.
        """
        ...

    @abstractmethod
    def full_cov(self, cov: Array) -> Array:
        """
        Convert covariance parameterization to full covariance matrix.

        Parameters
        ----------
        cov : Array
            Covariance parameters (may be diagonal, low-rank, etc.).

        Returns
        -------
        Array, shape (D, D)
            Full covariance matrix.
        """
        ...

    @abstractmethod
    def constrain_moment(self, unconstrained: Array) -> Array:
        """
        Transform unconstrained parameters to valid moment parameters.

        Parameters
        ----------
        unconstrained : Array
            Unconstrained parameter vector from optimization.

        Returns
        -------
        Array
            Valid moment parameters satisfying distribution constraints
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
    def unconstrain_moment(self, moment: Array) -> Array:
        """
        Transform valid moment parameters to unconstrained space.

        Inverse of :meth:`constrain_moment`.

        Parameters
        ----------
        moment : Array
            Valid moment parameter vector.

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
        Create unconstrained noise moment parameters.

        Each exponential-family subclass decides how to interpret
        *scale* for its own parameterisation and returns an array
        suitable for storage in
        ``XFADS.unconstrained_noise_moment``.

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
            Unconstrained moment parameters.
        """
        ...

    @abstractmethod
    def predict_moment(self, loc: Array, noise_moment: Array) -> Array:
        """
        Mean parameter of the predictive distribution p(z | loc, noise).

        Computes ``E_{p(z|loc,noise)}[T(z)]`` — the expected sufficient
        statistics (mean parameter) of the one-step-ahead distribution.
        The result lives in mean-parameter space where linear averaging
        is valid.

        Parameters
        ----------
        loc : Array, shape (state_dim,)
            Dynamics output (deterministic location).
        noise_moment : Array
            Noise parameters in the code's moment format.

        Returns
        -------
        Array
            Mean parameter vector of the predictive distribution.
        """
        ...

    @abstractmethod
    def mean_param_to_moment(self, mean_param: Array) -> Array:
        """
        Convert mean parameter to the code's moment format.

        Inverse mapping from the mean-parameter space (where averaging
        is valid) to the moment format used by the rest of the codebase.

        Parameters
        ----------
        mean_param : Array
            Mean parameter vector (output of :meth:`predict_moment`).

        Returns
        -------
        Array
            Moment vector compatible with :meth:`moment_to_natural`.
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
        approx: Approx,
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


__all__ = ["Approx", "Dynamics", "ObservationModel"]
