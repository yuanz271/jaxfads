"""
Dynamics models for XFADS.

This module implements various dynamics models for state transitions in
XFADS. It provides abstract interfaces for dynamics and noise models,
along with concrete implementations for common cases.
"""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Protocol

import jax
from jax import Array, numpy as jnp, random as jrnd
import equinox as eqx
from gearax.modules import ConfModule
from gearax.mixin import SubclassRegistryMixin

from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx


class Noise(Protocol):
    """
    Protocol for noise models in dynamics systems.

    Defines the interface that all noise models must implement to be
    compatible with XFADS dynamics models.
    """
    def cov(self) -> Array:
        """
        Get the noise covariance matrix.

        Returns
        -------
        Array
            Noise covariance matrix or covariance parameters.
        """
        ...


def predict_moment(
    z: Array, u: Array, c: Array, f, noise: Noise, approx: type[Approx], *, key=None
) -> Array:
    """
    Predict moment parameters for next state given current state.

    Computes the moment parameters of p(z_{t+1} | z_t, u_t, c_t) by
    applying the dynamics function and incorporating process noise.

    Parameters
    ----------
    z : Array, shape (state_dim,)
        Current state vector.
    u : Array, shape (input_dim,)
        Control/input vector.
    c : Array, shape (covariate_dim,)
        Covariate vector.
    f : Callable
        Dynamics function mapping (z, u, c) -> z_next.
    noise : Noise
        Noise model providing covariance structure.
    approx : type[Approx]
        Exponential family approximation class.
    key : PRNGKeyArray, optional
        Random key for stochastic dynamics.

    Returns
    -------
    Array
        Moment parameters of the predictive distribution p(z_{t+1} | z_t, u_t, c_t).

    Notes
    -----
    The predictive distribution is constructed as:
    p(z_{t+1} | z_t) = N(f(z_t, u_t, c_t), Σ_noise)

    where f is the deterministic dynamics and Σ_noise is the process noise.
    """
    ztp1 = f(z, u, c, key=key)
    M2 = approx.noise_moment(noise.cov())
    moment = approx.canon_to_moment(ztp1, M2)
    return moment


def sample_expected_moment(
    key: Array,
    moment: Array,
    u: Array,
    c: Array,
    f: Callable,
    noise: Noise,
    approx: type[Approx],
    mc_size: int,
) -> Array:
    """
    Compute expected moment parameters via Monte Carlo sampling.

    Approximates E_{p(z_t)}[μ(z_t, u_t, c_t)] where μ(·) gives the moment
    parameters of the predictive distribution p(z_{t+1} | z_t, u_t, c_t).
    This expectation is intractable for nonlinear dynamics, so we use
    Monte Carlo approximation.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for sampling.
    moment : Array
        Moment parameters of current state distribution p(z_t).
    u : Array, shape (input_dim,)
        Control/input vector.
    c : Array, shape (covariate_dim,)
        Covariate vector.
    f : Callable
        Dynamics function.
    noise : Noise
        Process noise model.
    approx : type[Approx]
        Exponential family approximation.
    mc_size : int
        Number of Monte Carlo samples.

    Returns
    -------
    Array
        Expected moment parameters E_{p(z_t)}[μ(z_t, u_t, c_t)].

    Notes
    -----
    The Monte Carlo approximation is:
    E[μ(z_t)] ≈ (1/K) Σ_{k=1}^K μ(z_t^{(k)})

    where z_t^{(k)} ~ p(z_t) are samples from the current state distribution.
    """
    key, subkey = jrnd.split(key)
    z = approx.sample_by_moment(subkey, moment, mc_size)
    u = jnp.broadcast_to(u, shape=(mc_size,) + u.shape)
    c = jnp.broadcast_to(c, shape=(mc_size,) + c.shape)
    f_vmap_sample_axis = jax.vmap(
        partial(predict_moment, f=f, noise=noise, approx=approx, key=key),
        in_axes=(0, 0, 0),
    )
    moment = jnp.mean(f_vmap_sample_axis(z, u, c), axis=0)
    return moment


class DiagGaussian(eqx.Module, strict=True):
    """
    Diagonal Gaussian noise model for dynamics systems.

    Implements process noise with diagonal covariance structure, assuming
    independence between state dimensions. More efficient than full
    covariance but less expressive.

    Parameters
    ----------
    cov : ArrayLike
        Initial covariance value (Array applied to all dimensions).
    size : int
        Dimensionality of the noise (should match state dimension).

    Attributes
    ----------
    unconstrained_cov : Array, shape (size,)
        Unconstrained covariance parameters for optimization.

    Notes
    -----
    The covariance is parameterized in unconstrained space to ensure
    positive values during optimization. The actual covariance is
    obtained via constrain_positive() transformation.
    """
    unconstrained_cov: Array

    def __init__(self, cov: Array, size: int):
        self.unconstrained_cov = jnp.full(size, fill_value=unconstrain_positive(cov))

    def cov(self) -> Array:
        """
        Get the diagonal covariance vector.

        Returns
        -------
        Array, shape (size,)
            Diagonal elements of the covariance matrix.

        Notes
        -----
        Applies positive constraint to ensure valid covariance values.
        """
        return constrain_positive(self.unconstrained_cov)

    # def set_static(self, static=True) -> None:
    #     self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


class Dynamics(SubclassRegistryMixin, ConfModule):
    """
    Abstract base class for dynamics models in XFADS.

    Defines the interface for state transition models that describe how
    the latent state evolves over time. Concrete subclasses implement
    specific dynamics such as linear, nonlinear, or neural dynamics.

    Attributes
    ----------
    noise : Noise
        Process noise model for the dynamics.

    Notes
    -----
    The dynamics model defines the state transition:
    z_{t+1} = f(z_t, u_t, c_t) + ε_t

    where f is implemented by the forward() method and ε_t ~ noise.
    """
    noise: eqx.AbstractVar[Noise]

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

        Notes
        -----
        This method should implement the deterministic function f in:
        z_{t+1} = f(z_t, u_t, c_t) + ε_t

        The noise ε_t is handled separately by the noise model.
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

    def cov(self) -> Array:
        """
        Get the process noise covariance.

        Returns
        -------
        Array
            Noise covariance matrix or parameters.
        """
        return self.noise.cov()  # type: ignore

    def loss(self) -> Array | float:
        """
        Compute regularization loss for the dynamics.

        Returns
        -------
        ArrayLike
            Regularization loss (default: 0.0).

        Notes
        -----
        Subclasses can override this to add parameter regularization,
        stability constraints, or other penalties.
        """
        return 0.0
