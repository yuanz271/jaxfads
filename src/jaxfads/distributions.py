"""
Exponential-family variational distributions for XFADS.

This module provides implementations of exponential family distributions
with natural and moment parameterizations for variational inference in
XFADS. Supports various approximations including full covariance, low-rank,
and diagonal multivariate normal distributions.

Functions
---------
damping_inv
    Compute matrix inverse with damping for numerical stability.

Classes
-------
Approx
    Abstract base class for exponential family approximations.
FullMVN
    Full covariance multivariate normal approximation.
LoRaMVN
    Low-rank multivariate normal approximation.
DiagMVN
    Diagonal covariance multivariate normal approximation.
"""

from abc import abstractmethod, ABC
import math

from jax import Array, numpy as jnp, random as jrnd
from tensorflow_probability.substrates.jax import distributions as tfd
from gearax.mixin import SubclassRegistryMixin

from .constraints import constrain_positive, unconstrain_positive


def damping_inv(a: Array, damping: float = 1e-6) -> Array:
    """
    Compute the inverse of a matrix with damping for numerical stability.

    Parameters
    ----------
    a : Array, shape (..., D, D)
        Input square matrix to be inverted.
    damping : float, default=1e-6
        Damping factor added to the diagonal for numerical stability.

    Returns
    -------
    Array, shape (..., D, D)
        Inverse of (a + damping * I).

    Notes
    -----
    The damping term prevents singular matrices and improves numerical
    stability by adding a small positive value to the diagonal elements.
    """
    return jnp.linalg.inv(a + damping * jnp.eye(a.shape[-1]))


class Approx(SubclassRegistryMixin, ABC):
    """
    Abstract base class for exponential family approximations in XFADS.

    This class defines the interface for exponential family distributions
    used in variational inference, providing conversions between natural
    and moment parameterizations, sampling methods, and other utilities.
    """

    @classmethod
    @abstractmethod
    def natural_to_moment(cls, natural) -> Array:
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

    @classmethod
    @abstractmethod
    def moment_to_natural(cls, moment) -> Array:
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

    @classmethod
    @abstractmethod
    def sample_by_moment(cls, key, moment, mc_size) -> Array:
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

    @classmethod
    @abstractmethod
    def param_size(cls, state_dim) -> int:
        """
        Get the total parameter size for given state dimension.

        Parameters
        ----------
        state_dim : int
            Dimensionality of the state space.

        Returns
        -------
        int
            Total number of parameters needed to parameterize the distribution.
        """
        ...

    @classmethod
    @abstractmethod
    def kl(cls, moment1, moment2) -> Array:
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

    @classmethod
    @abstractmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        """
        Convert moment parameters to canonical mean and covariance.

        Parameters
        ----------
        moment : Array
            Moment parameter vector.

        Returns
        -------
        mean : Array
            Mean vector.
        cov : Array
            Covariance matrix or parameters.
        """
        ...

    @classmethod
    @abstractmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        """
        Convert canonical mean and covariance to moment parameters.

        Parameters
        ----------
        mean : Array
            Mean vector.
        cov : Array
            Covariance matrix or parameters.

        Returns
        -------
        Array
            Moment parameter vector.
        """
        ...

    @classmethod
    @abstractmethod
    def full_cov(cls, cov: Array) -> Array:
        """
        Convert covariance parameterization to full covariance matrix.

        Parameters
        ----------
        cov : Array
            Covariance parameters (may be diagonal, low-rank, etc.).

        Returns
        -------
        Array
            Full covariance matrix.
        """
        ...

    @classmethod
    @abstractmethod
    def constrain_moment(cls, unconstrained: Array) -> Array: ...

    @classmethod
    @abstractmethod
    def constrain_natural(cls, unconstrained: Array) -> Array: ...

    @classmethod
    @abstractmethod
    def unconstrain_natural(cls, natural: Array) -> Array: ...

    @classmethod
    @abstractmethod
    def noise_moment(cls, noise_cov) -> Array: ...

    @classmethod
    @abstractmethod
    def prior_natural(cls, state_dim) -> Array: ...


class FullMVN(Approx):
    """
    Full covariance multivariate normal approximation.

    Implements exponential family operations for multivariate normal
    distributions with full covariance matrices. Uses natural parameters
    η = (Σ^{-1}μ, -½Σ^{-1}) where μ is mean and Σ is covariance.

    Notes
    -----
    Parameter layout: [mean_vec, cov_matrix_flattened]
    Total parameters: D + D²
    """
    @classmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        """
        Convert natural parameters to moment parameters.

        Parameters
        ----------
        natural : Array
            Natural parameters [Σ^{-1}μ, -½Σ^{-1}].

        Returns
        -------
        Array
            Moment parameters [μ, Σ].

        Notes
        -----
        Transforms from natural parameterization (Σ^{-1}μ, -½Σ^{-1})
        to moment parameterization (μ, Σ).
        """
        n = jnp.size(natural)
        m = cls.variable_size(n)
        nat1, nat2 = jnp.split(natural, [m])
        p = -2 * nat2  # vectorized precision
        P = jnp.reshape(p, (m, m))  # precision matrix
        loc = jnp.linalg.solve(P, nat1)
        V = damping_inv(P)
        v = V.flatten()
        moment = jnp.concatenate((loc, v))
        return moment

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        loc, V = cls.moment_to_canon(moment)
        P = damping_inv(V)
        Nat2 = -0.5 * P
        nat2 = Nat2.flatten()
        nat1 = P @ loc
        natural = jnp.concatenate((nat1, nat2))
        return natural

    @classmethod
    def sample_by_moment(cls, key: Array, moment: Array, mc_size: int) -> Array:
        loc, V = cls.moment_to_canon(moment)
        return jrnd.multivariate_normal(
            key, loc, V, shape=(mc_size,)
        )  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        n = jnp.size(moment)
        m = cls.variable_size(n)
        loc, v = jnp.split(moment, [m])
        V = jnp.reshape(v, (m, m))
        return loc, V

    @classmethod
    def variable_size(cls, param_size: int) -> int:
        """
        Get the variable size given a parameter vector size.

        Parameters
        ----------
        param_size : int
            Total size of the parameter vector.

        Returns
        -------
        int
            Dimensionality of the random variable.

        Notes
        -----
        For full MVN: param_size = D + D² where D is variable dimension.
        Solves: D² + D - param_size = 0 for D.
        """
        # n: size of vectorized mean param
        # m: size of random variable

        # n = m + m*m
        # m = (sqrt(1 + 4n) - 1) / 2. See doc for simpler solution m = floor(sqrt(n)).
        return int(math.sqrt(param_size))

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        v = cov.flatten()
        moment = jnp.concatenate((mean, v))
        return moment

    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Array:
        m1, V1 = cls.moment_to_canon(moment1)
        m2, V2 = cls.moment_to_canon(moment2)
        return tfd.kl_divergence(
            tfd.MultivariateNormalFullCovariance(m1, V1),
            tfd.MultivariateNormalFullCovariance(m2, V2),
            allow_nan_stats=False,
        )

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        return state_dim + state_dim * state_dim

    @classmethod
    def prior_natural(cls, state_dim) -> Array:
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.eye(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        return cov

    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + m + m
        # m = sqrt(n + 1) - 1
        m = n // 3

        loc, diag, lora = jnp.split(unconstrained, [m, m + m])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = V.flatten()
        return jnp.concatenate((loc, v))

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + m + m
        # m = sqrt(n + 1) - 1
        m = n // 3

        loc, diag, lora = jnp.split(unconstrained, [m, m + m])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = -V.flatten()  # negative definite
        return jnp.concatenate((loc, v))

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        return jnp.diag(noise_cov)


class LoRaMVN(Approx):
    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + 1 + m
        m = (n - 1) // 2

        loc, diag, lora = jnp.split(unconstrained, [m, m + 1])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = V.flatten()
        return jnp.concatenate((loc, v))

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        return jnp.diag(noise_cov)


class DiagMVN(Approx):
    """
    Diagonal covariance multivariate normal approximation.

    Implements exponential family operations for multivariate normal
    distributions with diagonal covariance matrices. More efficient
    than full covariance but less expressive.

    Notes
    -----
    Parameter layout: [mean_vec, diag_cov_vec]
    Total parameters: 2D
    """
    @classmethod
    def natural_to_moment(cls, natural) -> Array:
        nat1, nat2 = jnp.split(natural, 2)
        cov = -0.5 / nat2
        mean = -0.5 * nat1 / nat2
        return jnp.concatenate((mean, cov))

    @classmethod
    def moment_to_natural(cls, moment) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        nat2 = -0.5 / cov
        nat1 = mean / cov
        return jnp.concatenate((nat1, nat2))

    @classmethod
    def sample_by_moment(cls, key, moment, mc_size=None) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        shape = None if mc_size is None else (mc_size,)
        return jrnd.multivariate_normal(
            key, mean, jnp.diag(cov), shape=shape
        )  # It seems JAX does the reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment) -> tuple:
        mean, cov = jnp.split(
            moment, 2, -1
        )  # trick: the 2nd moment here is actually cov diag
        return mean, cov

    @classmethod
    def canon_to_moment(cls, mean, cov) -> Array:
        moment = jnp.concatenate((mean, cov))
        return moment

    @classmethod
    def variable_size(cls, param_size: int) -> int:
        # n: size of vectorized mean param
        # m: size of random variable
        # n = m + m*m
        # m = (sqrt(1 + 4n) - 1) / 2. See doc for simpler solution m = floor(sqrt(n)).
        return param_size // 2

    @classmethod
    def kl(cls, moment1, moment2) -> Array:
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        return tfd.kl_divergence(
            tfd.MultivariateNormalDiag(m1, cov1),
            tfd.MultivariateNormalDiag(m2, cov2),
            allow_nan_stats=False,
        )

    @classmethod
    def param_size(cls, state_dim) -> int:
        return 2 * state_dim

    @classmethod
    def prior_natural(cls, state_dim) -> Array:
        """Return standard normal in natural parameter form"""
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.ones(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        return jnp.diag(cov)

    @classmethod
    def constrain_moment(cls, unconstrained) -> Array:
        loc, v = jnp.split(unconstrained, 2)
        v = constrain_positive(v)
        return jnp.concatenate((loc, v))

    @classmethod
    def constrain_natural(cls, unconstrained) -> Array:
        n1, n2 = jnp.split(unconstrained, 2)
        n2 = -constrain_positive(n2)
        return jnp.concatenate((n1, n2))

    @classmethod
    def unconstrain_natural(cls, natural) -> Array:
        n1, n2 = jnp.split(natural, 2)
        n2 = unconstrain_positive(-n2)
        return jnp.concatenate((n1, n2))

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        return noise_cov
