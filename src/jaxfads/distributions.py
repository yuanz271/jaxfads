"""
Exponential-family variational distributions for XFADS.

This module provides implementations of exponential-family distributions
with natural and moment parameterizations for variational inference in XFADS.
It supports full-covariance, low-rank, and diagonal multivariate normal
approximations.
"""

import math
from abc import ABC, abstractmethod

from jax import Array
from jax import numpy as jnp
from jax import random as jrnd
from tensorflow_probability.substrates.jax import distributions as tfd

from gearax.mixin import SubclassRegistryMixin

from .constraints import _EPS, constrain_positive, unconstrain_positive


def damping_inv(a: Array, damping: float = _EPS) -> Array:
    """
    Compute the inverse of a matrix with damping for numerical stability.

    Uses ``jnp.linalg.solve`` instead of explicit ``inv`` for better
    numerical conditioning.

    Parameters
    ----------
    a : Array, shape (..., D, D)
        Input square matrix to be inverted.
    damping : float, default=_EPS
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
    eye = jnp.eye(a.shape[-1])
    return jnp.linalg.solve(a + damping * eye, eye)


class Approx(SubclassRegistryMixin, ABC):
    """
    Abstract base class for exponential family approximations in XFADS.

    This class defines the interface for exponential family distributions
    used in variational inference, providing conversions between natural
    and moment parameterizations, sampling methods, and other utilities.
    """

    @classmethod
    @abstractmethod
    def natural_to_moment(cls, natural: Array) -> Array:
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
    def moment_to_natural(cls, moment: Array) -> Array:
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
    def sample_by_moment(cls, key: Array, moment: Array, mc_size: int) -> Array:
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
    def param_size(cls, state_dim: int) -> int:
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
    def kl(cls, moment1: Array, moment2: Array) -> Array:
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
    def constrain_moment(cls, unconstrained: Array) -> Array:
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

    @classmethod
    @abstractmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
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

    @classmethod
    @abstractmethod
    def unconstrain_natural(cls, natural: Array) -> Array:
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

    @classmethod
    @abstractmethod
    def noise_moment(cls, noise_cov: Array) -> Array:
        """
        Convert noise covariance to moment parameter format.

        Parameters
        ----------
        noise_cov : Array
            Diagonal noise covariance values.

        Returns
        -------
        Array
            Noise covariance in the format expected by this approximation
            (e.g., diagonal vector or full matrix).
        """
        ...

    @classmethod
    @abstractmethod
    def prior_natural(cls, state_dim: int) -> Array:
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
        """See base class. Inverts covariance to get precision."""
        loc, V = cls.moment_to_canon(moment)
        P = damping_inv(V)
        Nat2 = -0.5 * P
        nat2 = Nat2.flatten()
        nat1 = P @ loc
        natural = jnp.concatenate((nat1, nat2))
        return natural

    @classmethod
    def sample_by_moment(cls, key: Array, moment: Array, mc_size: int) -> Array:
        """See base class. Uses JAX multivariate_normal with reparameterization."""
        loc, V = cls.moment_to_canon(moment)
        return jrnd.multivariate_normal(
            key, loc, V, shape=(mc_size,)
        )  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        """See base class. Extracts mean vector and reshapes covariance matrix."""
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
        """See base class. Flattens covariance and concatenates with mean."""
        v = cov.flatten()
        moment = jnp.concatenate((mean, v))
        return moment

    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Array:
        """See base class. Uses TFP for full covariance KL computation."""
        m1, V1 = cls.moment_to_canon(moment1)
        m2, V2 = cls.moment_to_canon(moment2)
        return tfd.kl_divergence(
            tfd.MultivariateNormalFullCovariance(m1, V1),
            tfd.MultivariateNormalFullCovariance(m2, V2),
            allow_nan_stats=False,
        )

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        """See base class. Returns D + D² for full covariance."""
        return state_dim + state_dim * state_dim

    @classmethod
    def prior_natural(cls, state_dim: int) -> Array:
        """See base class. Returns N(0, I) in natural form."""
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.eye(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        """See base class. Identity for full covariance."""
        return cov

    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        """See base class. Builds PSD covariance via Cholesky LL^T.

        Input layout: [mean (D), lower-triangular entries (D²)].
        The D² block is reshaped to (D, D), zeroed above the diagonal,
        and Σ = LL^T is computed (PSD by construction).
        """
        n = jnp.size(unconstrained)
        m = cls.variable_size(n)
        loc, flat_L = jnp.split(unconstrained, [m])
        L = jnp.tril(jnp.reshape(flat_L, (m, m)))
        V = L @ L.T
        return jnp.concatenate((loc, V.flatten()))

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        """See base class. Builds negative definite precision via Cholesky.

        Input layout: [nat1 (D), lower-triangular entries (D²)].
        Produces nat2 = -LL^T (negative semi-definite by construction).
        """
        n = jnp.size(unconstrained)
        m = cls.variable_size(n)
        nat1, flat_L = jnp.split(unconstrained, [m])
        L = jnp.tril(jnp.reshape(flat_L, (m, m)))
        V = L @ L.T
        return jnp.concatenate((nat1, (-V).flatten()))

    @classmethod
    def unconstrain_natural(cls, natural: Array) -> Array:
        """See base class. Recovers Cholesky factor L from nat2 = -LL^T."""
        n = jnp.size(natural)
        m = cls.variable_size(n)
        nat1, nat2_flat = jnp.split(natural, [m])
        neg_nat2 = jnp.reshape(-nat2_flat, (m, m))
        L = jnp.linalg.cholesky(neg_nat2 + _EPS * jnp.eye(m))
        return jnp.concatenate((nat1, L.flatten()))

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        """See base class. Converts diagonal noise to full matrix."""
        return jnp.diag(noise_cov)



class LoRaMVN(Approx):
    """
    Low-rank plus diagonal multivariate normal approximation.

    Implements exponential family operations for multivariate normal
    distributions with low-rank plus diagonal covariance structure:
    Σ = diag(d) + vv^T where d is diagonal and v is a rank-1 factor.

    This provides a trade-off between the expressiveness of full
    covariance and the efficiency of diagonal covariance.

    .. warning::

        **Incomplete implementation.** Only ``constrain_moment`` and
        ``noise_moment`` are implemented. All other ``Approx`` methods
        (``natural_to_moment``, ``moment_to_natural``, ``sample_by_moment``,
        ``param_size``, ``kl``, ``moment_to_canon``, ``canon_to_moment``,
        ``full_cov``, ``constrain_natural``, ``unconstrain_natural``,
        ``prior_natural``) raise ``NotImplementedError``. Do not select
        this class as the approximation family until the remaining methods
        are filled in.

    Notes
    -----
    Parameter layout: [mean_vec, diag_scalar, low_rank_vec]
    Total parameters: D + 1 + D = 2D + 1

    The covariance is parameterized as:

    .. math::

        \\Sigma = \\text{diag}(d) + vv^T

    This allows capturing one principal direction of correlation while
    maintaining O(D) storage and computation.

    See Also
    --------
    FullMVN : Full covariance (D + D² parameters).
    DiagMVN : Diagonal covariance (2D parameters).
    """

    @classmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.natural_to_moment is not yet implemented.")

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.moment_to_natural is not yet implemented.")

    @classmethod
    def sample_by_moment(cls, key: Array, moment: Array, mc_size: int) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.sample_by_moment is not yet implemented.")

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        """See base class."""
        raise NotImplementedError("LoRaMVN.param_size is not yet implemented.")

    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.kl is not yet implemented.")

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        """See base class."""
        raise NotImplementedError("LoRaMVN.moment_to_canon is not yet implemented.")

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.canon_to_moment is not yet implemented.")

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.full_cov is not yet implemented.")

    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        """See base class. Builds PSD covariance via diag + rank-1.

        Input layout: [mean (D), diag (D), low-rank factor (D)].
        Produces Σ = diag(d) + vv^T.
        """
        n = jnp.size(unconstrained)
        # n = m + m + m
        m = n // 3

        loc, diag, lora = jnp.split(unconstrained, [m, m + m])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = V.flatten()
        return jnp.concatenate((loc, v))

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        """See base class. Builds negative definite precision via diag + rank-1.

        Input layout: [nat1 (D), diag (D), low-rank factor (D)].
        Produces nat2 = -(diag(d) + vv^T) (negative definite).
        """
        n = jnp.size(unconstrained)
        # n = m + m + m
        m = n // 3

        loc, diag, lora = jnp.split(unconstrained, [m, m + m])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = -V.flatten()  # negative definite
        return jnp.concatenate((loc, v))

    @classmethod
    def unconstrain_natural(cls, natural: Array) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.unconstrain_natural is not yet implemented.")

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        """See base class. Converts diagonal noise to full matrix."""
        return jnp.diag(noise_cov)

    @classmethod
    def prior_natural(cls, state_dim: int) -> Array:
        """See base class."""
        raise NotImplementedError("LoRaMVN.prior_natural is not yet implemented.")



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
    def natural_to_moment(cls, natural: Array) -> Array:
        """
        See base class. Diagonal case: element-wise inversion.
        """
        nat1, nat2 = jnp.split(natural, 2)
        cov = -0.5 / nat2
        mean = -0.5 * nat1 / nat2
        return jnp.concatenate((mean, cov))

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        """See base class. Diagonal case: element-wise operations.

        Notes
        -----
        A minimum floor of _EPS is applied to the covariance before
        inversion to prevent extreme natural parameters (nat2 ~ -1/2cov)
        when the covariance is near zero.  This mirrors the damping used
        in ``FullMVN.moment_to_natural`` via ``damping_inv``.
        """
        mean, cov = cls.moment_to_canon(moment)
        cov = jnp.maximum(cov, _EPS)
        nat2 = -0.5 / cov
        nat1 = mean / cov
        return jnp.concatenate((nat1, nat2))

    @classmethod
    def sample_by_moment(
        cls, key: Array, moment: Array, mc_size: int | None = None
    ) -> Array:
        """See base class. Reparameterized diagonal sampling (O(D), no Cholesky)."""
        mean, cov = cls.moment_to_canon(moment)
        std = jnp.sqrt(jnp.maximum(cov, _EPS))
        shape = mean.shape if mc_size is None else (mc_size,) + mean.shape
        return mean + std * jrnd.normal(key, shape)

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        """See base class. Splits into mean and diagonal covariance."""
        mean, cov = jnp.split(
            moment, 2, -1
        )  # trick: the 2nd moment here is actually cov diag
        return mean, cov

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        """See base class. Concatenates mean and diagonal covariance."""
        moment = jnp.concatenate((mean, cov))
        return moment

    @classmethod
    def variable_size(cls, param_size: int) -> int:
        """See base class. Returns param_size // 2 for diagonal case."""
        # n: size of vectorized mean param
        # m: size of random variable
        # n = m + m*m
        # m = (sqrt(1 + 4n) - 1) / 2. See doc for simpler solution m = floor(sqrt(n)).
        return param_size // 2

    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Array:
        """
        See base class. Uses TFP diagonal MVN for efficient KL.

        Notes
        -----
        ``moment_to_canon`` returns *variance* vectors, but TFP's
        ``MultivariateNormalDiag`` expects ``scale_diag`` (std).
        We convert via ``sqrt(max(cov, _EPS))`` to avoid sqrt-of-zero.
        """
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        return tfd.kl_divergence(
            tfd.MultivariateNormalDiag(
                loc=m1, scale_diag=jnp.sqrt(jnp.maximum(cov1, _EPS))
            ),
            tfd.MultivariateNormalDiag(
                loc=m2, scale_diag=jnp.sqrt(jnp.maximum(cov2, _EPS))
            ),
            allow_nan_stats=False,
        )

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        """See base class. Returns 2D for diagonal case."""
        return 2 * state_dim

    @classmethod
    def prior_natural(cls, state_dim: int) -> Array:
        """See base class. Returns N(0, I) in natural form."""
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.ones(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        """See base class. Converts diagonal to full matrix."""
        return jnp.diag(cov)

    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        """See base class. Applies positivity to variance terms."""
        loc, v = jnp.split(unconstrained, 2)
        v = constrain_positive(v)
        return jnp.concatenate((loc, v))

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        """See base class. Ensures negative precision (nat2 < 0)."""
        n1, n2 = jnp.split(unconstrained, 2)
        n2 = -constrain_positive(n2)
        return jnp.concatenate((n1, n2))

    @classmethod
    def unconstrain_natural(cls, natural: Array) -> Array:
        """See base class. Inverts constrain_natural."""
        n1, n2 = jnp.split(natural, 2)
        n2 = unconstrain_positive(-n2)
        return jnp.concatenate((n1, n2))

    @classmethod
    def noise_moment(cls, noise_cov: Array) -> Array:
        """See base class. Identity for diagonal case."""
        return noise_cov

