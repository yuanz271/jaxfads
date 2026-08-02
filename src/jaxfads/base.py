"""
Abstract base interfaces for XFADS.

This module centralizes abstract base classes for dynamics, observation,
and approximate-distribution components to keep concrete implementations
in their respective modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
from gearax.mixin import SubclassRegistryMixin
from gearax.modules import ConfModule
from jax import Array
from jax import numpy as jnp
from jaxtyping import PyTree

from .logging import get_logger

logger = get_logger(__name__)


def _monte_carlo_transition_points(
    approx: Approx, key: Array, moment: Array, mc_size: int
) -> tuple[Array, Array]:
    """Default transition-point policy: ``mc_size`` i.i.d. samples via
    ``approx.sample_by_moment``, uniform weights ``1/mc_size``.

    Standalone, composable, independently testable -- ``Approx.
    transition_points`` merely delegates to it. Warns when ``mc_size`` is
    small enough relative to the ambient state dimension that the
    between-sample "spread" term is rank-deficient (see
    ``docs/transition_points.md``).
    """
    points = approx.sample_by_moment(key, moment, mc_size)
    dim = points.shape[-1]  # ambient state dim, read off sample_by_moment's
    # own output shape -- no new abstract Approx property needed
    # (param_size() isn't dim; e.g. full-rank MVN has param_size = dim + dim**2).
    if mc_size <= dim:
        logger.warning(
            "transition_points: mc_size=%d <= state_dim=%d; the MC spread "
            "term is rank-deficient (rank <= mc_size-1 < state_dim), the "
            "same structural precondition behind the Heywood-style exploit "
            "against R. Use mc_size >= state_dim + 1, or a deterministic "
            "policy (e.g. MVN(use_sigma_points=True)) to avoid this entirely.",
            mc_size,
            dim,
        )
    weights = jnp.full((mc_size,), 1.0 / mc_size)
    return points, weights


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
        """Return the flat natural/moment parameter size for this instance."""
        ...

    # ------------------------------------------------------------------
    # Encoder-facing hooks
    # ------------------------------------------------------------------

    def free_size(self) -> int:
        """Return the size of the encoder free-form output vector.

        By default, encoders emit free-form vectors in the same flat layout as
        the distribution's natural parameters (so this returns
        :meth:`param_size`).

        Approximations may override this to support compact encoder outputs
        (e.g. low-rank updates) while still producing full-size natural
        parameter updates for filtering.
        """

        return self.param_size()

    def free_to_natural(self, free: Array) -> Array:
        """Convert a free-form vector into an additive natural update.

        Default implementation follows the standard conversion chain:

        ``free → canon → moment → natural``.
        """

        canon = self.free_to_canon(free)
        moment = self.canon_to_moment(canon)
        return self.moment_to_natural(moment)

    def transition_points(
        self, key: Array, moment: Array, mc_size: int
    ) -> tuple[Array, Array]:
        """Representative (points, weights) approximating q(z_{t-1}) for
        propagation through the transition.

        Parameters
        ----------
        key : Array
            JAX PRNG key.
        moment : Array
            Moment parameters of q(z_{t-1}).
        mc_size : int
            Requested point/sample count (may be ignored by deterministic
            overrides).

        Returns
        -------
        points : Array, shape (n_points, state_dim)
        weights : Array, shape (n_points,)
            Nonnegative-summing-to-1 by default; overrides may use signed
            weights (e.g. unscented-transform weights).

        Notes
        -----
        Default: plain Monte Carlo via :func:`_monte_carlo_transition_points`
        (today's behavior, bit-for-bit). Approximations may override this
        with a deterministic point set (e.g. unscented-transform sigma
        points) -- see ``MVN(use_sigma_points=True)`` and
        ``docs/transition_points.md``.
        """
        return _monte_carlo_transition_points(self, key, moment, mc_size)

    def transition_stat(self, zs: Array, weights: Array) -> Any:
        """Reduce a propagated, noise-free point set ``(zs, weights)`` --
        as produced by ``core.propagate_transition_points`` -- to whatever
        family-specific statistic this subclass's own :meth:`shrink`
        needs as its ``transition_stat`` argument.

        Called once per (batch, time) pair by ``XFADS``'s own forward
        pass (``core._site_filter``/``nofilt``/``causal``),
        **unconditionally**, for every ``Approx`` subclass -- not gated
        behind whether ``shrink``/``Q``-estimation is actually configured
        for a given model (see ``docs/mstep_dynamics_noise.md`` for why:
        gating this behind a flag was tried and rejected once checked
        against the actual marginal cost). This method must therefore be
        cheap and safe to call regardless of whether the model ever calls
        ``shrink``.

        ``core.py``'s recursions call this method polymorphically and
        never interpret its return value themselves -- they only stack it
        across time steps via ``jax.lax.scan``/``jax.vmap`` (which
        requires a fixed pytree structure/shape across steps, satisfied
        as long as this method's output shape doesn't depend on the
        *values* of ``zs``/``weights``, only their static shapes). This
        keeps ``core.py`` itself fully agnostic to what a "transition
        statistic" means for any concrete family -- unlike an earlier
        design that had ``core.py`` reduce the point set to a
        mean/covariance pair directly, presuming a Gaussian-shaped
        sufficient statistic (rejected once checked against ``core.py``'s
        own agnosticism invariant).

        Default: identity -- returns ``(zs, weights)`` unchanged, i.e. no
        reduction at all. This is the safe, zero-assumption default: any
        ``Approx`` subclass that doesn't override this behaves exactly as
        if this method didn't exist (the raw point set passed straight
        through as ``transition_stat``). A subclass overrides this only when
        it wants a smaller, reduced per-pair summary instead of the raw
        point set (e.g. a Gaussian family reducing to a weighted
        mean/covariance pair, which is asymptotically smaller than the
        raw point set whenever the point count exceeds ``state_dim``),
        pairing its override with its own :meth:`shrink` implementation
        that knows how to consume that reduced form.

        Parameters
        ----------
        zs : Array, shape (n_points, state_dim)
            Propagated points (no noise added).
        weights : Array, shape (n_points,)
            Corresponding point weights.

        Returns
        -------
        Any
            Subclass-defined reduced statistic (default: ``(zs,
            weights)`` unchanged).
        """
        return zs, weights

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
    Abstract base class for latent-state dynamics modules in XFADS.

    Concrete subclasses implement ``eval(z, u, c)``.
    """

    @abstractmethod
    def eval(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        """
        Evaluate the latent-state dynamics.

        Parameters
        ----------
        z : Array, shape (state_dim,)
            Current state vector.
        u : Array, shape (input_dim,)
            Control/input vector.
        c : Array, shape (covariate_dim,)
            Covariate vector.
        key : PRNGKeyArray, optional
            Random key for stochastic map components (e.g., dropout).

        Returns
        -------
        Array, shape (state_dim,)
            Map output. For continuous systems this is ``dz/dt``; for
            discrete systems this is ``z_{t+1}``.
        """
        ...

    def __call__(self, *args, **kwargs) -> Array:
        """
        Convenience method to call eval().

        Returns
        -------
        Array
            Result of eval(*args, **kwargs).
        """
        return self.eval(*args, **kwargs)


class Integrator(SubclassRegistryMixin, ABC):
    """Abstract base class for state-evolution integrators in XFADS."""

    @abstractmethod
    def step(
        self,
        z: Array,
        u: Array,
        c: Array,
        dynamics: Dynamics,
        *,
        key=None,
    ) -> Array:
        """Advance state by one step using ``dynamics``."""
        ...


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

    def batch_stat(
        self, t, y, u, c, moment, transition_stat, approx
    ):
        """Return one additive observation statistic, or ``None``."""
        return None

    def mstep(self, *, t, y, moment, transition_stat, approx):
        """Finalize one accumulated observation statistic into an Observation."""
        return self

    def accumulate_stat(self, total, delta):
        """Add fixed-shape observation statistic pytrees, preserving ``None``."""
        if total is None:
            return delta
        if delta is None:
            return total
        return jax.tree.map(lambda left, right: left + right, total, delta)

    def mstep_from_data(self, t: Array, moment: Array, y: Array, approx: Approx) -> Observation:
        """Closed-form, non-SGD parameter update from a full forward pass.

        Computes a closed-form (EM M-step-style) update for this
        Observation's own parameters (e.g. Gaussian observation noise
        covariance), given ``(t, moment, y)`` for the *entire* dataset (or
        an entire batch treated as such) and the current ``approx``. This
        is deliberately separate from gradient-based training. Not
        required to be gradient-free on its own terms -- the actual
        guarantee against interfering with SGD lives at the call site
        (e.g. ``mstep_observation_cov``, or ``train()``'s own
        ``train_step``/``apply_mstep``), which only ever invokes this
        after the current step's gradient has already been computed and
        applied, using an already-concrete, already-updated model.
        Implementations should not add their own defensive
        ``stop_gradient`` either -- the invariant belongs at the call
        site, not duplicated into every implementation.

        Default implementation is a no-op: returns ``self`` unchanged.
        Concrete subclasses that support this (e.g. ``GLM`` wrapping a
        ``Gaussian`` likelihood) override it; subclasses that don't
        (e.g. ``GLM`` wrapping ``Poisson``) simply inherit this default.

        Parameters
        ----------
        t : Array
            Time indices for the dataset/batch, shape matching ``y``'s
            leading axes.
        moment : Array
            Moment parameters of the posterior ``q(z_t)`` for every
            (batch, time) instance, from a forward pass over ``t, y, u, c``.
        y : Array
            Observed data for every (batch, time) instance.
        approx : Approx
            Exponential-family approximation instance (needed to unpack
            ``moment`` into mean/covariance; not owned by ``Observation``).

        Returns
        -------
        Observation
            A (possibly) updated Observation instance. Must not depend on
            or produce gradients.
        """
        return self

    def frozen_paths(self) -> list[str]:
        """Attribute paths (relative to this Observation) that must be
        excluded from gradient updates whenever :meth:`mstep`-driven
        updates are active.

        Callers (e.g. ``train()``, which does this unconditionally, every
        training run) use this to automatically derive which leaves to
        freeze from the optimizer, so that gradient descent does not fight
        a closed-form update computed by :meth:`mstep`. No user-facing
        configuration is required for this; ``train()`` folds these paths
        into its own freeze mask automatically.

        Default implementation returns ``[]`` (nothing to freeze).
        Concrete subclasses that override :meth:`mstep` non-trivially
        should override this too, returning the paths :meth:`mstep`
        actually writes to.

        Returns
        -------
        list of str
            Dot-separated attribute paths, relative to this Observation
            instance (e.g. ``["likelihood.unconstrained_cov"]`` for a
            ``GLM`` wrapping a ``Gaussian`` likelihood).
        """
        return []


class Encoder(SubclassRegistryMixin, ConfModule):
    """Abstract base class for observation encoders.

    Subclasses are auto-registered and can be looked up via
    ``Encoder.get_subclass(name)``.
    """

    @abstractmethod
    def __call__(self, y: Array, *, key: Array | None = None) -> Array:
        """Encode a single observation vector.

        Parameters
        ----------
        y : Array, shape (observation_dim,)
            Observation vector.
        key : Array or None
            Optional PRNG key for stochastic encoders.

        Returns
        -------
        Array
            Encoded representation. Shape depends on subclass.
        """
        ...


__all__ = [
    "Approx",
    "Dynamics",
    "Integrator",
    "Observation",
    "Encoder",
]
