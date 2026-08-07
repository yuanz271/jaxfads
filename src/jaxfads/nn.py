"""
Neural network utilities and custom layers for XFADS models.

This module provides building blocks for constructing neural networks used in
XFADS (eXtended Filtered Approximate Dynamical Systems), including:

- MLPs with flexible architectures
- Weight normalization for training stability
- Custom linear layers with variant biases
- Radial basis function networks
- Data masking for regularization

All components are built using JAX and Equinox for high performance and
clean functional programming patterns.
"""

import math
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
from equinox import Module
from equinox import nn as enn
from jax import Array
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrnd

from . import constraints
from .constraints import _EPS


def make_mlp(
    in_size: int,
    out_size: int,
    width: int,
    depth: int,
    *,
    key: Array,
    activation: Callable = jnn.swish,
    final_bias: bool = True,
    final_activation: Callable | None = None,
    dropout: float | None = None,
) -> enn.Sequential:
    """
    Create a multi-layer perceptron (MLP) with configurable architecture.

    Parameters
    ----------
    in_size : int or "scalar"
        Input dimension size. Use "scalar" for 1D input.
    out_size : int or "scalar"
        Output dimension size. Use "scalar" for 1D output.
    width : int
        Hidden layer width (number of units per layer).
    depth : int
        Number of hidden layers.
    key : Array
        PRNGKey for parameter initialization.
    activation : Callable, default=jnn.swish
        Activation function for hidden layers.
    final_bias : bool, default=True
        Whether to include bias in the final layer.
    final_activation : Callable or None, default=None
        Optional activation function for the final layer.
    dropout : float or None, default=None
        Dropout probability. If None, no dropout is applied.

    Returns
    -------
    enn.Sequential
        Configured MLP as an Equinox Sequential module.

    Examples
    --------
    >>> key = jax.random.key(0)
    >>> mlp = make_mlp(10, 5, width=64, depth=3, key=key)
    >>> output = mlp(input_array)

    Create an MLP with dropout:
    >>> mlp = make_mlp(784, 10, width=128, depth=2, dropout=0.2, key=key)

    Create an MLP with custom activations:
    >>> mlp = make_mlp(64, 32, width=256, depth=4,
    ...               activation=jnn.relu, final_activation=jnn.tanh, key=key)
    """
    key, layer_key = jrnd.split(key)
    layers = [enn.Linear(in_size, width, key=layer_key), enn.Lambda(activation)]
    for i in range(depth - 1):
        key, layer_key = jrnd.split(key)
        layers.extend(
            [enn.Linear(width, width, key=layer_key), enn.Lambda(activation)]
        )
        if dropout is not None:
            layers.append(enn.Dropout(dropout))
    key, layer_key = jrnd.split(key)
    layers.append(enn.Linear(width, out_size, key=layer_key, use_bias=final_bias))
    if final_activation is not None:
        layers.append(enn.Lambda(final_activation))
    if dropout is not None:
        layers.append(enn.Dropout(dropout))

    return enn.Sequential(layers)


def _norm_except_axis(
    v: Array, norm: Callable[[Array], Array], axis: int | None
) -> Array:
    """
    Compute norm along all axes except the specified one.

    Parameters
    ----------
    v : Array
        Input array to compute norm for.
    norm : Callable[[Array], Scalar]
        Function that computes a norm (e.g., jnp.linalg.norm).
    axis : int or None
        Axis to exclude from norm computation. If None, compute over all axes.

    Returns
    -------
    Array
        Norm values with same shape as input along the excluded axis.
    """
    if axis is None:
        return norm(v)
    else:
        return jax.vmap(norm, in_axes=axis, out_axes=axis)(v)


class NormalizedLinear(Module):
    """
    Linear layer wrapper that normalizes the weight matrix.

    Wraps an ``enn.Linear`` layer and projects its weight matrix to unit
    norm in the forward pass.  This resolves the scale ambiguity between
    the readout matrix and the latent state: only the *direction* of the
    weight is identifiable, so the magnitude is fixed to one.

    The wrapper contains no additional trainable parameters — the
    underlying ``layer`` holds all learnable weights and biases.

    Parameters
    ----------
    layer : enn.Linear
        The linear layer to wrap.
    weight_name : str, default="weight"
        Name of the weight attribute on *layer* to normalize.
    axis : int or None, default=None
        Axis to exclude from the norm computation.  ``None`` normalizes
        globally (Frobenius norm → 1).
    """

    layer: enn.Linear
    weight_name: str = eqx.field(static=True)
    axis: int | None = eqx.field(static=True)

    def __init__(
        self,
        layer: enn.Linear,
        weight_name: str = "weight",
        axis: int | None = None,
    ):
        """
        Initialize normalized linear wrapper.

        Parameters
        ----------
        layer : enn.Linear
            The linear layer to wrap.
        weight_name : str, default="weight"
            Name of the weight attribute to normalize.
        axis : int or None, default=None
            Axis to exclude from the norm computation.  ``None``
            normalizes globally.
        """
        self.layer = layer
        self.weight_name = weight_name
        self.axis = axis

    def _norm(self, w):
        """
        Compute the norm used for weight normalization.

        Parameters
        ----------
        w : Array
            Weight array.

        Returns
        -------
        Array
            Norm value(s) with shape suitable for broadcasting.
        """
        return _norm_except_axis(
            w, norm=partial(jnp.linalg.norm, keepdims=True), axis=self.axis
        )

    @property
    def weight(self) -> Array:
        """
        Get the unit-norm weight matrix.

        Returns
        -------
        Array
            ``w / ||w||`` where *w* is the raw weight from the wrapped
            layer.
        """
        w = getattr(self.layer, self.weight_name)
        w = w / (self._norm(w) + _EPS)

        return w

    @property
    def bias(self) -> Array | None:
        """
        Get the bias vector from the wrapped layer.

        Returns
        -------
        Array or None
            Bias vector, or ``None`` if the layer has no bias.
        """
        return self.layer.bias

    @jax.named_scope("xfads.nn.NormalizedLinear")
    def __call__(self, x: Array) -> Array:
        """
        Apply the normalized linear transformation.

        Parameters
        ----------
        x : Array
            Input array of shape (..., input_dim).

        Returns
        -------
        Array
            ``(w / ||w||) @ x + bias``.
        """
        weight: Array = self.weight
        layer: Callable = eqx.tree_at(
            lambda layer: getattr(layer, self.weight_name), self.layer, weight
        )
        return layer(x)


class StationaryLinear(Module):
    """
    Time-invariant readout layer for state-space models.

    Maps latent states to observations via a linear transformation whose
    parameters do not depend on time.

    Parameters
    ----------
    state_dim : int
        Dimension of the latent state.
    observation_dim : int
        Dimension of the observations.
    key : Array
        PRNGKey for parameter initialization.
    norm_readout : bool, default=False
        Whether to normalize the weight matrix to unit norm for
        identifiability (wraps the layer with :class:`NormalizedLinear`).

    Attributes
    ----------
    layer : enn.Linear or NormalizedLinear
        The underlying linear transformation layer.
    """

    layer: enn.Linear | NormalizedLinear

    def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False):
        """
        Initialize stationary linear layer.

        Parameters
        ----------
        state_dim : int
            Dimension of the input state vector.
        observation_dim : int
            Dimension of the output observation vector.
        key : Array
            PRNGKey for parameter initialization.
        norm_readout : bool, default=False
            Whether to normalize the weight matrix for identifiability.
        """
        self.layer = enn.Linear(state_dim, observation_dim, key=key, use_bias=True)

        if norm_readout:
            self.layer = NormalizedLinear(self.layer)

    def __call__(self, idx, x) -> Array:
        """
        Apply linear transformation (ignoring time index).

        Parameters
        ----------
        idx : Any
            Time index (ignored for stationary layer).
        x : Array
            Input state vector of shape (..., state_dim).

        Returns
        -------
        Array
            Output observation vector of shape (..., observation_dim).
        """
        return self.layer(x)

    @property
    def weight(self) -> Array:
        """
        Get the weight matrix.

        Returns
        -------
        Array
            Weight matrix of shape (observation_dim, state_dim).
        """
        return self.layer.weight

    def set_weight(self, weight: Array) -> "StationaryLinear":
        """
        Set the readout weight matrix.

        Parameters
        ----------
        weight : Array
            Weight matrix of shape (observation_dim, state_dim).

        Returns
        -------
        StationaryLinear
            Updated layer with weight set.
        """
        if isinstance(self.layer, NormalizedLinear):
            layer = eqx.tree_at(lambda ly: ly.layer.weight, self.layer, weight)
        else:
            layer = eqx.tree_at(lambda ly: ly.weight, self.layer, weight)
        return eqx.tree_at(lambda r: r.layer, self, layer)

    def set_bias(self, bias: Array) -> "StationaryLinear":
        """
        Set the bias vector.

        Parameters
        ----------
        bias : Array
            Bias vector of shape (observation_dim,).

        Returns
        -------
        StationaryLinear
            Updated layer with bias set.
        """
        if isinstance(self.layer, NormalizedLinear):
            layer = eqx.tree_at(lambda ly: ly.layer.bias, self.layer, bias)
        else:
            layer = eqx.tree_at(lambda ly: ly.bias, self.layer, bias)
        return eqx.tree_at(lambda r: r.layer, self, layer)


class IdentityReadout(Module):
    """Fixed (non-trainable) identity readout.

    A pure stub for the case where the latent state *is* the observation space
    (``state_dim == observation_dim``), e.g. when observations are already a
    PCA projection and no learned decoder is desired. It carries **no array
    fields**, so it contributes zero leaves to the optimizer pytree and is
    non-trainable by construction (no ``freeze_paths`` needed).

    The readout map is the identity: ``__call__(idx, x) -> x``. The ``weight``
    accessor returns ``I`` lazily (materialized on demand, not stored) because
    the analytic Gaussian ``eloglik`` reads ``readout.weight`` for the
    covariance term ``C @ Cov(z) @ C.T``; for identity this equals ``Cov(z)``.
    ``set_weight``/``set_bias`` are no-ops returning ``self``.

    Parameters
    ----------
    state_dim : int
        Latent state dimension. Must equal ``observation_dim``.
    observation_dim : int
        Observation dimension. Must equal ``state_dim``.
    """

    state_dim: int = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)

    def __init__(self, state_dim: int, observation_dim: int):
        if int(state_dim) != int(observation_dim):
            raise ValueError(
                "IdentityReadout requires state_dim == observation_dim "
                f"(got {state_dim} != {observation_dim})."
            )
        self.state_dim = int(state_dim)
        self.observation_dim = int(observation_dim)

    def __call__(self, idx, x) -> Array:
        """Identity map. The time index ``idx`` is part of the readout
        interface but not used by the identity map."""
        return x

    @property
    def weight(self) -> Array:
        """Lazily-materialized identity matrix (not a stored/trainable leaf)."""
        return jnp.eye(self.observation_dim)

    def set_weight(self, weight: Array) -> "IdentityReadout":
        """Return ``self`` unchanged; the identity map has no weight to set."""
        return self

    def set_bias(self, bias: Array) -> "IdentityReadout":
        """Return ``self`` unchanged; the identity map has no bias to set."""
        return self


class VariantBiasLinear(Module):
    """
    Readout layer with time-variant bias for non-stationary observations.

    Shares a single weight matrix across all time steps but uses a
    different bias vector at each step, suitable for modelling
    non-stationary baselines in dynamical systems.

    Parameters
    ----------
    state_dim : int
        Dimension of the latent state.
    observation_dim : int
        Dimension of the observations.
    n_biases : int
        Number of different bias vectors (typically number of time points).
    key : Array
        PRNGKey for parameter initialization.
    norm_readout : bool, default=False
        Whether to normalize the weight matrix to unit norm for
        identifiability (wraps the layer with :class:`NormalizedLinear`).

    Attributes
    ----------
    biases : Array
        Array of bias vectors of shape (n_biases, observation_dim).
    layer : enn.Linear or NormalizedLinear
        The underlying linear transformation layer (without bias).
    """

    biases: Array
    layer: enn.Linear | NormalizedLinear

    def __init__(
        self, state_dim, observation_dim, n_biases, *, key, norm_readout: bool = False
    ):
        """
        Initialize variant bias linear layer.

        Parameters
        ----------
        state_dim : int
            Dimension of the input state vector.
        observation_dim : int
            Dimension of the output observation vector.
        n_biases : int
            Number of different bias vectors to create.
        key : Array
            PRNGKey for parameter initialization.
        norm_readout : bool, default=False
            Whether to normalize the weight matrix for identifiability.
        """
        wkey, bkey = jrnd.split(key, 2)

        self.layer = enn.Linear(state_dim, observation_dim, key=wkey, use_bias=False)
        lim = 1 / math.sqrt(state_dim)
        self.biases = jrnd.uniform(
            bkey,
            (n_biases, observation_dim),
            dtype=self.layer.weight.dtype,
            minval=-lim,
            maxval=lim,
        )

        if norm_readout:
            self.layer = NormalizedLinear(self.layer)

    def __call__(self, idx, x) -> Array:
        """
        Apply linear transformation with time-specific bias.

        Parameters
        ----------
        idx : int or Array
            Time index or indices specifying which bias to use.
        x : Array
            Input state vector of shape (..., state_dim).

        Returns
        -------
        Array
            Output observation vector of shape (..., observation_dim)
            with the appropriate bias added.
        """
        x = self.layer(x)
        return x + self.biases[idx]

    @property
    def weight(self) -> Array:
        """
        Readout weight matrix.

        Returns
        -------
        Array
            Underlying linear layer weight of shape (observation_dim, state_dim).
        """
        return self.layer.weight

    def set_weight(self, weight: Array) -> "VariantBiasLinear":
        """
        Set the readout weight matrix.

        Parameters
        ----------
        weight : Array
            Weight matrix of shape (observation_dim, state_dim).

        Returns
        -------
        VariantBiasLinear
            Updated layer with weight set.
        """
        if isinstance(self.layer, NormalizedLinear):
            layer = eqx.tree_at(lambda ly: ly.layer.weight, self.layer, weight)
        else:
            layer = eqx.tree_at(lambda ly: ly.weight, self.layer, weight)
        return eqx.tree_at(lambda r: r.layer, self, layer)

    def set_bias(self, biases: Array) -> "VariantBiasLinear":
        """
        Set the bias terms.

        Parameters
        ----------
        biases : Array
            Bias array of shape (n_biases, observation_dim) or (observation_dim,).
            If 1-D, the bias is broadcast to all time steps.

        Returns
        -------
        VariantBiasLinear
            Updated layer with biases set.
        """
        if biases.ndim == 1:
            biases = jnp.broadcast_to(biases, self.biases.shape).copy()
        return eqx.tree_at(lambda r: r.biases, self, biases)


def gauss_rbf(x, c, s):
    """
    Gaussian radial basis function kernel.

    Parameters
    ----------
    x : Array
        Input point.
    c : Array
        Center point of the RBF kernel.
    s : Array
        Scale/width parameter for the kernel.

    Returns
    -------
    Scalar
        RBF kernel value: exp(-sum((x - c)^2 * s)).
    """
    return jnp.exp(-jnp.sum(jnp.square((x - c)) * s))


class RBFN(Module):
    """
    Radial Basis Function Network (RBFN).

    An RBFN uses radial basis functions as activation functions in a neural
    network. Each RBF is centered at a specific point in input space and
    the network output is a linear combination of the RBF responses.

    Parameters
    ----------
    input_size : int
        Dimension of the input space.
    output_size : int
        Dimension of the output space.
    network_size : int
        Number of RBF centers/hidden units.
    key : Array
        PRNGKey for parameter initialization.
    normalized : bool, default=False
        Whether to normalize the RBF outputs (currently unused).

    Attributes
    ----------
    centers : Array
        Fixed RBF center locations of shape (network_size, input_size).
    scale : Array
        Trainable scale parameters of shape (input_size,).
    readout : Module
        Linear layer for combining RBF outputs.
    """

    centers: Array = eqx.field(static=True)
    scale: Array
    readout: Module

    def __init__(
        self, input_size, output_size, network_size, *, key, normalized: bool = False
    ):
        """
        Initialize radial basis function network.

        Parameters
        ----------
        input_size : int
            Dimension of the input vectors.
        output_size : int
            Dimension of the output vectors.
        network_size : int
            Number of RBF centers/hidden units.
        key : Array
            PRNGKey for parameter initialization.
        normalized : bool, default=False
            Whether to normalize RBF outputs (currently unused).
        """
        key, ckey = jrnd.split(key)
        self.centers = jrnd.uniform(
            ckey, shape=(network_size, input_size), minval=-1, maxval=1
        )
        self.scale = jnp.ones(input_size)
        self.readout = enn.Linear(network_size, output_size, key=key)

    def __call__(self, x):
        """
        Compute RBFN output for input vector.

        Parameters
        ----------
        x : Array
            Input vector of shape (..., input_size).

        Returns
        -------
        Array
            Output vector of shape (..., output_size) computed as a
            linear combination of RBF kernel responses.
        """
        kernels = jax.vmap(gauss_rbf, in_axes=(None, 0, None))(
            x, self.centers, constraints.constrain_positive(self.scale)
        )
        return self.readout(kernels)  # type: ignore


class DataMasker(eqx.Module, strict=True):
    """
    Data masking module for regularization and robustness.

    Applies random binary masks to input data during training, which can
    help with regularization and improve model robustness to missing data.
    In inference mode, returns a mask of all ones (no masking).

    Parameters
    ----------
    p : float, default=0.5
        Probability of masking each element (0 means no masking, 1 means
        mask everything).
    inference : bool, default=False
        Whether to operate in inference mode (no random masking).

    Attributes
    ----------
    p : float
        Masking probability.
    inference : bool
        Inference mode flag.
    """

    p: float
    inference: bool

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
    ):
        """
        Initialize data masker.

        Parameters
        ----------
        p : float, default=0.5
            Probability of masking each element. Must be in [0, 1].
        inference : bool, default=False
            Whether to operate in deterministic inference mode.
        """
        self.p = p
        self.inference = inference

    @jax.named_scope("xfads.DataMasker")
    def __call__(
        self,
        x: Array,
        *,
        key: Array | None = None,
        inference: bool | None = None,
    ) -> Array:
        """
        Apply data masking to input array.

        Parameters
        ----------
        x : Array
            Input data array of shape (batch, time, features).
        key : Array or None, optional
            PRNGKey for random mask generation. Required when not in
            inference mode.
        inference : bool or None, optional
            Whether to use inference mode (deterministic). If None,
            uses the instance's inference setting.

        Returns
        -------
        mask : Array
            Binary mask array of shape (batch, time, 1) that can be
            broadcast with the input. In inference mode, returns all ones.

        Raises
        ------
        RuntimeError
            If key is None when running in non-deterministic mode.

        Notes
        -----
        The mask is generated with keep probability (1 - p), so p=0.5 means
        approximately half the elements will be masked out. The mask shape
        allows broadcasting across the feature dimension.
        """
        if inference is None:
            inference = self.inference

        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True

        shape = x.shape[:2] + (1,)  # broadcast to the last dimension
        if inference:
            return jnp.ones(shape)
        elif key is None:
            raise RuntimeError(
                f"{DataMasker.__name__} requires a key when running in non-deterministic mode."
            )
        else:
            q = 1 - jax.lax.stop_gradient(self.p)
            mask = jrnd.bernoulli(key, q, shape)
            return mask
