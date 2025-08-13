"""
Training utilities for XFADS models.

This module provides training routines for XFADS models using JAX and Optax.
It implements efficient batch training with multi-device support, progress
tracking, and validation-based early stopping. The training is based on
maximizing the Evidence Lower Bound (ELBO) objective.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, random as jrnd, lax, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
import optax
import equinox as eqx
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from . import vi


def training_progress():
    """
    Create a Rich progress bar for training visualization.

    Returns
    -------
    Progress
        Configured Rich Progress instance with columns for:
        - Spinner animation
        - Task description
        - Progress counter
        - Elapsed time
        - Remaining time estimate
        - Current loss value
        - Moving average loss

    Notes
    -----
    The progress bar provides real-time feedback during training including
    current and smoothed loss values to monitor convergence.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TextColumn("•"),
        "Elapsed",
        TimeElapsedColumn(),
        TextColumn("•"),
        "Remaining",
        TimeRemainingColumn(),
        TextColumn("•"),
        "Loss",
        TextColumn("{task.fields[loss]:.3f}"),
        "MA Loss",
        TextColumn("{task.fields[mean]:.3f}"),
    )


@dataclass
class Opt:
    """
    Configuration dataclass for XFADS training hyperparameters.

    Parameters
    ----------
    min_iter : int, default=50
        Minimum number of training iterations before early stopping.
    max_iter : int, default=50
        Maximum number of training iterations.
    learning_rate : float, default=1e-3
        Learning rate for the optimizer.
    clip_norm : float, default=5.0
        Maximum gradient norm for gradient clipping.
    batch_size : int, default=1
        Batch size for training (will be adjusted for multi-device).
    weight_decay : float, default=1e-3
        L2 regularization coefficient.
    beta : float, default=0.95
        Exponential moving average coefficient for loss smoothing.
    seed : int, default=0
        Random seed for reproducibility.
    noise_eta : float, default=0.5
        Noise scale parameter for gradient noise injection.
    noise_gamma : float, default=0.8
        Noise decay parameter for gradient noise injection.
    valid_ratio : float, default=0.2
        Fraction of data to use for validation.
    validation_size : int, default=80
        Fixed validation set size (overrides valid_ratio if specified).

    Notes
    -----
    The configuration supports various regularization techniques:
    - Gradient clipping for training stability
    - Weight decay for parameter regularization
    - Gradient noise injection for better generalization
    - Validation-based early stopping
    """
    min_iter: int = 50
    max_iter: int = 50
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1
    weight_decay: float = 1e-3
    beta: float = 0.95
    seed: int = 0
    noise_eta: float = 0.5
    noise_gamma: float = 0.8
    valid_ratio: float = 0.2
    validation_size: int = 80


# >>> Full JAX
def train_test_split(arrays, *, rng, test_ratio=None, test_size=None, train_size=None):
    """
    Split arrays into training and test sets with random permutation.

    Parameters
    ----------
    arrays : tuple of Array
        Input arrays to split, all must have same first dimension.
    rng : numpy.random.Generator
        Random number generator for reproducible splits.
    test_ratio : float, optional
        Fraction of data to use for testing (ignored if test_size specified).
    test_size : int, optional
        Fixed number of samples for test set.
    train_size : int, optional
        Fixed number of samples for training set (computed if not specified).

    Returns
    -------
    train_arrays : tuple of Array
        Training set arrays.
    test_arrays : tuple of Array
        Test set arrays.

    Notes
    -----
    The function randomly permutes the data before splitting to ensure
    random sampling. If both test_size and test_ratio are specified,
    test_size takes precedence.
    """
    data_size = arrays[0].shape[0]
    if test_size is None:
        test_size = int(test_ratio * data_size)
    if train_size is None:
        train_size = data_size - test_size
    perm = rng.permutation(data_size)

    return tuple(
        array[perm[test_size : train_size + test_size]] for array in arrays
    ), tuple(array[perm[:test_size]] for array in arrays)


def to_shard(arrays, sharding=None):
    """
    Place arrays on specified devices with optional sharding.

    Parameters
    ----------
    arrays : tuple of Array
        Arrays to place on devices.
    sharding : Sharding, optional
        JAX sharding specification for multi-device placement.

    Returns
    -------
    tuple of Array
        Arrays placed on specified devices.

    Notes
    -----
    Used for efficient multi-device training by distributing data
    across available accelerators according to the sharding specification.
    """
    return tuple(jax.device_put(arr, sharding) for arr in arrays)


def batch_elbo(model, key, ts, moment_s, moment_p, ys) -> Array:
    """
    Compute Evidence Lower Bound (ELBO) for batched sequences.

    Vectorizes the ELBO computation across both batch and sequence dimensions
    to efficiently process multiple sequences simultaneously.

    Parameters
    ----------
    model : XFADS
        The XFADS model containing likelihood and hyperparameters.
    key : PRNGKeyArray
        Random key for stochastic computations.
    ts : Array, shape (T,)
        Time indices for the sequences.
    moment_s : Array, shape (N, T, param_dim)
        Posterior moment parameters for N sequences of length T.
    moment_p : Array, shape (N, T, param_dim)
        Prior/predictive moment parameters.
    ys : Array, shape (N, T, observation_dim)
        Observed data sequences.

    Returns
    -------
    Array, shape (N, T)
        ELBO values for each time point in each sequence.

    Notes
    -----
    The function uses jax.vmap to vectorize across both batch (N) and
    sequence (T) dimensions, generating appropriate random keys for
    each computation.
    """
    _elbo = jax.vmap(
        jax.vmap(
            partial(
                vi.elbo,
                eloglik=model.likelihood.eloglik,
                approx=model.hyperparam.approx,
                mc_size=model.hyperparam.mc_size,
            )
        )
    )  # (batch, seq)

    keys = jrnd.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)

    return _elbo(keys, ts, moment_s, moment_p, ys)


def train_fast(model, data, *, conf):
    """
    Fast training routine for XFADS models with multi-device support.

    Implements efficient training using JAX transformations, automatic
    differentiation, and multi-device data parallelism. Features include
    gradient clipping, weight decay, noise injection, and validation-based
    early stopping with exponential moving averages.

    Parameters
    ----------
    model : XFADS
        The XFADS model to train.
    data : tuple of Array
        Training data as tuple (t, y, u, c) where:
        - t: time indices, shape (N, T)
        - y: observations, shape (N, T, observation_dim)
        - u: control inputs, shape (N, T, input_dim)
        - c: covariates, shape (N, T, covariate_dim)
    conf : Opt
        Training configuration with hyperparameters.

    Returns
    -------
    XFADS
        Trained XFADS model with optimized parameters.

    Notes
    -----
    The training procedure follows these steps:

    1. **Data Preparation**: Split data into train/validation sets and
       distribute across available devices using JAX sharding.

    2. **Optimizer Setup**: Configure Optax optimizer chain with:
       - Gradient clipping for stability
       - Gradient noise injection for regularization
       - Adam optimizer with weight decay
       - Learning rate scaling

    3. **Training Loop**: Iterative optimization with:
       - Mini-batch gradient descent
       - Validation loss monitoring
       - Exponential moving average smoothing
       - Early stopping based on convergence criteria

    4. **Loss Computation**: Maximizes ELBO (Evidence Lower Bound):
       Loss = -E[log p(y|z)] + KL(q(z)||p(z)) + noise_penalty

    The implementation is optimized for performance with:
    - JIT compilation of critical functions
    - Efficient memory management with equinox
    - Multi-device data parallelism
    - Dynamic batch permutation for better mixing

    Examples
    --------
    >>> from omegaconf import DictConfig
    >>> conf = Opt(max_iter=1000, learning_rate=1e-3, batch_size=32)
    >>> trained_model = train_fast(model, (t, y, u, c), conf=conf)
    """
    key = jrnd.key(conf.seed)
    rng = np.random.default_rng(conf.seed)

    # >>> Prepare data
    n_devices = len(jax.devices())
    mesh = jax.make_mesh((n_devices,), ("batch",))
    sharding = NamedSharding(mesh, P("batch"))

    # batch size is required to be multiple of the number of devices
    # validation size is required to be multile of batch_size

    data_size = len(data[0])
    batch_size = int(conf.batch_size / n_devices) * n_devices
    valid_size = int(conf.validation_size / batch_size) * batch_size
    train_size = int((data_size - valid_size) / batch_size) * batch_size

    train_set, valid_set = train_test_split(
        data, rng=rng, test_size=valid_size, train_size=train_size
    )

    train_set = to_shard(train_set, sharding)
    valid_set = to_shard(valid_set, sharding)
    # <<<

    # >>> Prepare optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(conf.clip_norm),
        optax.add_noise(conf.noise_eta, conf.noise_gamma, conf.seed),
        optax.scale_by_adam(),
        optax.add_decayed_weights(conf.weight_decay),
        optax.scale_by_learning_rate(conf.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    # <<<

    # Define loss and grad
    @eqx.filter_jit
    def batch_loss(model, batch, key):
        """Compute negative ELBO loss for a batch of sequences."""
        tb, yb, ub, cb = batch

        key, ky = jrnd.split(key)
        _, moment, moment_p = model(tb, yb, ub, cb, key=ky)

        key, ky = jrnd.split(key)
        free_energy = -batch_elbo(model, ky, tb, moment, moment_p, yb)

        loss = (
            jnp.mean(free_energy)
            + model.hyperparam.noise_penalty * model.forward.loss()
            # + hyperparam.noise_penalty * model.backward.loss()
        )

        return loss

    @eqx.filter_jit
    def batch_grad_step(model, opt_state, batch, key):
        """Perform one gradient update step."""
        lss, grads = eqx.filter_value_and_grad(batch_loss)(
            model, batch, key
        )  # The gradient will be computed with respect to all floating-point JAX/NumPy arrays in the first argument
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return lss, model, opt_state

    # Main loop
    key, cey = jrnd.split(key)
    perm = jrnd.permutation(cey, train_size)
    min_iter = conf.min_iter
    max_iter = conf.max_iter
    beta = conf.beta

    with training_progress() as pbar:
        key, cey = jrnd.split(key)
        valid_loss = lax.stop_gradient(
            batch_loss(eqx.nn.inference_mode(model), valid_set, cey)
        )
        task_id = pbar.add_task(
            "Training", total=max_iter, loss=valid_loss, mean=valid_loss
        )

        params, static = eqx.partition(model, eqx.is_inexact_array)

        def train_cond(carry):
            """Training continuation condition."""
            params, opt_state, i, converged, *_ = carry
            return jnp.logical_and(
                i < max_iter, jnp.logical_or(jnp.logical_not(converged), i < min_iter)
            )

        def train_step(carry):
            """Single training step with validation."""
            params, opt_state, i, converged, mean_loss, key, idx, perm = carry

            def new_perm(key, perm, idx):
                perm = jrnd.permutation(key, train_size)
                return perm, 0

            def old_perm(key, perm, idx):
                return perm, idx

            key, cey = jrnd.split(key)
            perm, idx = lax.cond(
                idx + batch_size >= train_size, new_perm, old_perm, cey, perm, idx
            )

            model = eqx.combine(params, static)
            batch_idx = lax.dynamic_slice_in_dim(perm, idx, batch_size)
            batch = tuple(arr[batch_idx] for arr in train_set)
            key, cey = jrnd.split(key)
            _, model, opt_state = batch_grad_step(model, opt_state, batch, cey)

            key, cey = jrnd.split(key)
            valid_loss = lax.stop_gradient(
                batch_loss(eqx.nn.inference_mode(model), valid_set, cey)
            )
            jax.debug.callback(
                lambda vl, ml: pbar.update(task_id, advance=1, loss=vl, mean=ml),
                valid_loss,
                mean_loss,
            )

            params, _ = eqx.partition(model, eqx.is_inexact_array)
            converged = jnp.logical_and(
                jnp.isclose(mean_loss, valid_loss), valid_loss <= mean_loss
            )
            mean_loss = mean_loss * beta + valid_loss * (1 - beta)

            return (
                params,
                opt_state,
                i + 1,
                converged,
                mean_loss,
                key,
                idx + batch_size,
                perm,
            )

        key, cey = jrnd.split(key)
        params, *_ = lax.while_loop(
            train_cond,
            train_step,
            (params, opt_state, 0, False, valid_loss, cey, 0, perm),
        )
        model = eqx.combine(params, static)

    return model
