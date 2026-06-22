"""
Training utilities for XFADS models.

This module provides training routines for XFADS models using JAX and Optax.
It implements efficient batch training with multi-device support, progress
tracking, and validation-based early stopping. The training is based on
maximizing the Evidence Lower Bound (ELBO) objective.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import time
from typing import Any

import jax
import numpy as np
import optax
import equinox as eqx
from jax import Array, lax
from jax import numpy as jnp
from jax import random as jr
from jax import sharding as jshd
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import vi
from .logging import get_logger


logger = get_logger(__name__)

#: Default configuration for XFADS training hyperparameters.
#: Contains settings for optimization (learning_rate, clip_norm, weight_decay),
#: training schedule (min/max_iter, min/max_epoch, batch_size), early stopping
#: (patience, valid_ratio, validation_size), and noise injection (noise_eta, noise_gamma).
DEFAULT_TRAINER_CONFIG = DictConfig(
    {
        "min_iter": 0,
        "max_iter": 50,
        "min_epoch": 0,
        "max_epoch": 50,
        "learning_rate": 1e-3,
        "clip_norm": 5.0,
        "batch_size": 1,
        "weight_decay": 1e-3,
        "beta": 0.95,
        "seed": 0,
        "noise_eta": 0.5,
        "noise_gamma": 0.8,
        "valid_ratio": 0.2,
        "validation_size": 80,
        "patience": 10,
        "kl_warmup_steps": 0,
        # Optional user-provided regularizer: Callable[[XFADS], Array]
        "noise_regularizer": None,
        # Optional list of dot-separated attribute paths to freeze.
        # Example: ["noise_free", "unconstrained_prior_natural"]
        "freeze_paths": [],
    }
)


def _resolve_attr_path(obj, parts: tuple[str, ...]):
    cur = obj
    for name in parts:
        if not hasattr(cur, name):
            raise ValueError(f"Invalid freeze path: {'.'.join(parts)}")
        cur = getattr(cur, name)
    return cur


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


def batch_elbo(
    model, key, times, posterior_moments, predicted_moments, observations, *, beta=1.0
) -> Array:
    """
    Compute Evidence Lower Bound (ELBO) for batched sequences.

    Vectorizes the ELBO computation across both batch and sequence dimensions
    to efficiently process multiple sequences simultaneously.

    Parameters
    ----------
    model : XFADS
        The XFADS model containing observation model and hyperparameters.
    key : PRNGKeyArray
        Random key for stochastic computations.
    times : Array, shape (T,)
        Time indices for the sequences.
    posterior_moments : Array, shape (N, T, param_dim)
        Posterior moment parameters for N sequences of length T.
    predicted_moments : Array, shape (N, T, param_dim)
        Prior/predictive moment parameters.
    observations : Array, shape (N, T, observation_dim)
        Observed data sequences.
    beta : float, optional
        KL weight for warm-up annealing, in ``[0, 1]``.  Default is ``1.0``.

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
                eloglik=model.observation.eloglik,
                approx=model.approx,
                mc_size=model.conf.mc_size,
                beta=beta,
            )
        )
    )  # (batch, seq)

    keys = jr.split(key, observations.shape[:2])  # observations.shape[:2] + (2,)

    return _elbo(keys, times, posterior_moments, predicted_moments, observations)


def batch_loss(
    model,
    batch,
    key,
    step,
    *,
    kl_warmup_steps=0,
    noise_regularizer=None,
):
    """
    Compute negative ELBO loss for a batch of sequences.

    Parameters
    ----------
    model : XFADS
        Model providing `__call__` to produce posterior/prior means and
        configuration for loss terms.
    batch : tuple[Array, Array, Array, Array]
        Tuple `(times, observations, controls, covariates)` with shapes
        matching the training data layout.
    key : Array
        JAX PRNGKey used for stochastic components.
    step : Array
        Scalar ``jnp.int32`` training step counter provided by the
        training loop.
    kl_warmup_steps : int, optional
        Number of training steps over which the KL weight β is linearly
        annealed from 0 to 1.  When ``0`` (default) the standard ELBO is
        used (β = 1 from the start).

    Returns
    -------
    Array
        Scalar loss equal to the mean negative ELBO over the batch, plus
        optional regularization terms.
    """
    beta = jnp.where(
        kl_warmup_steps > 0,
        jnp.minimum(1.0, step / kl_warmup_steps),
        1.0,
    )

    times, observations, controls, covariates = batch

    key, model_key = jr.split(key)
    _, posterior_moments, prior_moments = model(
        times, observations, controls, covariates, key=model_key
    )

    key, elbo_key = jr.split(key)
    free_energy = -batch_elbo(
        model,
        elbo_key,
        times,
        posterior_moments,
        prior_moments,
        observations,
        beta=beta,
    )

    mean_fe = jnp.mean(free_energy)
    reg = (
        noise_regularizer(model)
        if noise_regularizer is not None
        else jnp.asarray(0.0, dtype=mean_fe.dtype)
    )
    return mean_fe + reg


def dataloader(arrays, batch_size, num_epochs, key, shuffle=True):
    """
    Yield mini-batches with epoch/batch counters.

    Parameters
    ----------
    arrays : tuple[Array, ...]
        Data arrays with equal first dimension (dataset size).
    batch_size : int
        Number of samples per batch.
    num_epochs : int
        Number of epochs to iterate (negative treated as 0).
    key : Array
        JAX PRNGKey for shuffling.
    shuffle : bool, optional
        Whether to shuffle indices each epoch. Default is ``True``.

    Yields
    ------
    batch : tuple[Array, ...]
        Batch slices from each input array.
    epoch : int
        Zero-based epoch index.
    batch_in_epoch : int
        Zero-based batch index within the epoch.
    """
    # Treat negative num_epochs as 0
    num_epochs = max(0, num_epochs)

    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)

    epoch = 0  # index of epoch

    def single_epoch_generator(epoch_key):
        perm = jr.permutation(epoch_key, indices) if shuffle else indices
        start = 0
        end = batch_size
        batch_in_epoch = 0
        while end <= dataset_size:
            batch_perm = perm[start:end]
            batch_data = tuple(array[batch_perm] for array in arrays)
            yield batch_data, epoch, batch_in_epoch
            batch_in_epoch += 1
            start = end
            end = start + batch_size

    # Finite number of epochs
    for _ in range(num_epochs):
        key, epoch_key = jr.split(key)
        yield from single_epoch_generator(epoch_key)
        epoch += 1


def compute_patience(max_epoch, data_size, batch_size, scale=0.1):
    """
    Compute early-stopping patience (in epochs) from total training steps.

    Parameters
    ----------
    max_epoch : int
        Maximum number of epochs configured for training.
    data_size : int
        Number of samples in the dataset.
    batch_size : int
        Mini-batch size.
    scale : float, optional
        Fraction of total training steps used as patience. Default is ``0.1``.

    Returns
    -------
    int
        Patience measured in epochs (at least ``1``).
    """
    n_batches = data_size // batch_size
    total_steps = max_epoch * n_batches
    patience_steps = int(total_steps * scale)
    patience_epochs = max(1, patience_steps // n_batches)
    return patience_epochs


def _training_progress() -> Progress:
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
        - Best observed loss

    Notes
    -----
    The progress bar provides real-time feedback during training including
    the instantaneous loss and the best loss encountered so far.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        "Epoch",
        MofNCompleteColumn(),
        # TextColumn("•"),
        "Elapsed",
        TimeElapsedColumn(),
        "Remaining",
        TimeRemainingColumn(),
        "Loss",
        TextColumn("{task.fields[loss]:.3f}"),
        "Best",
        TextColumn("{task.fields[best]:.3f}"),
    )


def _copy_pytree(pt):
    """Return a defensive copy of a pytree, copying only array leaves."""
    return jax.tree.map(lambda x: jnp.copy(x) if eqx.is_array(x) else x, pt)


@dataclass
class Monitor:
    """Early-stopping helper that tracks validation performance.

    Attributes
    ----------
    evaluate : Callable
        Function computing the validation loss given a model, dataset, PRNG key,
        and training step.
    valid_set : Any
        Validation data prepared for `evaluate`.
    patience : int
        Number of epochs to wait before stopping once the loss stalls.
    patience_left : int
        Remaining epochs before early stopping triggers.
    max_epoch : int
        Maximum number of epochs to display in the progress bar.
    min_epoch : int
        Minimum number of epochs that must elapse before early stopping engages.
    best_model : eqx.Module
        Snapshot of the best model parameters encountered so far.
    best_loss : float
        Best validation loss recorded throughout training.
    losses : list
        History of validation losses across epochs.
    _pbar : Progress
        Rich progress bar used for monitoring.
    """

    evaluate: Callable
    valid_set: Any
    patience: int
    best_model: eqx.Module
    best_loss: float
    callback: Callable | None = None
    patience_left: int = field(init=False)
    losses: list = field(init=False, default_factory=list)
    _pbar: Any = field(init=False)

    def __init__(
        self, model, valid_set, eval_fun, max_epoch, patience, min_epoch: int = 0
    ) -> None:
        """Initialize the monitor and attach a Rich progress bar.

        Parameters
        ----------
        model : eqx.Module
            Model state to track as the current baseline.
        valid_set : Any
            Validation data passed to `eval_fun`.
        eval_fun : Callable
            Callable with signature ``(model, valid_set, key, step) -> Array``
            returning the loss.
        max_epoch : int
            Maximum number of epochs to display in the progress bar.
        patience : int
            Number of epochs to wait without improvement before stopping.
        min_epoch : int, optional
            Minimum number of epochs that must elapse before early stopping engages.
        """
        self.evaluate = eval_fun
        self.valid_set = valid_set
        self.patience = patience
        self.patience_left = patience
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch

        self.best_model = _copy_pytree(model)
        self.best_loss = jnp.inf
        self.losses = []

        self._pbar = _training_progress()
        self._task_id = self._pbar.add_task(
            "Training", total=max_epoch, loss=jnp.inf, best=jnp.inf
        )
        self._pbar.start()

    def step(self, model, key: Array, step: Array | None = None) -> bool:
        val_loss = self.evaluate(model, self.valid_set, key, step).item()
        self.losses.append(val_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = _copy_pytree(model)
            self.patience_left = self.patience
        else:
            if len(self.losses) > self.min_epoch:
                self.patience_left -= 1

        self._pbar.update(self._task_id, advance=1, loss=val_loss, best=self.best_loss)

        return self.patience_left > 0

    def stop(self) -> None:
        self._pbar.stop()


def _run_training_loop(
    model,
    train_set,
    valid_set,
    key,
    batch_loss_fun,
    dataloader,
    batch_size,
    max_epoch,
    patience,
    optimizer,
    data_sharding,
    model_sharding,
    min_epoch: int = 0,
):
    """
    Train a model with early stopping and sharded data/model execution.

    Parameters
    ----------
    model : eqx.Module
        Model to optimise; may contain PyTree leaves requiring sharding.
    train_set : Any
        Training dataset consumed by `dataloader`.
    valid_set : Any
        Validation dataset used for early-stopping evaluation.
    key : Array
        Base PRNG key; internally split for data loading and evaluation.
    batch_loss_fun : Callable
        Function computing the loss for a ``(model, batch, key, step)`` call,
        where *step* is a scalar ``jnp.int32`` counting training batches
        processed so far (starting from 0).  During validation the current
        training step is forwarded unchanged so that any step-dependent
        schedule (e.g. KL warm-up) stays consistent.
    dataloader : Callable
        Generator producing `(batch, epoch, batch_in_epoch)` tuples for training.
    batch_size : int
        Size of each training batch.
    max_epoch : int
        Maximum number of epochs to train for.
    patience : int
        Early-stopping patience supplied to the `Monitor`.
    optimizer : Any
        Optimiser matching the Equinox Optax-like interface with `init`/`update`.
    data_sharding : Any
        Partitioning specification applied to batch data via `eqx.filter_shard`.
    model_sharding : Any
        Partitioning specification applied to the model and optimiser state.
    min_epoch : int, optional
        Minimum number of epochs that must run before early stopping can trigger.

    Returns
    -------
    eqx.Module
        Copy of the best-performing model encountered during training.
    """

    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, batch, key, step):
        """One optimization step: shard inputs, compute gradients, and update model."""
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)

        grads = eqx.filter_grad(batch_loss_fun)(model, batch, key, step)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

        return model, opt_state, step + 1

    @eqx.filter_jit
    def evaluate(model, batch, key, step):
        """Sharded validation step that runs the loss function in inference mode."""
        model = eqx.filter_shard(eqx.nn.inference_mode(model), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)
        return lax.stop_gradient(batch_loss_fun(model, batch, key, step))

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # put on device
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
    valid_set = eqx.filter_shard(valid_set, data_sharding)

    monitor = Monitor(
        model,
        valid_set,
        evaluate,
        max_epoch,
        patience,
        min_epoch,
    )

    # Training loop with per-epoch validation and best model tracking
    step = jnp.array(0, dtype=jnp.int32)
    key, loader_key = jr.split(key)  # Key for dataloader
    for batch, epoch, batch_in_epoch in dataloader(
        train_set, batch_size, max_epoch, loader_key
    ):
        try:
            key, batch_key = jr.split(key)
            batch = eqx.filter_shard(batch, data_sharding)
            model, opt_state, step = train_step(model, opt_state, batch, batch_key, step)

            # Evaluate at the start of each new epoch
            if batch_in_epoch == 0:
                # Evaluate on validation set only
                key, monitor_key = jr.split(key)
                if not monitor.step(model, monitor_key, step) and epoch >= min_epoch:
                    break

        except KeyboardInterrupt:
            break
    else:
        # Final validation check
        key, monitor_key = jr.split(key)
        monitor.step(model, monitor_key, step)

    monitor.stop()

    return monitor.best_model


def train(model, data, *, conf):
    """
    Training routine for XFADS models with multi-device support.

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
    conf : dict or DictConfig
        Training configuration with hyperparameters. If dict or partial config,
        missing values will be filled with defaults from DEFAULT_TRAINER_CONFIG.

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
       Loss = -E[log p(y|z)] + KL(q(z)||p(z)) + optional regularizers

    The implementation is optimized for performance with:
    - JIT compilation of critical functions
    - Efficient memory management with equinox
    - Multi-device data parallelism
    - Dynamic batch permutation for better mixing
    """
    user_set_patience = "patience" in conf
    conf = OmegaConf.merge(DEFAULT_TRAINER_CONFIG, conf)

    t0 = time.perf_counter()

    key = jr.key(conf.seed)
    rng = np.random.default_rng(conf.seed)

    # >>> Prepare sharding
    n_devices = len(jax.devices())
    mesh = jax.make_mesh((n_devices,), ("batch",))
    data_sharding = jshd.NamedSharding(mesh, jshd.PartitionSpec("batch"))
    model_sharding = jshd.NamedSharding(mesh, jshd.PartitionSpec())

    # Prepare data
    # batch size is required to be multiple of the number of devices
    # validation size is required to be multile of batch_size
    data_size = len(data[0])
    batch_size = conf.batch_size
    if conf.validation_size > 0:
        valid_size = conf.validation_size
    else:
        valid_size = int(data_size * conf.valid_ratio)
    train_size = data_size - valid_size

    train_set, valid_set = train_test_split(
        data, rng=rng, test_size=valid_size, train_size=train_size
    )
    if not user_set_patience:
        conf.patience = compute_patience(conf.max_epoch, data_size, batch_size)

    logger.info(
        "train start: devices=%d batch_size=%d data=%d train=%d valid=%d max_epoch=%d patience=%d seed=%d kl_warmup_steps=%d",
        n_devices,
        int(conf.batch_size),
        int(data_size),
        int(train_size),
        int(valid_size),
        int(conf.max_epoch),
        int(conf.patience),
        int(conf.seed),
        int(conf.kl_warmup_steps),
    )
    logger.debug(
        "sharding: data=%s model=%s",
        data_sharding.spec,
        model_sharding.spec,
    )
    logger.debug(
        "optimizer: lr=%s clip_norm=%s weight_decay=%s noise_eta=%s noise_gamma=%s",
        conf.learning_rate,
        conf.clip_norm,
        conf.weight_decay,
        conf.noise_eta,
        conf.noise_gamma,
    )

    # Prepare optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(conf.clip_norm),
        optax.add_noise(conf.noise_eta, conf.noise_gamma, conf.seed),
        optax.scale_by_adam(),
        optax.add_decayed_weights(conf.weight_decay),
        optax.scale_by_learning_rate(conf.learning_rate),
    )

    freeze_mask = jax.tree.map(lambda _: False, model)
    freeze_paths = [str(p) for p in conf.freeze_paths]
    for path in freeze_paths:
        parts = tuple(path.split("."))
        _ = _resolve_attr_path(model, parts)  # fail fast if path is invalid

        def getter(m, _parts=parts):
            return _resolve_attr_path(m, _parts)

        freeze_mask = eqx.tree_at(getter, freeze_mask, True)

    if len(freeze_paths) > 0:
        optimizer = optax.chain(
            optimizer,
            optax.masked(optax.set_to_zero(), freeze_mask),
        )

    def loss_fn(model, batch, key, step):
        return batch_loss(
            model,
            batch,
            key,
            step,
            kl_warmup_steps=conf.kl_warmup_steps,
            noise_regularizer=conf.noise_regularizer,
        )

    model = _run_training_loop(
        model,
        train_set,
        valid_set,
        key,
        loss_fn,
        dataloader,
        conf.batch_size,
        conf.max_epoch,
        conf.patience,
        optimizer,
        data_sharding,
        model_sharding,
        conf.min_epoch,
    )

    dt = time.perf_counter() - t0
    logger.info("train end: elapsed=%.2fs", dt)

    return model
