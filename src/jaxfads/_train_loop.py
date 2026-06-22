"""Training loop primitives for XFADS.

This module is a local copy of the generic training loop originally provided by
``gearax.trainer`` (authored by the same author). It is vendored here so the
training loop can evolve independently of the upstream package.

Public entry point: :func:`train`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
from jax import Array, lax
from jax import numpy as jnp
from jax import random as jr
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


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


def train(
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


__all__ = ["train", "Monitor"]
