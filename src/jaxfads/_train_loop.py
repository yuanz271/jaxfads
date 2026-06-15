"""Training loop for jaxfads, vendored from gearax 0.1.0 (yuanz271).

Adapted from ``gearax.trainer`` (same author) and brought in-tree so jaxfads
owns its training loop. The only functional addition over the upstream loop is
an optional ``checkpoint_callback`` invoked at each epoch boundary, which lets
callers persist periodic checkpoints during training (best-val return is
unchanged). The ``dataloader`` remains supplied by ``jaxfads.trainer``.
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
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        "Epoch",
        MofNCompleteColumn(),
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
    """Early-stopping helper that tracks validation performance."""

    evaluate: Callable
    valid_set: Any
    patience: int
    best_model: eqx.Module
    best_loss: float
    patience_left: int = field(init=False)
    losses: list = field(init=False, default_factory=list)
    _pbar: Any = field(init=False)

    def __init__(self, model, valid_set, eval_fun, max_epoch, patience, min_epoch: int = 0) -> None:
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
    checkpoint_callback: Callable | None = None,
):
    """Train with early stopping + sharded execution.

    ``checkpoint_callback(model, epoch, step)`` (if given) is called once at the
    start of each epoch, letting the caller persist periodic checkpoints. It is
    purely a side-effecting hook and does not affect the returned best-val model.
    """

    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, batch, key, step):
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)

        grads = eqx.filter_grad(batch_loss_fun)(model, batch, key, step)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, step + 1

    @eqx.filter_jit
    def evaluate(model, batch, key, step):
        model = eqx.filter_shard(eqx.nn.inference_mode(model), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)
        return lax.stop_gradient(batch_loss_fun(model, batch, key, step))

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
    valid_set = eqx.filter_shard(valid_set, data_sharding)

    monitor = Monitor(model, valid_set, evaluate, max_epoch, patience, min_epoch)

    step = jnp.array(0, dtype=jnp.int32)
    key, loader_key = jr.split(key)
    for batch, epoch, batch_in_epoch in dataloader(train_set, batch_size, max_epoch, loader_key):
        try:
            key, batch_key = jr.split(key)
            batch = eqx.filter_shard(batch, data_sharding)
            model, opt_state, step = train_step(model, opt_state, batch, batch_key, step)

            if batch_in_epoch == 0:
                if checkpoint_callback is not None:
                    checkpoint_callback(model, int(epoch), int(step))
                key, monitor_key = jr.split(key)
                if not monitor.step(model, monitor_key, step) and epoch >= min_epoch:
                    break

        except KeyboardInterrupt:
            break
    else:
        key, monitor_key = jr.split(key)
        monitor.step(model, monitor_key, step)

    monitor.stop()

    return monitor.best_model
