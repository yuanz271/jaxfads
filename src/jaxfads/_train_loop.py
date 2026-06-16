"""Training loop for jaxfads.

Originally vendored from gearax 0.1.0 (yuanz271), then refactored for jaxfads's
diagnostic workflow:

  * **No validation split / early stopping** — train the full ``max_epoch``
    budget on all data and return the final model.
  * **Periodic checkpoints** — optional ``checkpoint_callback`` invoked at each
    epoch boundary (caller persists every N epochs).
  * **Per-epoch training metrics** — optional ``metrics_callback`` receives the
    mean training loss and gradient norm per epoch, for diagnosis (e.g. spotting
    free-run divergence onset). The ``dataloader`` is supplied by
    ``jaxfads.trainer``.
"""

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import optax
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
    )


def train(
    model,
    train_set,
    key,
    batch_loss_fun,
    dataloader,
    batch_size,
    max_epoch,
    optimizer,
    data_sharding,
    model_sharding,
    *,
    checkpoint_callback: Callable | None = None,
    metrics_callback: Callable | None = None,
    model_callback: Callable | None = None,
) -> Any:
    """Train for the full ``max_epoch`` budget (no validation / early stopping).

    Parameters mirror the optimiser/sharding setup of the previous loop. The two
    optional callbacks fire once per completed epoch:

    * ``checkpoint_callback(model, epoch, step)`` — persist a checkpoint.
    * ``metrics_callback(epoch, step, train_loss, grad_norm)`` — record diagnostics.
    * ``model_callback(model, epoch, step) -> model`` — optional epoch-boundary
      model update, e.g. externally scheduled frozen parameters.

    Returns the **final** model (not a best-validation snapshot).
    """

    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, batch, key, step):
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)

        loss, grads = eqx.filter_value_and_grad(batch_loss_fun)(model, batch, key, step)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_inexact_array))
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, step + 1, loss, grad_norm

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    pbar = _training_progress()
    task_id = pbar.add_task("Training", total=max_epoch, loss=jnp.inf)
    pbar.start()

    def _finish_epoch(epoch, loss_sum, gnorm_sum, n, model, step):
        mean_loss = loss_sum / max(n, 1)
        mean_gnorm = gnorm_sum / max(n, 1)
        pbar.update(task_id, advance=1, loss=mean_loss)
        if metrics_callback is not None:
            metrics_callback(int(epoch), int(step), mean_loss, mean_gnorm)
        if checkpoint_callback is not None:
            checkpoint_callback(model, int(epoch), int(step))

    step = jnp.array(0, dtype=jnp.int32)
    key, loader_key = jr.split(key)
    if model_callback is not None:
        model = model_callback(model, 0, 0)
    prev_epoch = 0
    loss_sum = gnorm_sum = 0.0
    n_batches = 0
    for batch, epoch, batch_in_epoch in dataloader(train_set, batch_size, max_epoch, loader_key):
        try:
            if epoch != prev_epoch:  # previous epoch completed
                _finish_epoch(prev_epoch, loss_sum, gnorm_sum, n_batches, model, step)
                loss_sum = gnorm_sum = 0.0
                n_batches = 0
                prev_epoch = epoch
                if model_callback is not None:
                    model = model_callback(model, int(epoch), int(step))

            key, batch_key = jr.split(key)
            model, opt_state, step, loss, grad_norm = train_step(
                model, opt_state, batch, batch_key, step
            )
            loss_f = float(loss)
            loss_sum += loss_f
            gnorm_sum += float(grad_norm)
            n_batches += 1
            # NaN-guard: a non-finite loss never recovers; stop early instead of
            # burning the rest of the epoch budget (still records the diverged epoch).
            if not math.isfinite(loss_f):
                print(f"[train] non-finite loss at epoch {int(epoch)} step {int(step)}; stopping early")
                break
        except KeyboardInterrupt:
            break

    _finish_epoch(prev_epoch, loss_sum, gnorm_sum, n_batches, model, step)  # final epoch
    pbar.stop()
    return model
