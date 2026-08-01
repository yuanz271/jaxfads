"""
Training utilities for XFADS models.

This module provides training routines for XFADS models using JAX and Optax.
It implements efficient batch training with multi-device support, progress
tracking, and validation-based early stopping. The training is based on
maximizing the Evidence Lower Bound (ELBO) objective.
"""

import json
import math
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import optax
from gearax.modules import save_model
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
#: Contains the single default-optimizer setting (learning_rate), the training
#: schedule (max_epoch, batch_size), and KL warm-up (kl_warmup_steps). The
#: default optimizer is vanilla Adam; pass ``optimizer=`` to :func:`train` for
#: anything else (gradient clipping, weight decay, custom schedules).
#: Validation, checkpointing, and early stopping are not part of training
#: config; they live in handlers (see :class:`EpochHandler`).
DEFAULT_TRAINER_CONFIG = DictConfig(
    {
        "max_epoch": 50,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "seed": 0,
        "kl_warmup_steps": 0,
        # Optional list of dot-separated attribute paths to freeze.
        # Example: ["noise.free", "unconstrained_prior_natural"]
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
    model,
    key,
    times,
    posterior_moments,
    predicted_moments,
    observations,
    *,
    beta,
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
        KL weight in ``[0, 1]`` (required). ``beta = 1`` is the standard,
        un-annealed ELBO; warm-up supplies a value ramping up to it.

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
    *,
    beta=1.0,
):
    """
    Compute negative ELBO loss for a batch of sequences.

    This is a pure objective evaluator: it depends only on ``(model, batch,
    key, beta)``. It has no notion of training step, KL warm-up, or
    regularization -- the KL weight ``beta`` is supplied by the caller, and any
    parameter penalty is composed by the trainer (see :func:`train`).

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
    beta : float or Array, optional
        KL weight in ``[0, 1]``. Default ``1.0`` (standard, un-annealed ELBO).

    Returns
    -------
    Array
        Scalar mean negative ELBO over the batch.
    """
    times, observations, controls, covariates = batch

    key, model_key = jr.split(key)
    _, posterior_moments, prior_moments, _transition_stat = model(
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

    return jnp.mean(free_energy)


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
        - Current (per-epoch) training loss

    Notes
    -----
    The progress bar provides real-time feedback during training, showing the
    per-epoch training loss.
    """
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


def _copy_pytree(pt):
    """Return a defensive copy of a pytree, copying only array leaves."""
    return jax.tree.map(lambda x: jnp.copy(x) if eqx.is_array(x) else x, pt)


class EpochHandler:
    """Self-contained ``on_epoch_end`` handler.

    Owns all epoch-level policy so the training loop stays a pure mechanism:
    validation, best-model tracking, periodic checkpointing, metrics
    persistence, and optional early stopping. Construct it yourself and pass it
    as ``on_epoch_end`` to :func:`train`; read :attr:`best_model` afterwards.

    Validation is fully owned here: given ``valid_data``, the handler builds its
    own jitted evaluation from :func:`batch_loss` (``beta=1``, no regularizer),
    so JAX concerns stay out of user code.

    All behavior is opt-in:

    - validation and best-model tracking are enabled whenever ``valid_data`` is
      given (best tracking is automatic; there is no separate flag)
    - ``checkpoint_path is None`` disables checkpoint/metric writing
    - ``patience is None`` disables early stopping

    Parameters
    ----------
    valid_data : tuple or None
        Validation ``(t, y, u, c)`` evaluated as a single batch each epoch.
        When given, the best-by-validation model is tracked automatically in
        :attr:`best_model`.
    checkpoint_path : str or Path or None
        Directory for checkpoints/metrics/config. Created if missing.
    checkpoint_every : int or None
        Save the current model every ``checkpoint_every`` epochs.
    patience : int or None
        Epochs without validation improvement before requesting a stop.
    save_fn : Callable or None
        ``save_fn(model, path)`` persisting a model; defaults to ``save_model``.
    config : Any or None
        Optional resolved config dumped to ``checkpoint_path/config.yaml``.
    data_sharding, model_sharding : Any or None
        Optional sharding for the validation step; defaults to replicated.
    seed : int
        Seed for the validation PRNG key.
    """

    def __init__(
        self,
        *,
        valid_data=None,
        checkpoint_path=None,
        checkpoint_every=None,
        patience=None,
        save_fn: Callable | None = None,
        config=None,
        data_sharding=None,
        model_sharding=None,
        seed: int = 0,
    ) -> None:
        self.valid_data = valid_data
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        )
        self.checkpoint_every = checkpoint_every
        self.patience = patience
        self.save_fn = save_fn if save_fn is not None else (lambda m, p: save_model(p, m))
        self.data_sharding = data_sharding
        self.model_sharding = model_sharding
        self.key = jr.key(seed)

        self.has_valid = valid_data is not None
        self.best_loss = float("inf")
        self.best_model = None
        self.patience_left = patience
        self.valid_losses: list = []

        if self.has_valid:
            self._evaluate = self._build_evaluate()
            if self.data_sharding is not None:
                self.valid_data = eqx.filter_shard(self.valid_data, self.data_sharding)

        if self.checkpoint_path is not None:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
            if config is not None:
                OmegaConf.save(config, self.checkpoint_path / "config.yaml")

    def _build_evaluate(self) -> Callable:
        model_sharding = self.model_sharding
        data_sharding = self.data_sharding

        @eqx.filter_jit
        def _evaluate(model, batch, key):
            model = eqx.nn.inference_mode(model)
            if model_sharding is not None:
                model = eqx.filter_shard(model, model_sharding)
            if data_sharding is not None:
                batch = eqx.filter_shard(batch, data_sharding)
            # Validation uses the true ELBO (beta = 1), no regularizer.
            return lax.stop_gradient(batch_loss(model, batch, key, beta=1.0))

        return _evaluate

    def __call__(self, model, info) -> bool:
        epoch = info["epoch"]

        valid_loss = None
        if self.has_valid:
            eval_key = jr.fold_in(self.key, epoch)
            valid_loss = float(self._evaluate(model, self.valid_data, eval_key))
            self.valid_losses.append(valid_loss)

        improved = False
        if valid_loss is not None and math.isfinite(valid_loss):
            # Seed best_model on the first finite validation loss, then track
            # subsequent improvements. A never-finite run leaves best_model None
            # to signal a failed/diverged training.
            if self.best_model is None or valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_model = _copy_pytree(model)
                self.patience_left = self.patience
                improved = True
            elif self.patience is not None:
                self.patience_left -= 1

        if self.checkpoint_path is not None:
            if self.checkpoint_every and (epoch + 1) % self.checkpoint_every == 0:
                self.save_fn(
                    model, self.checkpoint_path / f"checkpoint_epoch{epoch:04d}.zip"
                )
            if improved:
                self.save_fn(model, self.checkpoint_path / "best.zip")
            self._write_metrics(info)

        if self.patience is not None and self.patience_left is not None:
            return self.patience_left <= 0
        return False

    def _write_metrics(self, info) -> None:
        metrics = {
            "train_losses": info["train_losses"],
            "valid_losses": self.valid_losses,
        }
        (self.checkpoint_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )


def _run_training_loop(
    model,
    train_set,
    key,
    dataloader,
    batch_size,
    max_epoch,
    optimizer,
    data_sharding,
    model_sharding,
    on_epoch_end=None,
    param_schedule=None,
    beta_schedule=None,
    regularizer=None,
):
    """Run direct post-SGD minibatch R/Q M-steps alongside minibatch SGD.

    Each step computes loss and gradients, applies SGD, then performs one
    fresh inference pass over the same batch to derive and apply its direct
    R/Q statistic. Statistics are discarded immediately; no epoch accumulator
    or full-data inference is used by normal training.
    """

    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, batch, key, step):
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)
        if param_schedule is not None:
            model = param_schedule(model, step)

        beta = beta_schedule(step) if beta_schedule is not None else 1.0

        def loss_fn(current_model):
            loss = batch_loss(current_model, batch, key, beta=beta)
            if regularizer is not None:
                loss = loss + regularizer(current_model)
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        key, stat_key = jr.split(key)
        t, y, u, c = batch
        model = model.mstep(model._batch_stat(t, y, u, c, key=stat_key))
        return model, opt_state, step + 1, loss

    opt_state = optimizer.init(_copy_pytree(eqx.filter(model, eqx.is_inexact_array)))
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    train_losses: list = []
    pbar = _training_progress()
    task_id = pbar.add_task("Training", total=max_epoch, loss=jnp.inf)
    pbar.start()

    step = jnp.array(0, dtype=jnp.int32)
    key, loader_key = jr.split(key)
    current_epoch = 0
    epoch_batch_losses: list = []

    def finalize_epoch(epoch_idx, batch_losses):
        train_loss = (
            float(jnp.mean(jnp.stack(batch_losses)))
            if batch_losses
            else float("nan")
        )
        train_losses.append(train_loss)
        pbar.update(task_id, advance=1, loss=train_loss)
        if on_epoch_end is None:
            return False
        info = {
            "epoch": epoch_idx,
            "step": step,
            "train_loss": train_loss,
            "train_losses": train_losses,
        }
        return bool(on_epoch_end(model, info))

    try:
        for batch, epoch, _batch_in_epoch in dataloader(
            train_set, batch_size, max_epoch, loader_key
        ):
            if epoch != current_epoch:
                if finalize_epoch(current_epoch, epoch_batch_losses):
                    break
                epoch_batch_losses = []
                current_epoch = epoch

            key, batch_key = jr.split(key)
            model, opt_state, step, loss = train_step(
                model, opt_state, batch, batch_key, step
            )
            epoch_batch_losses.append(loss)
        else:
            finalize_epoch(current_epoch, epoch_batch_losses)
    except KeyboardInterrupt:
        logger.info(
            "training interrupted; discarding partial epoch M-step statistics"
        )
    finally:
        pbar.stop()

    return model


def train(
    model,
    train_data,
    *,
    conf,
    on_epoch_end=None,
    regularizer=None,
    optimizer=None,
    param_schedule=None,
):
    """
    Training routine for XFADS models with multi-device support.

    The trainer is a pure mechanism: it optimizes ``model`` on ``train_data``
    for ``conf.max_epoch`` epochs and returns the final-epoch model. It has no
    notion of validation, checkpointing, best models, or early stopping --
    those are epoch-level policy supplied via ``on_epoch_end`` (see
    :class:`EpochHandler`). The caller owns the train/validation split.

Each batch's existing pre-SGD ELBO forward pass emits additive observation
statistics and, when ``model.conf.q_mstep`` is true, transition-noise
statistics. The trainer accumulates them over the epoch and finalizes R and
enabled Q once at the epoch boundary before callbacks/checkpoints, without an
additional inference pass. For a Gaussian-likelihood model R is therefore
M-step-owned and excluded from SGD via
``model.frozen_paths()``. When ``q_mstep`` is true and Noise owns Q,
``noise.free`` is excluded the same way; when false, it remains
SGD-managed. See [mstep_dynamics_noise](../docs/mstep_dynamics_noise.md).

    Parameters
    ----------
    model : XFADS
        The XFADS model to train.
    train_data : tuple of Array
        Training data as tuple (t, y, u, c) where:
        - t: time indices, shape (N, T)
        - y: observations, shape (N, T, observation_dim)
        - u: control inputs, shape (N, T, input_dim)
        - c: covariates, shape (N, T, covariate_dim)
    conf : dict or DictConfig
        Training configuration with hyperparameters. If dict or partial config,
        missing values will be filled with defaults from DEFAULT_TRAINER_CONFIG.
    on_epoch_end : Callable or None
        ``on_epoch_end(model, info)`` called once per finished epoch with
        train-only ``info`` (``epoch``, ``step``, ``train_loss``,
        ``train_losses``); returning a truthy value stops training.
    regularizer : Callable or None, optional
        Optional ``regularizer(model) -> Array`` scalar penalty added to the
        per-batch objective (``loss = -ELBO + regularizer(model)``). It is a
        pure function of the model, so any parameter penalty must be written
        in the relevant parameter space (e.g. a penalty on the process-noise
        covariance Q must decode ``noise.free`` through ``model.noise``, not act on
        the raw free parameters).
    optimizer : optax.GradientTransformation or None, optional
        Optimizer to use. When ``None`` (default), the built-in optimizer is
        **vanilla Adam** (``optax.adam(conf.learning_rate)``) -- no gradient
        clipping, no gradient noise, no weight decay. In a plugin framework
        the trainer cannot know which leaves are weight matrices vs
        variances/biases, and gradient noise/clipping can destabilize
        sensitive objectives, so the default imposes no such policy. To add
        clipping, weight decay, gradient noise, or a custom schedule, build
        your own ``optax`` optimizer and pass it here; ``conf.learning_rate``
        is then ignored. ``conf.freeze_paths`` is still applied on top.
    param_schedule : Callable or None, optional
        Optional ``param_schedule(model, step) -> model`` applied at the start
        of every step (before the loss/gradient computation), for driving a
        model attribute through a step-indexed schedule -- e.g. annealing the
        process-noise scale. This is a general mechanism, not specific to any
        one attribute: the trainer only calls
        the function and does not interpret what it changes. The
        corresponding path(s) should also be listed in ``conf.freeze_paths``,
        otherwise the optimizer's own gradient-based update will fight the
        schedule (e.g. via gradient noise or optimizer momentum, even where
        the raw gradient is itself zero).
    Returns
    -------
    XFADS
        The final-epoch model.
    """
    conf = OmegaConf.merge(DEFAULT_TRAINER_CONFIG, conf)

    t0 = time.perf_counter()

    key = jr.key(conf.seed)

    # >>> Prepare sharding
    n_devices = len(jax.devices())
    # axis_types=(Auto,): jax's mesh axes default to Explicit since ~0.9.0 (the
    # new "sharding-in-types" mode), under which lax.with_sharding_constraint
    # (used by eqx.filter_shard below) rejects any spec, raising "only refer to
    # Auto axes of the mesh ... meant to use the reshard API?". Requesting Auto
    # axes explicitly restores with_sharding_constraint's pre-0.9 behavior.
    mesh = jax.make_mesh((n_devices,), ("batch",), axis_types=(jax.sharding.AxisType.Auto,))
    data_sharding = jshd.NamedSharding(mesh, jshd.PartitionSpec("batch"))
    model_sharding = jshd.NamedSharding(mesh, jshd.PartitionSpec())

    # batch size is required to be a multiple of the number of devices.
    data_size = len(train_data[0])

    logger.info(
        "train start: devices=%d batch_size=%d data=%d max_epoch=%d seed=%d kl_warmup_steps=%d",
        n_devices,
        int(conf.batch_size),
        int(data_size),
        int(conf.max_epoch),
        int(conf.seed),
        int(conf.kl_warmup_steps),
    )
    logger.debug(
        "sharding: data=%s model=%s",
        data_sharding.spec,
        model_sharding.spec,
    )
    logger.debug(
        "optimizer: %s lr=%s",
        "user-supplied" if optimizer is not None else "default (vanilla Adam)",
        conf.learning_rate,
    )

    # Prepare optimizer. The default is deliberately **vanilla Adam** -- no
    # gradient clipping, no gradient noise, no weight decay. In a plugin
    # framework the trainer cannot know which leaves are weights vs
    # variances/biases, so it imposes no regularization policy; and gradient
    # noise/clipping can destabilize sensitive objectives (e.g. chaotic
    # dynamical-systems reconstruction). Users who want clipping, weight
    # decay, gradient noise, or a custom schedule pass their own ``optax``
    # optimizer via ``optimizer=`` (``freeze_paths`` is still applied on top).
    if optimizer is None:
        optimizer = optax.adam(conf.learning_rate)

    # Build the freeze mask over the *filtered* (trainable-array) structure so it
    # aligns with the params the optimizer sees inside the training loop.
    params = eqx.filter(model, eqx.is_inexact_array)
    freeze_mask = jax.tree.map(lambda _: False, params)
    # Always exclude whatever model.observation.mstep touches from gradient
    # updates -- no conf.freeze_paths entry, no flag, required from the
    # caller. mstep is applied to model.observation unconditionally at its
    # configured cadence, so gradient descent must never
    # fight it, for any model (a no-op path list for Observations that
    # don't override mstep, e.g. Poisson). Same reasoning for noise.free
    # whenever model.q_mstep_active is true: Noise's Q M-step update would
    # otherwise immediately overwrite whatever gradient descent just
    # computed for noise.free -- not a numerical-stability issue (the LL
    # and KL terms are independent given a fixed posterior, so computing
    # each update from that same posterior is well-defined regardless),
    # just wasted, silently-discarded gradient computation unless excluded
    # here, mirroring observation.mstep's own auto-exclusion exactly.
    freeze_paths = [str(p) for p in conf.freeze_paths] + model.frozen_paths()
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

    # KL weight schedule: optax curve evaluated on the loop's step counter.
    # beta is an objective coefficient (not an optimizer hyperparameter), so it
    # is evaluated here and passed into the loss -- never routed through the
    # optimizer. kl_warmup_steps == 0 means standard ELBO (beta = 1).
    kl_warmup_steps = int(conf.kl_warmup_steps)
    # beta ramps 0 -> 1; beta = 1 is the standard ELBO (the annealing target,
    # fixed by the variational objective, not a tunable hyperparameter).
    beta_schedule = (
        optax.linear_schedule(
            init_value=0.0, end_value=1.0, transition_steps=kl_warmup_steps
        )
        if kl_warmup_steps > 0
        else optax.constant_schedule(1.0)
    )

    final_model = _run_training_loop(
        model,
        train_data,
        key,
        dataloader,
        conf.batch_size,
        conf.max_epoch,
        optimizer,
        data_sharding,
        model_sharding,
        on_epoch_end=on_epoch_end,
        param_schedule=param_schedule,
        beta_schedule=beta_schedule,
        regularizer=regularizer,
    )

    dt = time.perf_counter() - t0
    logger.info("train end: elapsed=%.2fs", dt)

    return final_model
