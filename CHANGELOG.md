# Changelog

## Unreleased

* KL warm-up is now driven by an `optax` schedule built from
  `conf.kl_warmup_steps`, evaluated on the training step and passed as the
  `beta` KL weight. `beta` stays an objective coefficient (never routed through
  the optimizer).
* **Breaking:** `batch_loss` is now a pure objective evaluator
  `batch_loss(model, batch, key, *, beta=1.0)`; the `step`, `kl_warmup_steps`,
  and `noise_regularizer` arguments were removed. Callers compute the KL weight
  and pass `beta` directly.
* **Breaking:** the model regularizer is no longer a config field
  (`conf.noise_regularizer` removed). Pass it directly as
  `train(model, data, conf=..., regularizer=...)`; the trainer composes
  `loss = -ELBO + regularizer(model)`. The config stays serializable.
* **Breaking:** the default optimizer no longer applies weight decay, and
  `weight_decay` was removed from the trainer config. In a plugin framework the
  trainer cannot know which leaves are weight matrices vs variances/biases, so
  it makes no decay assumption. `train` now accepts
  `optimizer=` (an `optax.GradientTransformation`); pass your own optimizer for
  (masked) weight decay or custom schedules. `freeze_paths` is still applied on
  top of either the default or a supplied optimizer.

## 0.8.0

* **Breaking:** `train` no longer splits data or handles validation. Its
  signature is now `train(model, train_data, *, conf, on_epoch_end=None)` and it
  returns the final-epoch model. Removed `valid_ratio`, `validation_size`,
  `min_epoch`, `min_iter`, `max_iter`, and `beta` from the trainer config.
* Added `EpochHandler`, a self-contained epoch-level handler owning validation,
  best-model tracking, periodic checkpointing, metrics, and optional early
  stopping. Construct it with `valid_data` and pass it as `on_epoch_end`; read
  `EpochHandler.best_model` afterwards.
* Exposed `train_test_split` for callers to build their own train/validation
  split.
* Vendored the training loop locally (previously from `gearax.trainer`).

## 0.7.0

* Renamed the public dynamics/integrator terminology to `Dynamics` / `Integrator` with short concrete classes (`Identity`, `OU`, `Functional`, `Euler`, `RK4`).
* Removed legacy compatibility aliases and import shims for `state_maps` / `steppers`.
* Removed `dyn_conf.system_type`; users now choose the appropriate dynamics and integrator directly.
* Updated the docs, examples, benchmarks, and tests to the new canonical names.

## 0.6.0

* Added `NOFILT` mode for encoder-defined latent states.
* Added PCA dynamics tooling and the ELBO-vs-MSE equivalence notes.
* Added benchmark and example support for the PCA / dynamics workflow.

## 0.5.0

* Unified the Gaussian approximation API around `MVN(dim, rank)`.
* Simplified low-rank / full-rank encoder sizing and related configuration.
* Updated docs, tests, and benchmarks to the rank-based MVN layout.

## 0.4.0

* Split latent dynamics into separate dynamics and integrator abstractions.
* Added declarative function-backed dynamics.
* Updated the examples and tests to use the new dynamics pipeline.

## 0.3.0

* Added `filter`, `smooth`, and `causal` inference modes.
* Added declarative parameter freezing via `freeze_paths` and `freeze_state_noise`.
* Stabilized the low-rank MVN / LoRa path with additional smoke coverage.

## 0.2.0

* Added KL warmup annealing via `kl_warmup_steps`.
* Refactored the dynamics / noise layout and moved dynamics into a dedicated subpackage.
* Overhauled the Van der Pol example and added flow-field evaluation tooling.

## 0.1.0

* Initial public release.
