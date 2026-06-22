# Changelog

## Unreleased

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
