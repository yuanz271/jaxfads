# Changelog

## Unreleased

## 0.9.0

* **Breaking:** the default optimizer is now **vanilla Adam**
  (`optax.adam(conf.learning_rate)`) with no gradient clipping, no gradient
  noise, and no weight decay. `clip_norm`, `noise_eta`, `noise_gamma`, and
  `weight_decay` were all removed from the trainer config. Rationale: in a
  plugin framework the trainer cannot know which leaves are weights vs
  variances/biases (so no weight decay), and gradient noise/clipping can
  destabilize sensitive objectives -- e.g. chaotic dynamical-systems
  reconstruction, where `optax.add_noise`'s large early-training stochasticity
  prevented the model from settling onto the true attractor. Clipping, weight
  decay, gradient noise, and custom schedules are now all opt-in via a
  user-supplied `optimizer=`; `freeze_paths` is still applied on top. Removing
  `add_noise` also avoids an `add_noise` + `donate="all"` buffer-aliasing
  crash in the jitted loop.
* `train` now accepts `param_schedule=` (a `Callable[[model, step], model]`)
  applied at the start of every training step, for driving an arbitrary model
  attribute through a step-indexed `optax` schedule. A new helper,
  `noise_schedule(approx, q_hi, q_lo, transition_steps)`, builds the common
  case: annealing the process-noise scale (Q) geometrically via
  `optax.exponential_decay`. `freeze_paths` should list the scheduled
  attribute so the optimizer's own gradient-based update does not fight the
  schedule.
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
* The training loop now supports **params-aware** optimizers (those that read
  the current parameters at update time): decoupled weight decay (`adamw`,
  `add_decayed_weights`), trust-ratio methods (`lamb`, `lars`), and
  learning-rate-free methods (`optax.contrib.prodigy`, `dadapt_adamw`). The loop
  carries the trainable-array partition as optimizer state (rebuilding the model
  only inside the loss) and initializes the optimizer on a copy of the params,
  so any `optax.GradientTransformation` passed via `optimizer=` works.
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
