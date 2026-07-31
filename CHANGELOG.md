# Changelog

## Unreleased

* **Default behavior change:** `MVN` now defaults to
  `use_sigma_points=True` -- propagating `q(z_{t-1})` through the
  transition uses deterministic unscented-transform (UT) sigma points
  instead of Monte Carlo sampling by default. `Approx.transition_points`
  is a new, overridable hook (returns `(points, weights)` for this
  propagation step); `MVN(use_sigma_points=False)` restores the previous
  Monte Carlo behavior exactly, bit-for-bit. Real-data validation across
  `state_dim` 8/16/32 showed UT is never worse, and often better/cheaper,
  than Monte Carlo on posterior-RMSE. See `docs/transition_points.md`.
* Added accumulated epoch-local closed-form EM M-steps for Gaussian
  observation noise `R` and, when enabled, transition/process noise `Q`.
  Their additive statistics are emitted by each existing pre-SGD minibatch
  inference pass and finalized once at the epoch boundary, avoiding both
  noisy replacement-style minibatch updates and an extra full-data inference
  pass. R remains always M-step-owned for Gaussian likelihoods. Standalone
  `mstep_gaussian_cov`/`mstep_observation_cov` and `XFADS.mstep(...)` remain
  available for explicit full-data recomputation outside `train()`. See
  `docs/mstep_gaussian_cov.md` and `docs/mstep_dynamics_noise.md`.
* **Breaking:** transition/process noise uses required top-level positive
  `q_scale` and `q_mstep` (default `true`). `q_scale` initializes Q; with
  `q_mstep=true`, epoch-local MAP Q finalization uses
  `(q_scale, state_dim + 1)` and `train()` auto-freezes `noise_free`.
  `q_mstep=false` omits Q statistics/finalization and leaves Q SGD-managed.
  `dyn_conf.state_noise`, `noise_prior`, and `noise_prior_dof` were removed.
* Added `cuda12`/`cuda13` optional-dependency extras
  (`pip install jaxfads[cuda12]` or `jaxfads[cuda13]`) for one-step GPU
  installs; the base `jax` dependency stays CPU-only. The two extras are
  mutually exclusive (each pulls in a distinct jaxlib/CUDA build);
  `cuda13` also drops some older GPU architectures (Maxwell/Volta/Pascal)
  that `cuda12` still supports.
* Removed the superseded direct-replacement `mstep_mode` trainer cadence
  and its redundant full-dataset recomputation path.

## 0.10.0

* `Gaussian.eloglik` now computes the analytic expected log-likelihood via a
  Woodbury/matrix-determinant-lemma form for the diagonal-plus-low-rank
  observation covariance, instead of building a dense
  `(observation_dim x observation_dim)` matrix. This cuts the dominant cost
  from `O(observation_dim^2)` memory / `O(observation_dim^3)` time to
  `O(observation_dim * state_dim^2 + state_dim^3)`, fixing an out-of-memory
  failure with high-dimensional observations and a linear readout (e.g.
  `observation_dim=497`, `state_dim=21`, `batch_size=16`). Verified to match
  the previous dense implementation to ~1e-7 in log-density and ~1e-5 in
  gradients.
* Upgraded the supported `jax` range: the transitive `jax<0.7.0` cap
  (inherited from `gearax`) is gone, `jax` now resolves to the latest
  release (0.11.0) with no `[tool.uv] override-dependencies` needed, and
  `requires-python` is bumped to `>=3.12` (jax 0.11.0 dropped Python 3.11
  support). This required switching to `tfp-nightly[jax]` (stable
  `tensorflow-probability` still depends on a jax internal removed in
  0.7.0) and requesting explicit `Auto` mesh axes in `trainer.py`'s device
  mesh to keep `eqx.filter_shard` working under jax's newer
  "sharding-in-types" default (jax >=0.9.0).
* Documented that custom `param_schedule` functions should anneal
  constrained (e.g. variance) values and convert to free-form parameters
  only as the final step, since interpolating in free-form space directly
  would distort the intended path. The dedicated `noise_schedule` helper
  was removed.

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
  attribute through a step-indexed `optax` schedule. `freeze_paths` should
  list the scheduled attribute so the optimizer's own gradient-based update
  does not fight the schedule.
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
* Added declarative parameter freezing via `freeze_paths`.
* Stabilized the low-rank MVN / LoRa path with additional smoke coverage.

## 0.2.0

* Added KL warmup annealing via `kl_warmup_steps`.
* Refactored the dynamics / noise layout and moved dynamics into a dedicated subpackage.
* Overhauled the Van der Pol example and added flow-field evaluation tooling.

## 0.1.0

* Initial public release.
