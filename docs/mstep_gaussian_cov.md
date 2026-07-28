# EM M-step for Gaussian observation covariance (`mstep_gaussian_cov` / `Observation.mstep`)

Status: implemented and unit-tested. `R` (the Gaussian observation noise
covariance) is estimated via a closed-form EM M-step in two forms sharing
the same underlying math: an **update built directly into `train()`**
(always on for any `Observation` implementing `mstep` — never opt-in — with
a configurable cadence via `mstep_mode="minibatch" | "epoch"`, plus a
guaranteed final full-dataset update after the last epoch regardless of
mode), and **standalone functions** (`mstep_gaussian_cov`,
`mstep_observation_cov`) for manual, full-dataset, or chunked use outside of
`train()`. The manual path is validated against real downstream training
(see below); the `train()`-integrated path's real-data validation is still
pending (see Open questions).

See also: [Training](training.md), [Algorithm](algorithm.md).

## The problem

`Gaussian.eloglik` computes the joint likelihood

```
log N(y; E[Cz], C Cov(z) C^T + R)
```

where `C Cov(z) C^T` is low-rank (rank ≤ state_dim) and `R` (the per-dimension
observation noise, `Gaussian.unconstrained_cov` under `constrain_positive`) is
diagonal. This is structurally identical to a **factor-analysis model**, with
`R` playing the role of the factor model's "uniquenesses" — and it inherits
factor analysis's best-known MLE pathology, the **Heywood case**: joint
gradient-based optimization of `R` can drive one or more of its components
toward the numerical floor while the corresponding dimension's residual stays
large, because the low-rank correction term lets the *joint* density favor
this even though it does not reflect a genuine improvement in fit for that
dimension.

This is not merely a theoretical concern: it has been observed directly in
practice, with a fitted covariance reaching the float32-safety floor
(`jaxfads.constraints._MIN_VARIANCE = 1e-6`) while the corresponding
dimension's *actual* residual variance — measured independently as
`mean[(y - reconstructed_mean)^2]` on the same data — was orders of
magnitude larger. The fitted parameter had completely decoupled from
reconstruction quality; `_MIN_VARIANCE` prevents a literal float32 underflow
(no NaN/Inf), but not this optimization-level exploit. (Specific figures
from the downstream campaign that surfaced this are project-specific data,
not general library documentation, and are intentionally not reproduced
here.)

## The fix

Estimate `R` via the closed-form EM M-step instead of joint gradient descent:

```
R_d = mean over (batch, time) of [ (y_d - E[Cz]_d)^2 + (C Cov(z) C^T)_dd ]
```

i.e. the expected squared residual under the current posterior, including the
propagated posterior uncertainty term. This is immune to the Heywood-case
exploit **by construction**: the estimate *is* the expected squared residual,
so it cannot decouple from actual reconstruction quality the way a freely
gradient-optimized parameter can.

Implementation (`src/jaxfads/base.py`, `src/jaxfads/observations.py`,
`src/jaxfads/trainer.py`):

- `Gaussian.mstep_stat(t, moment, y, approx, readout)` — the per-(batch,time)
  sufficient statistic above, mirroring `eloglik`'s call signature/conventions
  (dense `(state_dim, state_dim)` covariance from `approx.unpack`, `readout`
  exposing `.weight`). Unchanged since first shipped.
- `mstep_gaussian_cov(model, data, *, key, batch_size=None)` — runs the model
  forward (the E-step) over a dataset, aggregates `mstep_stat`, and returns a
  new model with `observation.likelihood.unconstrained_cov` replaced by the
  M-step-optimal value. Dispatch is **duck-typed** on `hasattr(likelihood,
  "mstep_stat")`, not `isinstance(likelihood, Gaussian)` — a future
  Gaussian-like likelihood can participate without subclassing `Gaussian`.
  Raises `NotImplementedError` for likelihoods without `mstep_stat` (e.g.
  `Poisson`, which has no free variance parameter to estimate this way).
  Unchanged since first shipped. Supports `batch_size`-chunked scanning for
  datasets too large for a single forward pass.
- `Observation.mstep(t, moment, y, approx)` / `Observation.mstep_frozen_paths()`
  — concrete, no-op-default ABC methods (`base.py`): the framework-neutral
  entry point any `Observation` implementation may opt into.
- `Likelihood.mstep(t, moment, y, approx, readout)` /
  `Likelihood.mstep_frozen_paths()` — documented on the `Likelihood`
  `Protocol`'s shape, optional (no runtime default — `Protocol` provides
  none). `GLM.mstep`/`GLM.mstep_frozen_paths` dispatch to `self.likelihood`
  via an internal `hasattr` check, mirroring `GLM.eloglik`'s delegation;
  `Poisson` needs no changes (nothing to inherit, nothing to implement).
  `Likelihood` intentionally stays a `Protocol`, not a
  `SubclassRegistryMixin`-registered plugin type: `Observation` is the
  framework's actual plugin surface, and `Likelihood`/`readout` are `GLM`'s
  own private internal composition — a different `Observation` subclass
  wouldn't need that split at all.
- `Gaussian.mstep` / `Gaussian.mstep_frozen_paths` — wraps `mstep_stat` into
  a single-forward-pass update (`jnp.mean` over the given data, versus
  `mstep_gaussian_cov`'s chunked accumulation) and reports
  `["unconstrained_cov"]` as the path needing exclusion from gradients.
- `mstep_observation_cov(model, data, *, key)` — the family-neutral,
  ABC-dispatched, standalone counterpart to `mstep_gaussian_cov`: a single
  full-dataset forward pass, no `batch_size` chunking, works for any
  `Observation` overriding `mstep` (a no-op otherwise, e.g. for `Poisson`).
  Not called by `train()` itself (see below) — a manual-use utility only.
- `train()` / `_run_training_loop()`'s `mstep_mode: {"minibatch", "epoch"} =
  "minibatch"` — cadence for the always-on update, talking only through the
  `Observation` ABC's `mstep` interface directly (not by calling
  `mstep_observation_cov` — `trainer.py` never imports anything from
  `observations.py`, the concrete-implementations module, keeping the
  trainer strictly `Observation`-agnostic):
  - `"minibatch"`: every `train_step` replaces `model.observation` with
    `model.observation.mstep(t, moment, y, model.approx)` computed from
    that minibatch's own forward pass.
  - `"epoch"`: instead, a shared `apply_mstep` closure runs once per
    completed epoch (inside `finalize_epoch`, after any user
    `on_epoch_end` callback), from a forward pass over the whole
    `train_set`.
  - Regardless of mode, the same `apply_mstep` closure is called once
    more, unconditionally, immediately after the entire
    `try`/`except`/`finally` training loop — covering normal completion,
    early stop via `on_epoch_end`, and `KeyboardInterrupt` alike — so the
    returned model always reflects a fresh, full-`train_set` mstep
    correction regardless of mode or how training ended. Invalid
    `mstep_mode` values raise `ValueError`.
  - `train()` always folds `model.observation.mstep_frozen_paths()` into
    its internal gradient-freeze mask regardless of `mstep_mode`, so
    gradient descent never fights this update — no `conf.freeze_paths`
    entry, and no flag, are needed.

`mstep`/`mstep_stat` must never carry gradients: all of the above run their
forward pass outside `eqx.filter_value_and_grad`, an architectural (not
conventional) isolation from the optimizer's gradient tape; `Gaussian.mstep`
additionally wraps its result in `jax.lax.stop_gradient` as cheap, local
insurance documenting the invariant.

## Usage

### Default: automatic, always-on, configurable cadence

Nothing to configure to get the closed-form update itself. Any `train()`
call on a Gaussian-likelihood model always estimates `R` this way; there is
no way to opt out and fall back to gradient-based `R` (a deliberate
simplicity choice). `mstep_mode` only controls *when* the update fires:

```python
trained = train(model, data, conf=trainer_conf)  # mstep_mode="minibatch" (default)
trained = train(model, data, conf=trainer_conf, mstep_mode="epoch")
```

`"minibatch"`: each minibatch's estimate is a noisy sample of the same
quantity a full-dataset pass computes exactly (like SGD vs. full-batch
gradient descent). `"epoch"`: an exact, full-dataset estimate, refreshed
once per epoch instead of continuously. Either way, `train()` always
applies one more full-dataset update immediately after the final epoch's
gradient steps, so the returned model is never left with only a
minibatch-noisy estimate.

### Manual, full-dataset, or chunked: `mstep_gaussian_cov` / `mstep_observation_cov`

For an exact, full-dataset recompute — e.g. a final correction, validation,
or classic EM-style alternation — call one of the standalone functions
directly. Neither is used by `train()` itself.

```python
from jaxfads.trainer import train
from jaxfads.observations import mstep_gaussian_cov

conf.freeze_paths = ["observation.likelihood.unconstrained_cov"]
for _ in range(n_rounds):
    model = train(model, data, conf=conf)             # gradient-based round (Adam, L-BFGS, ...)
    model = mstep_gaussian_cov(model, data, key=key)   # closed-form R update
```

`mstep_gaussian_cov` (Gaussian-specific, `batch_size`-chunked scanning) and
`mstep_observation_cov` (family-neutral, no chunking) compute the same
math, just orchestrated differently. Note `conf.freeze_paths` must be set
manually for *this* pattern — only `train()`'s own automatic mechanism
(either `mstep_mode`) derives its freeze mask automatically.

## Validation performed

- **Unit tests** (`tests/test_observations.py`):
  - `test_mstep_gaussian_cov_matches_independently_computed_residual_stat`
    — original `mstep_gaussian_cov` validation: exact residual-statistic
    match from a deliberately Heywood-degenerate start.
  - `test_glm_mstep_matches_mstep_gaussian_cov`,
    `test_glm_mstep_is_noop_for_poisson`, `test_glm_mstep_frozen_paths` —
    `Observation.mstep`/`mstep_frozen_paths` produce identical results to
    `mstep_gaussian_cov` and correctly no-op for `Poisson`.
  - (Purged: `test_mstep_gaussian_cov_raises_for_poisson_likelihood`,
    `test_mstep_gaussian_cov_dispatches_on_duck_typed_mstep_stat`,
    `test_mstep_observation_cov_matches_mstep_gaussian_cov`,
    `test_mstep_observation_cov_is_noop_for_poisson` — negative-path,
    architecture-guarantee, and redundant-wrapper coverage respectively;
    deprioritized per correctness/functionality-first testing philosophy.)
- **Unit/integration tests** (`tests/test_trainer.py`):
  - `test_mstep_updates_r_unconditionally`,
    `test_mstep_frozen_paths_always_excluded_from_gradients`,
    `test_mstep_composes_with_on_epoch_end`,
    `test_mstep_composes_with_user_freeze_paths` — the always-on update
    (default `mstep_mode="minibatch"`) moves `R` away from a
    deliberately-wrong init with no configuration, gradient descent never
    fights it (verified against an independent `mstep` call, within the
    model's inherent Monte Carlo sampling tolerance), and `on_epoch_end`/a
    user's own unrelated `conf.freeze_paths` entries keep working
    unmodified alongside it.
  - `test_mstep_mode_epoch_updates_only_at_epoch_boundaries` — with
    `mstep_mode="epoch"`, `R` stays exactly at its initial value through an
    entire epoch's minibatch training (confirmed via `on_epoch_end`, which
    fires before that epoch's own end-of-epoch `mstep` call), but has
    updated by the end of training.
  - `test_mstep_applied_after_final_epoch_regardless_of_mode`
    (parametrized over both `mstep_mode` values) — the guaranteed final
    update after the last epoch holds for both modes, verified as a fixed
    point against an independent `mstep` call.
  - Full suite: 115/115 tests pass.
- **Real-data validation** (`mstep_gaussian_cov` only; downstream project,
  not in this repo, not reproducible from this repo alone): confirmed to
  recover sane covariance values (matching independently-measured residuals)
  on previously-degenerate models, and a full re-run of the affected
  campaign with the EM-alternation pattern above converged cleanly with no
  recurrence of the degenerate loss. Specific figures are project-specific
  data and intentionally not reproduced here.
- **Real-data validation of the `train()`-integrated update** (either
  `mstep_mode`): not yet performed — see Open questions.

## Open questions for refinement

1. **`batch_size` chunking in `mstep_gaussian_cov` is simple sequential
   accumulation**, not parallelized beyond a single forward pass per chunk.
   Fine at current scale (datasets of ~50-500 trials); revisit if used on
   much larger datasets.
2. **`_MIN_VARIANCE` (the private float32-safety floor in `constraints.py`)
   remains necessary independent of this fix** — it guards against literal
   numerical failure (log/reciprocal of an exact float32 `0.0`), a
   different, narrower concern than the Heywood-case optimization-level
   exploit this M-step addresses. Both are needed; neither supersedes the
   other.
3. **Real-data validation of the `train()`-integrated update** (either
   `mstep_mode`): re-run (or spot-check) the same downstream campaign used
   to validate `mstep_gaussian_cov`, using plain `train()` (no manual
   alternation); compare final losses / `cov_min` against the
   already-validated results.
4. ~~Whether an occasional, exact full-dataset recompute on top of the
   per-minibatch update would be worth adding back~~ — **done**: `mstep_mode
   ="epoch"` gives an exact per-epoch recompute, and the guaranteed final
   full-dataset update after the last epoch applies regardless of mode.
   `mstep_gaussian_cov`/`mstep_observation_cov` remain available for a
   caller who wants a fully manual, `train()`-external recompute instead.
5. **Whether `Observation.mstep`/`mstep_frozen_paths` generalize to
   `Approx`/`Dynamics`** — a candidate component-uniform contract, the same
   generalization `eloglik`/`predictive_moment`/`initialize` already have.
   See `mstep_dynamics_noise.md`'s `mstep_transition_stat`/
   `mstep_noise_shrink` for the process-noise (`Q`) analogue — but `Q`'s own
   continuous per-minibatch cadence should **not** be adopted
   unconditionally the way `R`'s is here, given its runaway-collapse risk;
   only `R` has been argued safe for continuous, unconditional cadence.
6. **Whether to add `Observation.mstep_frozen_paths()`-driven auto-freezing
   for the *manual* `mstep_gaussian_cov`/`mstep_observation_cov` functions
   too** (currently only `train()`'s own automatic mechanism, either
   `mstep_mode`, derives its freeze mask automatically; the manual
   functions still require `conf.freeze_paths` to be set by hand around
   them) — a convenience improvement, not a correctness gap.

## Related commits (this branch)

- `64d2927` — `_MIN_VARIANCE` float32-safety floor (prerequisite; see #2 above).
- `f588af5` — `mstep_gaussian_cov` + `Gaussian.mstep_stat`, initial implementation and tests.
- `f8298e9` — switched dispatch from `isinstance(Gaussian)` to duck-typed `hasattr(mstep_stat)`.
- `28520be` — `Observation.mstep`/`mstep_frozen_paths`, `Likelihood.mstep`/`GLM.mstep` dispatch,
  `mstep_observation_cov`.
- `0165677` — stripped downstream-project-specific data from this document.
- `91b6161` — reworked the trainer integration from an opt-in
  `mstep_every_n_epochs` parameter to an unconditional, per-minibatch
  update built directly into `train_step`; removed the intermediate
  parameter entirely.
- `4123fc8` — added `mstep_mode: {"minibatch", "epoch"} = "minibatch"` and
  the guaranteed final full-dataset update after the last epoch regardless
  of mode; purged 5 low-value tests, added 2 new `mstep_mode` tests.
- `57786a0` — unrelated to this feature: `AGENTS.md` testing-workflow policy update.
