# EM M-step for Gaussian observation covariance (`mstep_gaussian_cov` / `mstep_every_n_epochs`)

Status: implemented and unit-tested. Two entry points share the same
underlying closed-form EM M-step: a standalone, chunking-capable function
(`mstep_gaussian_cov`) for manual EM alternation or large datasets, and a
trainer-integrated, automated path (`train(..., mstep_every_n_epochs=N)`)
for the common case. The manual path is validated against real downstream
training (see below); the automated path's real-data validation is still
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

Implementation (`src/jaxfads/base.py`, `src/jaxfads/observations.py`):

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
  Unchanged since first shipped.
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
  ABC-dispatched counterpart to `mstep_gaussian_cov`: a single full-dataset
  forward pass, no `batch_size` chunking, works for any `Observation`
  overriding `mstep` (a no-op otherwise, e.g. for `Poisson`).

`mstep`/`mstep_stat` must never carry gradients: both drivers run their
forward pass outside `eqx.filter_value_and_grad`, an architectural (not
conventional) isolation from the optimizer's gradient tape; `Gaussian.mstep`
additionally wraps its result in `jax.lax.stop_gradient` as cheap, local
insurance documenting the invariant.

## Two usage patterns

### 1. Automated (default recommendation): `train(..., mstep_every_n_epochs=N)`

```python
trained = train(model, data, conf=trainer_conf, mstep_every_n_epochs=1)
```

`train()` calls `mstep_observation_cov(model, train_data, key=...)`
automatically every `N` completed epochs, and automatically folds
`model.observation.mstep_frozen_paths()` into its internal gradient-freeze
mask — no `conf.freeze_paths` entry required from the caller.
`unconstrained_cov` cannot simply be marked `eqx.field(static=True)`
instead: static fields are compile-time-constant, hashable *auxiliary*
data in JAX/equinox's pytree system, not array leaves subject to numerical
replacement, and `mstep`'s own `eqx.tree_at`-based write-back needs a
genuine dynamic leaf to update — marking it static would very likely break
`mstep`'s own update mechanism, not just block gradients. The field must
also stay ordinary and trainable because `mstep_every_n_epochs` is opt-in:
a user who doesn't set it still gets the original, fully-gradient-trainable
`R`, so exclusion has to be a per-training-run decision, not a structural
one.

`mstep_every_n_epochs` is a dedicated `train()` parameter — the same
category as `regularizer`/`optimizer`/`param_schedule` (a modification to
*how the model gets updated during optimization*), not epoch-level policy
— rather than being routed through `on_epoch_end`. `on_epoch_end`/hook-style
slots are single-callable; routing this through `on_epoch_end` would force
a user who already uses it for checkpointing/early-stopping to hand-compose
it with an `mstep` closure. A user-supplied `on_epoch_end` keeps working
unmodified and independently, whether or not `mstep_every_n_epochs` is set.

No continuous per-minibatch update (EMA-style) was built: a *self-contained*
per-minibatch pass (a separate forward pass on every minibatch, not reusing
the training step's own forward pass) costs the same total compute over an
epoch as one big periodic pass — it doesn't save anything, it just
distributes the same cost differently. The only version that would
actually save compute requires threading the statistic back via
`eqx.filter_value_and_grad(..., has_aux=True)`, reusing the training step's
own forward pass — a bigger architectural commitment, not built
speculatively (see Open questions).

**Limitation**: no `batch_size`-chunked scanning — use pattern 2 below for
datasets too large for a single forward pass. See
[Training](training.md#automated-observation-noise-updates-mstep_every_n_epochs)
for full usage details.

### 2. Manual EM alternation (for chunked/large-dataset needs): `mstep_gaussian_cov`

```python
from jaxfads.trainer import train
from jaxfads.observations import mstep_gaussian_cov

conf.freeze_paths = ["observation.likelihood.unconstrained_cov"]
for _ in range(n_rounds):
    model = train(model, data, conf=conf)             # gradient-based round (Adam, L-BFGS, ...)
    model = mstep_gaussian_cov(model, data, key=key)   # closed-form R update
```

Same math as pattern 1 (`mstep_stat`), applied with `batch_size`-chunked
scanning and manual round control. This mirrors the standard batch-EM
cadence for state-space models (Shumway & Stoffer 1982): E-step and M-step
alternate over the *full dataset* per iteration, not per-minibatch. Here,
unlike pattern 1, `conf.freeze_paths` must be set manually — this driver
has no `train()`-level integration to derive it automatically.

## Validation performed

- **Unit tests** (`tests/test_observations.py`):
  - `test_mstep_gaussian_cov_matches_independently_computed_residual_stat`,
    `test_mstep_gaussian_cov_raises_for_poisson_likelihood`,
    `test_mstep_gaussian_cov_dispatches_on_duck_typed_mstep_stat` — original
    `mstep_gaussian_cov` validation (exact residual-statistic match from a
    deliberately Heywood-degenerate start, `NotImplementedError` for
    `Poisson`, duck-typed dispatch for a non-`Gaussian`-subclassing
    likelihood).
  - `test_glm_mstep_matches_mstep_gaussian_cov`,
    `test_glm_mstep_is_noop_for_poisson`, `test_glm_mstep_frozen_paths`,
    `test_mstep_observation_cov_matches_mstep_gaussian_cov`,
    `test_mstep_observation_cov_is_noop_for_poisson` — `Observation.mstep`/
    `mstep_frozen_paths` and `mstep_observation_cov` produce identical
    results to `mstep_gaussian_cov` and correctly no-op for `Poisson`.
- **Unit/integration tests** (`tests/test_trainer.py`):
  - `test_mstep_every_n_epochs_updates_r_and_freezes_it_from_gradients`,
    `test_mstep_every_n_epochs_none_leaves_existing_behavior_unaffected`,
    `test_mstep_every_n_epochs_composes_with_on_epoch_end`,
    `test_mstep_every_n_epochs_composes_with_user_freeze_paths`,
    `test_mstep_every_n_epochs_cadence` — automatic gradient-exclusion
    (no `conf.freeze_paths` entry needed), zero-footprint default,
    `on_epoch_end`/user-`freeze_paths` composition, and cadence (`N`-epoch
    firing) all verified.
  - 118/118 tests pass overall (108 pre-existing + 10 new).
- **Real-data validation** (`mstep_gaussian_cov` only; downstream project,
  not in this repo, not reproducible from this repo alone): confirmed to
  recover sane covariance values (matching independently-measured residuals)
  on previously-degenerate models, and a full re-run of the affected
  campaign with the EM-alternation pattern above converged cleanly with no
  recurrence of the degenerate loss. Specific figures are project-specific
  data and intentionally not reproduced here.
- **Real-data validation for `mstep_every_n_epochs`**: not yet performed —
  see Open questions.

## Open questions for refinement

1. **`batch_size` chunking in `mstep_gaussian_cov` is simple sequential
   accumulation**, not parallelized beyond a single forward pass per chunk.
   Fine at current scale (datasets of ~50-500 trials); revisit if used on
   much larger datasets. `mstep_every_n_epochs`'s automated path has no
   chunking at all (see pattern 1's limitation above); revisit only if a
   real need for chunked *and* automated scanning emerges.
2. **`_MIN_VARIANCE` (the private float32-safety floor in `constraints.py`)
   remains necessary independent of this fix** — it guards against literal
   numerical failure (log/reciprocal of an exact float32 `0.0`), a
   different, narrower concern than the Heywood-case optimization-level
   exploit this M-step addresses. Both are needed; neither supersedes the
   other.
3. **Two evidence-gated escalations for `mstep_every_n_epochs`, not built
   speculatively:**
   - *Stabilize with an EMA blend*: if a real-data run shows a visible
     training-loss transient right after a periodic `mstep` recompute
     (a smoothness concern), blend the whole `Observation` old-vs-new via a
     generic pytree blend (untouched leaves are unaffected, since `mstep`'s
     output equals the input everywhere but `unconstrained_cov`) instead of
     the current hard `eqx.tree_at` replace.
   - *Reuse the training step's own forward pass* (a compute concern,
     independent of smoothness): only worth it if profiling shows the
     periodic extra forward pass materially matters *and* eliminating it
     (not just amortizing it via a larger `N`) is the actual goal — requires
     `has_aux` threading through `batch_loss`/`vi.elbo`/`eloglik`, a real
     code-structure cost (though it does not change the optimization
     dynamics: same loss, same gradients, aux is a side-channel).
4. **Whether `Observation.mstep`/`mstep_frozen_paths` generalize to
   `Approx`/`Dynamics`** — a candidate component-uniform contract, the same
   generalization `eloglik`/`predictive_moment`/`initialize` already have.
   See `mstep_dynamics_noise.md`'s `mstep_transition_stat`/
   `mstep_noise_shrink` for the process-noise (`Q`) analogue — but `Q`'s own
   continuous per-step cadence should **not** be adopted by default given
   its runaway-collapse risk; only `R` has been argued safe for continuous
   cadence.
5. **Real-data validation of `mstep_every_n_epochs`**: re-run (or
   spot-check) the same downstream campaign used to validate
   `mstep_gaussian_cov`, using `mstep_every_n_epochs` instead of the manual
   alternation loop; compare final losses / `cov_min` against the
   already-validated results.
6. **Whether to add `Observation.mstep_frozen_paths()`-driven auto-freezing
   for the *manual* `mstep_gaussian_cov` pattern too** (currently only the
   automated path derives its freeze mask automatically; pattern 2 still
   requires `conf.freeze_paths` to be set by hand) — a convenience
   improvement, not a correctness gap (the manual pattern's requirement is
   already correctly documented above).

## Related commits (this branch)

- `64d2927` — `_MIN_VARIANCE` float32-safety floor (prerequisite; see #2 above).
- `f588af5` — `mstep_gaussian_cov` + `Gaussian.mstep_stat`, initial implementation and tests.
- `f8298e9` — switched dispatch from `isinstance(Gaussian)` to duck-typed `hasattr(mstep_stat)`.
- `28520be` — `Observation.mstep`/`mstep_frozen_paths`, `Likelihood.mstep`/`GLM.mstep` dispatch,
  `mstep_observation_cov`, and `train(..., mstep_every_n_epochs=...)`.
- `0165677` — stripped downstream-project-specific data from this document.
