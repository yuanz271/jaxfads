# EM M-step for Gaussian observation covariance (`mstep_gaussian_cov`)

Status: implemented, unit-tested, and validated against a real downstream
training run. Now also wired into a convenience training-loop path: see
`mstep_every_n_epochs` in [Training](training.md#automated-observation-noise-updates-mstep_every_n_epochs)
and the family-neutral `mstep_observation_cov` function (open question #1
below, resolved). This document remains a handoff for the underlying
Gaussian-specific mechanics — problem, fix, validation evidence, and
remaining refinement questions.

See also: [Training](training.md), [Algorithm](algorithm.md),
[r_running_stat](r_running_stat.md) (the `mstep_every_n_epochs`/
`mstep_observation_cov` design).

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

Implementation (`src/jaxfads/observations.py`):

- `Gaussian.mstep_stat(t, moment, y, approx, readout)` — the per-(batch,time)
  sufficient statistic above, mirroring `eloglik`'s call signature/conventions
  (dense `(state_dim, state_dim)` covariance from `approx.unpack`, `readout`
  exposing `.weight`).
- `mstep_gaussian_cov(model, data, *, key, batch_size=None)` — runs the model
  forward (the E-step) over a dataset, aggregates `mstep_stat`, and returns a
  new model with `observation.likelihood.unconstrained_cov` replaced by the
  M-step-optimal value. Dispatch is **duck-typed** on `hasattr(likelihood,
  "mstep_stat")`, not `isinstance(likelihood, Gaussian)` — a future
  Gaussian-like likelihood can participate without subclassing `Gaussian`.
  Raises `NotImplementedError` for likelihoods without `mstep_stat` (e.g.
  `Poisson`, which has no free variance parameter to estimate this way).

## Usage pattern (classic EM alternation)

`mstep_gaussian_cov` is a standalone utility, not wired into `train()`'s loop.
The caller freezes `unconstrained_cov` from gradient updates and alternates:

```python
from jaxfads.trainer import train
from jaxfads.observations import mstep_gaussian_cov

conf.freeze_paths = ["observation.likelihood.unconstrained_cov"]
for _ in range(n_rounds):
    model = train(model, data, conf=conf)          # gradient-based round (Adam, L-BFGS, ...)
    model = mstep_gaussian_cov(model, data, key=key)  # closed-form R update
```

This mirrors the standard batch-EM cadence for state-space models (Shumway &
Stoffer 1982): E-step and M-step alternate over the *full dataset* per
iteration, not per-minibatch — matching `mstep_gaussian_cov`'s own full-dataset
(or chunked-by-`batch_size`) aggregation.

## Validation performed

- **Unit tests** (`tests/test_observations.py`, 3 tests):
  - `test_mstep_gaussian_cov_matches_independently_computed_residual_stat` —
    verifies the returned value matches an independently-computed residual
    statistic exactly (not just plausibly), starting from a deliberately
    Heywood-degenerate initial `unconstrained_cov`.
  - `test_mstep_gaussian_cov_raises_for_poisson_likelihood` — confirms the
    `NotImplementedError` path.
  - `test_mstep_gaussian_cov_dispatches_on_duck_typed_mstep_stat` — an
    independent (non-`Gaussian`-subclassing) likelihood implementing its own
    `mstep_stat` is recognized and used correctly, verifying the duck-typed
    dispatch actually works, not just that the `isinstance` check was relaxed.
  - 108/108 tests pass overall (105 pre-existing + 3 new).
- **Real-data validation** (downstream project, not in this repo, and not
  reproducible from this repo alone): `mstep_gaussian_cov` was confirmed to
  recover sane covariance values (matching independently-measured residuals)
  on the previously-degenerate models, and a full re-run of the affected
  campaign with the EM-alternation pattern above converged cleanly with no
  recurrence of the degenerate loss. Specific figures (loss values, job/seed
  counts, etc.) are project-specific data and are intentionally not
  reproduced here — see the unit tests above for evidence reproducible
  within this repo.

## Open questions for refinement

1. **Resolved: native training-loop integration.** `train(..., 
   mstep_every_n_epochs=N)` now calls a family-neutral driver,
   `mstep_observation_cov`, automatically every `N` completed epochs (see
   [Training](training.md#automated-observation-noise-updates-mstep_every_n_epochs)
   and `docs/r_running_stat.md` for the full design). `mstep_gaussian_cov`
   itself is untouched and remains the answer for `batch_size`-chunked
   scanning of datasets too large for one forward pass, which the automated
   path does not support. Round-cadence tuning (see #2 below) is still the
   caller's responsibility via the `N` in `mstep_every_n_epochs`, not
   auto-tuned.
2. **Round cadence is caller policy, empirically tuned downstream, not
   documented here.** The downstream validation found that round length is a
   genuine, non-trivial tradeoff (too long wastes wall-clock on a stale `R`
   estimate; too short pays disproportionate per-round dispatch/compile
   overhead), and that a round-to-round relative-improvement stopping
   criterion is *not* scale-invariant to round length (needs a fixed-epoch
   comparison window instead). None of this tuning logic lives in `jaxfads`
   itself — if a native wrapper is added (see #1), it should account for
   this rather than re-introduce the same bug.
3. **`batch_size` chunking in `mstep_gaussian_cov` is simple sequential
   accumulation**, not parallelized beyond a single forward pass per chunk.
   Fine at current scale (datasets of ~50-500 trials); revisit if used on
   much larger datasets.
4. **Resolved: family-neutral naming.** `mstep_gaussian_cov` keeps its
   Gaussian-specific name (unchanged, still duck-typed on `mstep_stat`), but
   the new `mstep_observation_cov` function and `Observation.mstep`/
   `Likelihood.mstep` ABC methods (see `docs/r_running_stat.md`) are the
   family-neutral entry points going forward. If a future likelihood (e.g. a
   full-covariance or heavy-tailed Gaussian variant) adds its own `mstep`, it
   participates in `mstep_observation_cov`/`mstep_every_n_epochs`
   automatically, without needing `mstep_gaussian_cov`-style duck-typed
   dispatch of its own.
5. **`_MIN_VARIANCE` (the private float32-safety floor in
   `constraints.py`) remains necessary independent of this fix** — it guards
   against literal numerical failure (log/reciprocal of an exact float32
   `0.0`), which is a different, narrower concern than the Heywood-case
   optimization-level exploit this M-step addresses. Both are needed; neither
   supersedes the other.

## Related commits (this branch)

- `64d2927` — `_MIN_VARIANCE` float32-safety floor (prerequisite; see #5 above).
- `f588af5` — `mstep_gaussian_cov` + `Gaussian.mstep_stat`, initial implementation and tests.
- `f8298e9` — switched dispatch from `isinstance(Gaussian)` to duck-typed `hasattr(mstep_stat)`.
