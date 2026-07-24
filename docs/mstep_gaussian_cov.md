# EM M-step for Gaussian observation covariance (`mstep_gaussian_cov`)

Status: implemented, unit-tested, and validated against a real downstream
training run; not yet wired into a convenience training loop. This document
is a handoff for picking the work back up — problem, fix, validation
evidence, and open refinement questions.

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

This was not a theoretical concern — it was found directly in a real
downstream campaign (a plain shPLRNN dynamics model, `readout="linear"`, fit
with L-BFGS after an Adam burn-in, free-train regime, no noise freezing): 2 of
26 fits reached a wildly negative total loss (e.g. `-13334`, vs. the sane
`~340-350` range for sibling seeds on the same data), traced to 5-8
observation dimensions with `cov()` pinned at the float32-safety floor
(`jaxfads.constraints._MIN_VARIANCE = 1e-6`) while their *actual* residual
variance — measured independently as `mean[(y - reconstructed_mean)^2]` on
the same data — was `~0.85-1.3`, a **10^3-10^6× mismatch**. The fitted
parameter had completely decoupled from reconstruction quality; `_MIN_VARIANCE`
prevented a literal float32 underflow (no NaN/Inf), but not this
optimization-level exploit.

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
- **Real-data validation** (downstream project, not in this repo): loaded the
  two degenerate models from the campaign above and confirmed
  `mstep_gaussian_cov` recovers sane values (e.g. `0.87`-`9.67` for dimensions
  previously pinned at `1e-6`) matching the independently-measured residual.
  Then validated end-to-end in real training: re-ran the same two
  (day, seed) combinations with the EM-alternation pattern above and got sane
  final losses (`343.65`, `345.38`) matching the 18 originally-clean fits,
  with `cov()` stable and well above the floor across all EM rounds.
  Subsequently re-ran the **full 27-job campaign** (9 days × 3 seeds) with
  this fix in place: all 27 converged cleanly, no repeat of the degenerate
  loss, `cov_min` consistently in `0.08`-`0.18` across all jobs, tight
  cross-seed agreement per day.

## Open questions for refinement

1. **No native training-loop integration.** `mstep_gaussian_cov` is a
   standalone function; every caller currently hand-rolls the alternation
   loop (see the downstream project's
   `scripts/python/jaxfads_dev/jaxfads_dev_train_adam_lbfgs.py` for a full
   example, including adaptive round-length/stopping-criterion tuning that
   isn't part of this repo). Worth considering a `trainer.py`-level
   convenience wrapper (e.g. `em_train(model, data, conf, mstep_fn, n_rounds)`)
   if this pattern turns out to be broadly useful beyond Gaussian
   observation noise.
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
4. **Only `Gaussian` currently implements `mstep_stat`.** If a future
   likelihood (e.g. a full-covariance or heavy-tailed Gaussian variant) is
   added, it would need its own M-step derivation and `mstep_stat`
   implementation — the duck-typed dispatch in `mstep_gaussian_cov` already
   supports this without modification, but the function name itself
   (`mstep_gaussian_cov`) is Gaussian-specific; a more general name might be
   warranted if this expands (e.g. `mstep_observation_cov`).
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
