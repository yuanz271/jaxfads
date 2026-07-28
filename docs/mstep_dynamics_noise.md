# Plan: MAP-shrunk M-step for transition/process noise (`mstep_dynamics_noise`)

Status: proposed, not implemented; design substantially refined through
extended discussion (see below), but **still higher risk than the R
estimator** — read the runaway-loop analysis below before implementing, and
build the synthetic validation harness (Step 1) before writing any
production code. Several claims below (self-pacing subsuming annealing,
safe cadence) are stated as **hypotheses to test in that harness**, not
established conclusions — flagged explicitly where relevant.

See also: [mstep_gaussian_cov](mstep_gaussian_cov.md), [transition_points](transition_points.md).

## Problem

`model.noise_free` (process noise `Q`) is always gradient-trained jointly
with the rest of the model (dynamics, encoder, readout), via
`approx.canon_to_moment(approx.free_to_canon(model.noise_free))` in
`core.py`. This is subject to two distinct pathologies:

1. **Rank-deficiency exploit** (same family as the observation-`R` Heywood
   case): when `mc_size < state_dim`, the MC "spread" term in the predictive
   covariance is rank-deficient, giving gradient descent a route to shrink
   `Q` in the directions that term happens to cover. **Now mitigated by
   default**: `transition_points.md` shipped, and `MVN` now defaults to
   `use_sigma_points=True` (deterministic sigma points, whose spread term's
   rank tracks the model's genuine local sensitivity rather than an
   artificial `mc_size`-driven cap). See that doc's own residual open
   questions (the zero-weight-center-point masking gap) for what isn't
   fully closed.
2. **Premature/excessive shrinkage relative to dynamics competence** (a
   pacing problem, not a destination problem — reframed after discussion;
   see below). Unlike `R`, which is checked against *fixed, exogenous* data
   `y`, the KL term comparing `q(z_t)` to the `Q`-derived predictive prior
   has **both sides jointly trainable**. Gradient descent can shrink
   `q(z_t)`'s posterior width and `Q` together, driving KL toward zero
   without ever consulting real data — especially in low-SNR regimes, where
   the per-step observation evidence pulling the posterior mean is weak.
   `Q = 0` (deterministic dynamics) is a **self-consistent fixed point** of
   this process.

   **Important clarification: driving `Q` toward small values is not
   itself the failure mode — it is often the goal.** A small `Q` forces the
   dynamics network `f` to explain trajectory variance directly rather than
   attributing it to noise, which is exactly the inductive bias wanted for
   learning genuinely deterministic-ish systems (e.g. Lorenz). The actual
   risk is `Q` becoming small *before* `f` is competent enough to explain
   that variance, which then damages the encoder/dynamics/readout jointly
   (an empirically observed failure mode, not just a theoretical one: a
   pre-trained-small `Q` harms the other components even though a
   *converged*-small `Q` is fine or desirable). So the design goal is
   **pacing** `Q`'s descent to track `f`'s genuine improvement, with a
   floor against going *too* small even asymptotically — not preventing
   descent altogether.

### Why a naive residual-based estimator does not automatically fix (2)

A residual-based M-step statistic for `Q` (mirroring `Gaussian.mstep_stat`)
would be computed from the model's **own internal states** (`z_t`,
`z_{t-1}` smoothed moments) rather than exogenous data. This closes a loop:

```
smaller Q -> tighter KL penalty on q(z_t) deviating from f(z̄_{t-1})
          -> training pulls both mean AND variance of q(z_t) toward f(z̄_{t-1})
          -> smoothed residual and Cov_smoothed(z_t) both shrink
          -> Q_stat (computed from those same shrunk quantities) shrinks
          -> feed smaller Q back in -> repeat
```

`Q = 0` is a fixed point of this loop by construction. Whether it is
*attracting* depends on the ratio of "informativeness of `y`" to "precision
imposed by `Q`" at each step — small in low-SNR regimes, i.e. exactly the
regime this feature is meant to help with. This is a documented phenomenon
in the classical linear-Gaussian SSM/EM literature: EM-based `Q` estimation
for Kalman-filter-type models is known to be able to converge to a
degenerate/singular `Q`, causing identifiability loss and instability under
uninformative observations — not a hypothetical specific to this repo, and
not specific to a variational/deep-learning setting either (the same
fixed-point exists in classical linear-Gaussian EM). More recent evidence
from the deep-learning-era literature on this same class of models
(latent neural SDEs trained via ELBO/KL objectives) documents the opposite-
sounding but same-underlying-phenomenon finding: **systematic
underestimation of the diffusion/process-noise term is a known, general
issue** in this model class (Heck, Gelbrecht, Schaub & Boers, "Improving
the noise estimation of latent neural stochastic differential equations,"
*Chaos* 2025, arXiv:2412.17499) — addressed there via an explicit additive
loss penalty on small diffusion size, a *one-sided* regularizer (only ever
pushes `Q` up). Rejected as this plan's primary mechanism (see Design
below): it fights the underestimation bias unconditionally, including the
cases where that bias is genuinely wanted (see the clarification above),
and it does nothing for the opposite failure (residual inflated by `f`'s
own remaining bias early in training, before `f` is competent — see
Design's shrinkage-is-symmetric point).

**Even with fully known, non-latent `z`, jointly gradient-optimizing `f`
and `Q` has the same fundamental degeneracy** — a separate, more basic
justification for residual-based estimation that doesn't depend on any of
the latent-inference subtlety above. A sufficiently flexible `f` can drive
residuals toward zero by memorizing specific training transitions;
jointly-gradient-optimized `Q` then chases that residual down, with the
likelihood diverging as `Q→0` at an exactly-fit point — structurally
identical to factor analysis's Heywood case, just with `f`/`Q` playing the
role `C`/`R` play for observations. A residual-based M-step fixes this
part the same way it fixed `R`: `Q` can't *decouple* from the residual it's
supposed to equal, because it's computed as that residual by construction.
The latent-`z` complication above is a *distinct, additional* risk on top
of this — unlike `R`'s exogenous `y`, the "data" (smoothed `z`) an M-step
for `Q` computes from is itself shaped by `Q` through the inference
machinery, so the decoupling protection that's airtight for `R` is weaker
(not absent) here. This is exactly what Step 1's synthetic harness needs
to test directly: does the M-step behave cleanly in the known-`z` regime
(should, per this argument), and does that protection survive once `z` is
latent and entangled with `Q` via smoothing?

**Consequence for cadence — open question, not yet resolved either way.**
The original reasoning here (avoid fast per-minibatch EMA, since it only
slows collapse rather than preventing it) still holds for a *naive*
residual estimator with no real anchor. But if MAP shrinkage's `n`-vs-
`prior_dof` self-pacing genuinely damps this correctly (see Design), a
faster cadence may turn out to be safe after all — the self-pacing
mechanism, not a slow update schedule per se, would be doing the
protective work. Don't assume either answer; the synthetic harness (Step 1)
should sweep cadence explicitly rather than defaulting to "slow" by
assumption.

## Design

Two-method, duck-typed, **optional** extension on `Approx` (not part of the
abstract contract — mirrors `Gaussian.mstep_stat`'s pattern exactly, so a
future non-Gaussian noise family is free to implement neither, one, or both
with entirely different math):

```
class Approx:
    def mstep_transition_stat(self, t, moment_smoothed_t, moment_smoothed_tm1,
                               transition_fn, ...) -> Array:
        """Per-(batch,time) sufficient statistic:
        (z̄_t - f̄(z̄_{t-1}))^2 + Cov_smoothed(z_t) + spread_of[f(z_{t-1})].
        The spread term reuses Approx.transition_points (see
        transition_points.md) for consistency with the ELBO's own moment
        propagation. Optional — absence means this Approx does not support
        a closed-form transition-noise M-step."""

    def mstep_noise_shrink(self, raw_stat, n, prior, prior_dof) -> Any:
        """Family-specific MAP/shrinkage combination of the aggregated raw
        statistic with a prior, e.g. for MVN an inverse-Wishart-style blend:
        (n * raw_stat + prior_dof * prior) / (n + prior_dof). Kept separate
        from mstep_transition_stat because the correct combination rule is
        family-specific (a Student-t or mixture noise family would not use
        a simple weighted average)."""
```

Standalone driver, mirroring `mstep_gaussian_cov`'s structure:

```
def mstep_dynamics_noise(model, data, *, key, prior=None, prior_dof=None,
                          batch_size=None) -> model:
    """Dispatches on hasattr(model.approx, "mstep_transition_stat");
    raises NotImplementedError otherwise. Runs the smoothing E-step
    (core.smooth — Q's M-step target uses smoothed, not filtered,
    statistics per Shumway & Stoffer 1982), aggregates the raw stat across
    the dataset, calls approx.mstep_noise_shrink, writes the result back via
    approx.canon_to_free into model.noise_free."""
```

**The prior must be anchored to a data-derived quantity, not a hand-picked
constant** — this is the resolution to "MAP shrinkage is fine, but the
fitted model shouldn't be too sensitive to an arbitrary prior value"
(discussed at length; a *genuinely* noninformative prior gives zero
protection, since `prior_dof→0` collapses the shrinkage formula to the raw
MLE-like statistic — so the achievable goal is removing the need to
*hand-pick* the prior's value, not removing its informativeness). Candidate
anchors, in order of preference: (a) a quantity derived from `R`'s
already-well-estimated scale (e.g. a ratio expressing "how much process
noise relative to observation noise" — more transferable across systems
than an absolute number); (b) the empirical residual variance from an
initial encoder-only pass at `z`, before any dynamics fitting. Do **not**
default to reusing `dyn_conf.state_noise` as originally drafted here — that
field is itself a hand-picked, per-system constant, the exact thing this
resolution is meant to avoid.

**Self-pacing via `n` vs. `prior_dof` is the primary mechanism for both
"not too early" and "not too much" — hypothesis, not yet validated.** While
`n` (accumulated statistic mass) is small relative to `prior_dof`, the
prior dominates regardless of what the raw statistic says, *automatically*
— no hand-picked schedule needed. As training genuinely progresses and `n`
grows, the data takes over. This is the same *shape* of behavior a
hand-tuned annealing schedule (`trainer.noise_schedule`, already
implemented and validated empirically for Lorenz) provides, but paced by
genuine evidence accumulation instead of a fixed step count — **this
mechanism may fully subsume the need for `noise_schedule` in this role**,
which would directly satisfy the goal of learning systems like Lorenz
without hand-tuned annealing. Untested; Step 1 should compare self-pacing
shrinkage alone against self-pacing shrinkage plus annealing, not just
assume the schedule becomes redundant.

**The shrinkage formula protects against both under- and over-estimation
simultaneously, symmetrically, by construction** — a real advantage over a
one-sided penalty like Heck et al.'s (see above). `Q_hat = (n·raw_stat +
prior_dof·prior)/(n+prior_dof)` is a convex combination: if `raw_stat` is
biased low (collapse dynamics, or `f` overfitting), `Q_hat` is pulled up
toward `prior`; if `raw_stat` is biased high (early training, `f` not yet
competent, residual inflated by model bias rather than real noise),
`Q_hat` is pulled down toward `prior` — same mechanism, both directions,
for free. This symmetry doesn't itself implement an asymmetric preference
(want small `Q` eventually, no equivalent objection to it being large while
justified) — that comes from the self-pacing decay above, which weakens the
prior's pull from *both* directions as `n` grows, letting the estimate
track a genuinely small (or large) true `Q` once trustworthy.

**A hard numerical floor on `Q` is a separate, independent guard, not part
of this estimation logic** — exactly parallel to `_MIN_VARIANCE` for `R`
(`constraints.py`): needed regardless of whether the shrinkage/self-pacing
above works correctly, purely to prevent literal numerical failure
(log/inverse of an exact-zero `Q`). Do not conflate this with the MAP prior
above — the floor guards against numerical catastrophe; the prior guards
against statistically-unjustified collapse. Both needed, neither
supersedes the other (same relationship `_MIN_VARIANCE` has to the
Heywood-case fix for `R`).

### Tracked, related fix (not part of this plan's core scope, but should land alongside it)

`dyn_conf.state_noise` (a `Dynamics`-config field) currently seeds
`model.noise_free`, even though `noise_free`'s parameterization is entirely
owned by `Approx`, not by whichever `Dynamics` subclass is plugged in. This
is a config-ownership leak: swapping `Dynamics` implementations shouldn't
implicitly carry a noise-init parameter unrelated to the dynamics plugin.
Fix: relocate the init hyperparameter to a top-level `Approx`/`XFADS`-owned
config field (sibling to `conf.state_dim`), matching where this plan's
`prior`/`prior_dof` defaults naturally want to live. This is a config-shape
change and should be done explicitly/documented (README, `AGENTS.md`
config-invariants section, changelog), not silently bundled into a code
diff.

## Required safeguards (non-negotiable)

- **MAP shrinkage toward a data-derived prior is mandatory, not optional**
  — never use the raw statistic as a plain MLE, precisely because the raw
  MLE has a reachable, self-consistent `Q = 0` fixed point. The prior's
  *value* must come from data (anchored to `R`, or a pre-dynamics residual
  estimate — see Design), not be hand-picked per system.
- **A hard numerical floor on `Q`, independent of the above** — a
  `_MIN_VARIANCE`-style guard against literal numerical failure, present
  regardless of whether shrinkage is working correctly (see Design).
- **Prefer an exogenous diagnostic to gate/monitor the update**: a k-step-
  ahead forecast, decoded through the observation model and checked against
  held-out `y`, is the only way to anchor `Q` to something outside the
  self-referential loop (analogous to what `y` already provides for `R`
  directly) — kept as a backstop given the latent-`z` entanglement risk
  (see the "even with known z" discussion above), even though the
  shrinkage formula's own symmetric, self-pacing protection may do most of
  the work.
- **Monitor the trend**: track `||Q||` and posterior variance across
  rounds/epochs; a monotonically shrinking `Q` with no corresponding
  improvement in held-out ELBO/eloglik is the collapse signature to watch
  for, distinct from genuine convergence.

**No longer treated as non-negotiable, pending Step 1's results:**
- ~~Q-annealing / KL-beta warmup must run first~~ — self-pacing shrinkage
  (`n` vs. `prior_dof`) may subsume this role entirely; test both
  configurations rather than assuming annealing is still required
  alongside the M-step.
- ~~Slow/round cadence, not per-minibatch EMA~~ — this was reasoning about
  a *naive*, unanchored residual estimator. Whether a properly
  self-pacing, data-anchored shrinkage estimator is safe at a faster
  (even per-minibatch) cadence is now an open question for Step 1 to
  sweep, not a settled requirement.

## Steps

1. **Build a synthetic validation harness first** — simulate a known
   (linear or mildly nonlinear — Lorenz is the real target system, so
   include it or something comparably chaotic) SSM with a known true `Q`,
   at controlled SNR levels including a deliberately low-SNR case. This is
   essential given the identified risk that the estimator could reinforce
   collapse rather than correct it; internal consistency checks alone are
   not sufficient evidence of correctness here (unlike the `R` case). Also
   include a **fully-known-`z` mode** (no latent inference at all, `f`/`Q`
   fit directly against ground-truth transitions) as the clean baseline the
   "even with known z" argument predicts should work straightforwardly —
   confirming that *before* testing the harder latent-`z` case isolates
   which of the two distinct problems (joint-MLE degeneracy vs.
   latent-entanglement) is actually responsible for any observed failure.
2. Implement `mstep_transition_stat` / `mstep_noise_shrink` for `MVN`,
   reusing `Approx.transition_points` for the spread term. Implement the
   prior as a data-derived anchor (see Design), not a hand-picked constant,
   from the start — don't build the hand-picked version first and fix it
   later.
3. Implement the `mstep_dynamics_noise` driver, prior/prior_dof plumbing,
   the independent hard numerical floor, and the tracked config-relocation
   fix.
4. Run the synthetic harness across SNR levels and these specific,
   previously-unresolved comparisons:
   - Known-`z` baseline vs. latent-`z`: does the M-step recover the true
     `Q` cleanly in both, or only the former?
   - Self-pacing shrinkage alone vs. self-pacing shrinkage + `noise_schedule`
     annealing: does annealing add anything once self-pacing is in place,
     or is it redundant?
   - Cadence sweep (per-minibatch vs. slower rounds): does a properly
     anchored, self-pacing estimator tolerate a fast cadence safely, or
     does the original "slow cadence" caution still hold?
   Report all three comparisons explicitly, not just "did `Q` converge."
5. Only after the synthetic harness passes, validate on a real downstream
   dataset (Lorenz specifically, given that's the stated motivating case)
   without any hand-tuned annealing schedule — the actual test of whether
   this plan achieves its stated goal.
6. Document results, including any observed collapse cases at low SNR,
   which safeguard did or did not prevent them, and the outcome of the
   three comparisons in Step 4.

## Validation plan

- Unit tests for `mstep_transition_stat`/`mstep_noise_shrink` matching an
  independently computed statistic (mirroring the 3 existing
  `mstep_gaussian_cov` tests: exact-match, `NotImplementedError` for
  non-participating `Approx`, duck-typed dispatch).
- **Known-`z` recovery test** — the clean baseline (Step 1); should recover
  the true `Q` straightforwardly per the joint-MLE-degeneracy argument, no
  latent-inference complication involved.
- **Latent-`z`, synthetic-SSM recovery test across SNR levels** — the key
  discriminative test for whether the self-pacing/data-anchored shrinkage
  survives the entanglement the known-`z` case doesn't have.
- **Self-pacing-alone vs. self-pacing-plus-annealing comparison**, and
  **cadence sweep** (per-minibatch through slow rounds) — both on the
  synthetic harness, both explicitly reported regardless of outcome (see
  Steps 4/6).
- Real-data campaign (Lorenz specifically) re-run only after the synthetic
  tests pass, without a hand-tuned annealing schedule — the actual test of
  this plan's stated goal.

## Open questions

- Exact form of the data-derived prior anchor (tied to `R`'s scale? to a
  pre-dynamics residual estimate? some combination?) and how sensitive
  results are to that choice specifically (as opposed to sensitivity to an
  arbitrary hand-picked constant, which the data-derived anchor is meant to
  remove).
- Whether self-pacing shrinkage (`n` vs. `prior_dof`) actually subsumes
  `noise_schedule`-style annealing, or whether both are still needed
  together — Step 1's direct comparison, not assumed either way.
- Whether cadence can safely be per-minibatch (matching `R`'s cadence) once
  properly anchored and self-pacing, or whether the original slow-cadence
  caution still applies — Step 1's cadence sweep.
- Whether the shrink-combinator design generalizes cleanly beyond `MVN`
  once a non-Gaussian noise family is attempted.
- How to define/measure "low SNR" precisely for the stress test (e.g. ratio
  of observation Fisher information to `Q^{-1}` at a given time step).
- Whether the exogenous k-step-ahead diagnostic should be a hard gate
  (blocking the update) or a soft monitor (logged only) — and given the
  shrinkage formula's own symmetric protection, whether it's still needed
  as anything more than a monitored diagnostic (not a blocking gate).

## Dependencies

- `transition_points.md`: **done** — shipped and `MVN` now defaults to
  `use_sigma_points=True`, closing Problem 1 by default.
- Depends on the tracked `dyn_conf.state_noise` config-relocation fix.
- The data-derived prior anchor (Design) likely depends on `R` already
  being well-estimated (via `mstep_gaussian_cov`, already shipped) if that
  anchor choice is used — not a blocking dependency, but worth sequencing
  `R`'s estimation to run first if both are active on the same model.

## Future generalization (noted, out of scope here)

`Observation.mstep`/`Observation.mstep_frozen_paths` have now shipped (see
[mstep_gaussian_cov](mstep_gaussian_cov.md)). This plan's
`mstep_transition_stat`/`mstep_noise_shrink` are candidates to converge on
that same `mstep`/`mstep_frozen_paths` vocabulary, extending it to `Approx`
(and possibly `Dynamics` for subclasses whose map is linear-in-parameters).
Whether `Q` can safely adopt `R`'s continuous per-step cadence is now an
open question for Step 1 to resolve empirically (see "Consequence for
cadence" and the Required Safeguards' cadence item above), not a settled
conclusion either way. Deliberately deferred, not part of this plan's
scope.
