# Plan: M-step for transition/process noise (`mstep` on the noise-owning component)

Status: proposed, not implemented; design substantially refined through
extended discussion (see below), but **still higher risk than the R
estimator** — read the runaway-loop analysis below before implementing, and
build the synthetic validation harness (Step 1) before writing any
production code. Several claims below are stated as **hypotheses to test
in that harness**, not established conclusions — flagged explicitly where
relevant.

See also: [mstep_gaussian_cov](mstep_gaussian_cov.md), [transition_points](transition_points.md).

## The actual success metric: accurate dynamics, not accurate `Q`

**Stated explicitly up front because it reshapes everything below**: the
goal of this plan is *not* to recover the statistically "correct" `Q`. The
goal is accurate estimation of the dynamics function `f`. `Q` is
instrumental — a training mechanism that, when appropriately paced,
pressures `f` to explain trajectory variance directly instead of
attributing it to noise (exactly the inductive bias wanted for learning
genuinely deterministic-ish systems, e.g. Lorenz).

Consequence: **every design choice below should be validated against "does
this produce accurate `f`," not "does this produce an accurate `Q`."** `Q`'s
own trajectory (does it end up over- or under-estimated relative to
whatever the "true" value might be) is a *diagnostic* for understanding
training dynamics, never the pass/fail criterion. This resolves several
tensions that only existed under the older framing — see Design.

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
   pacing problem, not a destination problem). Unlike `R`, which is checked
   against *fixed, exogenous* data `y`, the KL term comparing `q(z_t)` to
   the `Q`-derived predictive prior has **both sides jointly trainable**.
   Gradient descent can shrink `q(z_t)`'s posterior width and `Q` together,
   driving KL toward zero without ever consulting real data — especially in
   low-SNR regimes, where the per-step observation evidence pulling the
   posterior mean is weak. `Q = 0` (deterministic dynamics) is a
   **self-consistent fixed point** of this process.

   **Important clarification: driving `Q` toward small values is not
   itself the failure mode — it is often the goal.** A small `Q` forces the
   dynamics network `f` to explain trajectory variance directly rather than
   attributing it to noise. The actual risk is `Q` becoming small *before*
   `f` is competent enough to explain that variance, which then damages the
   encoder/dynamics/readout jointly (an empirically observed failure mode,
   not just a theoretical one). So the design goal is **pacing** `Q`'s
   descent to track `f`'s genuine improvement — not preventing descent
   altogether, and not chasing an absolutely "correct" `Q` value either
   (see the success-metric section above).

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
from the deep-learning-era literature on this same class of models (latent
neural SDEs trained via ELBO/KL objectives) documents the opposite-sounding
but same-underlying-phenomenon finding: **systematic underestimation of the
diffusion/process-noise term is a known, general issue** in this model
class (Heck, Gelbrecht, Schaub & Boers, "Improving the noise estimation of
latent neural stochastic differential equations," *Chaos* 2025,
arXiv:2412.17499) — addressed there via an explicit additive loss penalty
on small diffusion size, a *one-sided* regularizer (only ever pushes `Q`
up). Not adopted here as the primary mechanism: it fights the
underestimation bias unconditionally, including the cases where that bias
is genuinely wanted, and it does nothing for the opposite failure (residual
inflated by `f`'s own remaining bias early in training). The mechanism
adopted instead (see Design) is directionally similar but self-regulating,
not a fixed-strength penalty.

**Even with fully known, non-latent `z`, jointly gradient-optimizing `f`
and `Q` has the same fundamental degeneracy** — a separate, more basic
justification for residual-based estimation that doesn't depend on any of
the latent-inference subtlety above. A sufficiently flexible `f` can drive
residuals toward zero by memorizing specific training transitions;
jointly-gradient-optimized `Q` then chases that residual down, with the
likelihood diverging as `Q→0` at an exactly-fit point — structurally
identical to factor analysis's Heywood case, just with `f`/`Q` playing the
role `C`/`R` play for observations. A residual-based M-step fixes this part
the same way it fixed `R`: `Q` can't *decouple* from the residual it's
supposed to equal, because it's computed as that residual by construction.
The latent-`z` complication above is a *distinct, additional* risk on top
of this — unlike `R`'s exogenous `y`, the "data" (smoothed `z`) an M-step
for `Q` computes from is itself shaped by `Q` through the inference
machinery, so the decoupling protection that's airtight for `R` is weaker
(not absent) here. This is exactly what Step 1's synthetic harness needs to
test directly: does the M-step behave cleanly in the known-`z` regime
(should, per this argument, and checkable there as a standard
generalization/overfitting question — does `f` fit held-out transitions or
just memorize training ones), and does that protection survive once `z` is
latent and entangled with `Q` via smoothing?

**Consequence for cadence — open question, not yet resolved either way.**
The original reasoning here (avoid fast per-minibatch EMA, since it only
slows collapse rather than preventing it) still holds for a *naive*
residual estimator with no real anchor. Whether the mechanisms in Design
(the cross-covariance-omission's self-regulating bias, primarily) damp this
adequately at a fast cadence is an open question for Step 1's cadence
sweep, not a settled requirement either way.

## Design

Two-method, duck-typed, **optional** extension, mirroring `Gaussian.
mstep_stat`'s pattern (a future non-Gaussian noise family is free to
implement neither, one, or both with entirely different math). Full
`mstep`/`train()` integration is the target, matching `R`'s precedent
exactly (not a standalone-only utility) — see "Integration into `train()`"
below.

**Where it lives**: leaning toward `Approx` directly (mirroring `Observation.
mstep`'s ABC-level placement), since `Q`'s parameterization already lives
there today (`approx.canon_to_moment(approx.free_to_canon(model.noise_free))`)
— not yet a final decision; a small, dedicated sub-object (mirroring how
`Gaussian` sits inside `GLM`) is the alternative if a future non-Gaussian
noise family wants sufficiently different math that a family-specific
component makes more sense than a method directly on `Approx`.

```
class Approx:  # or a small dedicated sub-object -- see above
    def mstep_transition_stat(self, moment_smoothed_t, moment_smoothed_tm1,
                               transition_fn) -> Array:
        """Per-(batch,time) sufficient statistic for the transition-noise
        M-step, computed entirely from smoothed quantities -- deliberately
        decoupled from Approx.transition_points (MC or UT). transition_points
        answers a different question ("propagate q(z_{t-1})'s uncertainty
        forward, before seeing y_t"); this statistic is a posterior
        expectation over the *joint* smoothed distribution of (z_{t-1}, z_t),
        which needs no forward-propagation machinery at all. v1
        approximation: no cross-covariance term (see below) -- computed
        directly from the two marginal smoothed moments and f evaluated at
        the smoothed mean, using a Jacobian-based (jax.jacrev(f)) correction
        for propagating Cov(z_{t-1}) through f's local linearization.
        Optional -- absence means this Approx does not support a
        closed-form transition-noise M-step."""

    def mstep_noise_shrink(self, raw_stat, floor) -> Any:
        """Combine the aggregated raw statistic with a numerical-safety
        floor -- NOT a meaningful Bayesian prior (see below). Candidate
        implementations: a straightforward jnp.maximum(raw_stat, floor)
        clip, or a shrinkage blend with a deliberately small/minimal weight
        on the floor term. Kept as its own method because the combination
        rule may still be family-specific even though it no longer needs to
        encode genuine prior belief."""
```

### The prior/floor is for numerical safety only, not a meaningful anchor

Resolved after discussion: **do not build a data-derived "belief" prior**
(an earlier draft of this plan proposed anchoring to `R`'s scale, or to a
pre-dynamics-fit residual variance — both rejected). Given the success
metric is accurate `f`, not accurate `Q`, there's no need for the floor
value to represent a defensible Bayesian belief about `Q`'s true scale —
it only needs to keep the M-step's arithmetic away from literal numerical
failure (log/inverse of an exact-zero `Q`), exactly the same narrow role
`_MIN_VARIANCE` plays for `R` (`constraints.py`). This collapses what were
previously two separate mechanisms (an informative prior + an independent
hard floor) into one.

**Tuning this value is explicitly out of scope, and it is not meant to be
calibrated per dataset.** The intended policy is "smallest value that avoids
underflow" — the same philosophy as `_MIN_VARIANCE` itself
(`jnp.finfo(dtype).eps`-scale), not a moderate, "reasonable-looking"
constant. Whatever is actually optimal is inherently data-dependent, and
finding that optimum is not this plan's job; picking the smallest safe
value sidesteps needing to. In practice this likely means reusing
`_MIN_VARIANCE` directly rather than introducing a second, separate
constant for `Q`.

**Important consequence for any residual "shrinkage" weight**: if this
floor value is tiny (as it should be, being purely a safety net), any
weight given to it in a blend must also be small — a tiny floor combined
with a *substantial* blend weight would anchor early training close to
that tiny value, directly reintroducing "starts too small too early."
There is therefore **no meaningful self-pacing to be had from a
floor/prior weighting mechanism** in this design — self-pacing, if it
exists at all, has to come from elsewhere (see next section). A plain
`jnp.maximum(raw_stat, floor)` clip is the simplest correct
implementation, and may be all that's needed; a soft blend with minimal
weight is a reasonable alternative if a smoother/differentiable-everywhere
combination is preferred, as long as its weight stays small enough not to
anchor anything.

### The cross-covariance omission — a self-regulating anti-collapse bias, adopted for v1

The classical closed-form M-step for `Q` (Shumway & Stoffer 1982, linear
case) needs the *smoothed cross-covariance* `Cov(z_t, z_{t-1})`, not just
the two marginal covariances — expanding `E[(z_t - Az_{t-1})(z_t-Az_{t-1})^T]`
under the joint smoothed distribution requires `E[z_t z_{t-1}^T]`, which
needs `Cov(z_t, z_{t-1})` whenever `z_t`/`z_{t-1}` are correlated (they
generally are, under smoothing). **This repo's smoothing algorithm
(exponential-family, additive-natural-parameter, not the classical
mean/covariance RTS recursion) does not currently expose an analogous
lag-one/cross-covariance quantity** (`core.py`'s `smooth()` returns only
marginal per-timestep moments) — deriving it would be genuinely new,
nontrivial machinery specific to this codebase's smoothing formalism, not
a drop-in reuse of the classical RTS-smoother-gain formula.

**v1 decision: omit the cross-covariance term**, accepting the resulting
approximation error, because the error is directionally favorable and
self-regulating rather than a fixed, arbitrary bias:

- Dropping the (positive, under normal dynamics) cross term means the
  approximation computes `Var(z_t) + A^2 Var(z_{t-1})` instead of the true
  (smaller) `Var(z_t) + A^2 Var(z_{t-1}) - 2A\,Cov(z_t,z_{t-1})` — i.e. it
  systematically **overestimates** the residual statistic, which is the
  favorable direction relative to the collapse pathology this whole plan
  worries about.
- Crucially, the *magnitude* of this overestimation **scales with how
  collapsed the posterior already is**: as `Q→0` and dynamics become
  near-deterministic, `z_t` and `z_{t-1}` become *more* correlated
  (`Cov(z_t,z_{t-1}) → A\cdot Var(z_{t-1})`), so the omitted term — and
  therefore the compensating overestimation — grows exactly when collapse
  risk is highest. This is a self-regulating counter-pressure, not a fixed
  offset, requiring no hand-tuned strength parameter.
- **Two costs, to be checked empirically, not assumed away**: (1) this is
  a qualitative, directional argument — whether it's *strong enough* on its
  own (with or without the floor/clip above) is unknown until measured;
  (2) it does not vanish once `f` is fully competent — any stable dynamics
  induces some `z_t`/`z_{t-1}` correlation, so the *converged* estimate
  will be systematically somewhat inflated relative to the true `Q`, not
  just protected during training. Per the success-metric framing, that's
  only a real problem if it measurably hurts `f`'s accuracy — check via the
  with/without-cross-covariance comparison in Steps/Validation, not by
  reasoning about `Q`'s own bias in isolation.

This is now the **primary candidate protection mechanism** against Problem
2, superseding an earlier draft of this plan that relied on `n`-vs-
`prior_dof` self-pacing from an informative shrinkage prior — that
mechanism required a floor value informative enough to matter, which the
previous section rules out. If the harness shows this alone is
insufficient, revisit either (a) deriving the exact cross-covariance term,
or (b) reintroducing some other explicit pacing mechanism (annealing,
gating) — not by quietly making the floor informative again.

**The v1 formula, written out precisely** (previously only described
conceptually): with `m = mean(moment_smoothed_tm1)`, `P = Cov(moment_
smoothed_tm1)`, `m' = mean(moment_smoothed_t)`, `P' = Cov(moment_smoothed_t)`,
and `J = jax.jacrev(transition_fn)(m)`:

```
r = m' - transition_fn(m)
raw_stat = outer(r, r) + P' + J @ P @ J.T
```

Note this is a first-order (Jacobian/delta-method) linearization of
"propagate `P` through `f`" — conceptually similar to what `transition_points`
does for the forward prediction step (UT is exact to second order for
smooth `f`; this is only first-order), but computed independently via
direct autodiff on `transition_fn`, not by calling `Approx.transition_points`
or depending on its `mc_size`/dispatch machinery. That's what "decoupled"
means here precisely: no shared code path or abstraction with the forward
prediction step, not "conceptually unrelated math" — the M-step's own
linearization can be cheaper (one Jacobian, not `2D+1` points) since it
only needs a local correction around a single smoothed mean, not a
globally accurate moment-matched propagation.

### Integration into `train()`

Full integration, matching `R`'s final design (not standalone-only): an
`mstep` on the noise-owning component (see placement question above),
called from `train()` analogous to `Observation.mstep`, with an equivalent
of `mstep_mode`/cadence control. Given the higher risk profile here, this
should land only after the synthetic harness (Steps 1-2) validates the
mechanism — unlike `R`, where the mechanism's correctness was
straightforward enough to validate via unit tests alone before wiring into
`train()`.

### Tracked, related fix (not part of this plan's core scope, but should land alongside it)

`dyn_conf.state_noise` (a `Dynamics`-config field) currently seeds
`model.noise_free`, even though `noise_free`'s parameterization is entirely
owned by `Approx`, not by whichever `Dynamics` subclass is plugged in. This
is a config-ownership leak: swapping `Dynamics` implementations shouldn't
implicitly carry a noise-init parameter unrelated to the dynamics plugin.
Fix: relocate the init hyperparameter to a top-level `Approx`/`XFADS`-owned
config field (sibling to `conf.state_dim`). This is a config-shape change
and should be done explicitly/documented (README, `AGENTS.md`
config-invariants section, changelog), not silently bundled into a code
diff. Not blocking for the synthetic harness (Step 1 can use direct
arguments/hardcoded values without this fix landing first) — worth deciding
explicitly that it's deferred rather than assuming it blocks Step 1.

## Required safeguards (non-negotiable)

- **A numerical-safety floor is needed, but it is *not* a Bayesian anchor**
  — see Design. Keep it minimal; do not let it (or any blend weight
  attached to it) become informative enough to anchor early training.
- **Prefer an exogenous diagnostic to gate/monitor the update**: a k-step-
  ahead forecast, decoded through the observation model and checked against
  held-out `y`, is the only way to anchor `Q` to something outside the
  self-referential loop (analogous to what `y` already provides for `R`
  directly) — kept as a backstop given the latent-`z` entanglement risk,
  even though the cross-covariance-omission bias may do most of the work.
  Start as a logged monitor, not a blocking gate, for the first harness
  pass; upgrade only if the monitor shows it's needed.
- **Monitor the trend**: track `||Q||`, posterior variance, and (per the
  success-metric framing) `f`'s own held-out forecast accuracy across
  rounds/epochs — a monotonically shrinking `Q` with no corresponding
  improvement in `f`'s accuracy is the collapse signature to watch for,
  distinct from genuine convergence.

**Open, not yet resolved either way (Step 1 must test, not assume):**
- Whether `noise_schedule`-style hand-tuned annealing is still needed
  alongside the cross-covariance-omission mechanism, or becomes redundant.
- Whether cadence can safely be per-minibatch (matching `R`'s cadence), or
  whether the original slow-cadence caution still applies.
- Whether the cross-covariance omission alone is sufficient, or the exact
  lag-one-covariance term needs to be derived after all.

## Steps

1. **Build a synthetic validation harness first**, reusing existing
   infrastructure rather than new generators where possible:
   - The Van der Pol (`examples/vdp_example.py`) and oscillator-bank
     (`benchmarks/benchmark_highd_oscillator.py`) systems, both already
     implemented — but currently **pure deterministic RK4 integration with
     no stochastic noise injection at all**. Adding a known `Q_true` requires
     modifying the generators (e.g. `z_next = rk4_step(z, dt) + sqrt(Q_true)
     @ noise` per step, or an Euler-Maruyama-consistent step) — real,
     necessary work, not just reuse.
   - **Add a Lorenz system generator** (not currently in this repo) — the
     actual motivating target system, same noise-injection treatment.
   - Include a **fully-known-`z` mode** (no latent inference at all, `f`/`Q`
     fit directly against ground-truth transitions) as the clean baseline —
     isolates the joint-MLE-degeneracy argument (should work straightforwardly)
     from the latent-`z` entanglement risk (may not).
   - SNR: start with a single value (`SNR = Q_true / R = 1`) rather than a
     full sweep, to get a first working comparison quickly; expand to a
     sweep only if that single point doesn't already resolve the open
     questions below.
2. Implement `mstep_transition_stat` / `mstep_noise_shrink`, decoupled from
   `Approx.transition_points`, with the cross-covariance term omitted (v1)
   and the floor/clip as described in Design.
3. Implement `train()` integration (cadence control analogous to
   `mstep_mode`) and the tracked config-relocation fix (or explicitly defer
   the latter — see Design).
4. Run the harness with these specific, previously-unresolved comparisons,
   using **`f`'s accuracy against the true dynamics as the primary metric**
   (vector-field/forecast error against ground truth; `Q`'s own trajectory
   as diagnostic only):
   - Known-`z` baseline vs. latent-`z`: does `f` recover the true dynamics
     cleanly in both, or only the former (i.e. does the latent-`z`
     entanglement matter in practice)?
   - With vs. without the cross-covariance term (approximate now,
     exact/derived later if this comparison shows it matters): does
     omitting it measurably hurt `f`'s accuracy, or is the self-regulating
     bias good enough on its own?
   - With vs. without `noise_schedule`-style annealing, on top of the
     cross-covariance-omission mechanism: does annealing still add
     anything, or is it now redundant?
   - Cadence sweep (per-minibatch vs. slower rounds): does the mechanism
     tolerate a fast cadence safely?
   Report all comparisons explicitly, not just "did `f` converge well."
5. Only after the synthetic harness passes, validate on a real downstream
   Lorenz dataset without any hand-tuned annealing schedule — the actual
   test of whether this plan achieves its stated goal.
6. Document results, including any observed collapse cases, which
   safeguard did or did not prevent them (measured via `f`'s accuracy, not
   `Q`'s value), and the outcome of the comparisons in Step 4.

## Validation plan

- Unit tests for `mstep_transition_stat`/`mstep_noise_shrink` matching an
  independently computed statistic (mirroring the exact-match /
  `NotImplementedError` / duck-typed-dispatch pattern from the existing
  `mstep_gaussian_cov` tests, adapted to this design's decoupled
  computation).
- **Known-`z` recovery test** — the clean baseline (Step 1); should recover
  accurate `f` straightforwardly, checkable as a standard held-out
  generalization test (does `f` fit unseen transitions, or has it
  memorized training ones).
- **Latent-`z`, synthetic-SSM recovery test (VDP, oscillator-bank, Lorenz)**
  at `SNR=1` initially — the key discriminative test for whether the
  cross-covariance-omission mechanism survives the entanglement the
  known-`z` case doesn't have. Primary metric: `f`'s accuracy against the
  true dynamics, not `Q`'s value.
- **With/without cross-covariance term, with/without annealing, and a
  cadence sweep** — all three explicitly reported regardless of outcome
  (see Steps 4/6).
- Real-data campaign (Lorenz specifically) re-run only after the synthetic
  tests pass, without a hand-tuned annealing schedule — the actual test of
  this plan's stated goal.

## Open questions

- Final placement of `mstep`/`mstep_transition_stat`: directly on `Approx`,
  or a small dedicated sub-object (see Design)?
- Whether the cross-covariance-omission bias alone is sufficient (per the
  with/without comparison), or the exact lag-one-covariance term needs to
  be derived for this codebase's smoothing formalism after all.
- Whether `noise_schedule`-style annealing is still needed alongside this
  mechanism, or becomes redundant.
- Whether cadence can safely be per-minibatch, or the original slow-cadence
  caution still applies.
- Whether the design generalizes cleanly beyond `MVN` once a non-Gaussian
  noise family is attempted.
- Whether the exogenous k-step-ahead diagnostic should ever become a hard
  gate (blocking the update) rather than staying a logged monitor.

## Dependencies

- `transition_points.md`: **done** — shipped and `MVN` now defaults to
  `use_sigma_points=True`, closing Problem 1 by default. Note this plan's
  own M-step statistic is explicitly *not* built on top of it (see Design).
- The tracked `dyn_conf.state_noise` config-relocation fix: real, but
  explicitly deferred — not blocking for the synthetic harness (Step 1).
- Deriving the exact lag-one/cross-covariance term (if the harness shows
  the v1 approximation insufficient) is new, XFADS-specific machinery, not
  a dependency on anything already planned elsewhere.

## Future generalization (noted, out of scope here)

`Observation.mstep`/`Observation.mstep_frozen_paths` have now shipped (see
[mstep_gaussian_cov](mstep_gaussian_cov.md)). This plan's
`mstep_transition_stat`/`mstep_noise_shrink` are candidates to converge on
that same `mstep`/`mstep_frozen_paths` vocabulary (see the "Integration
into `train()`" section above, which already targets this). Whether `Q`
can safely adopt `R`'s continuous per-step cadence is an open question for
Step 1 to resolve empirically, not a settled conclusion either way.
