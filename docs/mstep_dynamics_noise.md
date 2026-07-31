# Plan: M-step for transition/process noise (`mstep` on the noise-owning component)

Status: **implemented and empirically validated on synthetic data
(known-z and latent-z Lorenz).** Two independent experiments
(`benchmarks/mstep_known_z_baseline.py`, `benchmarks/mstep_lorenz_latent.py`)
support the same mechanism: keep `Q` inside the training loss (never fully
decoupled), update it periodically via a **MAP-shrunk M-step toward a
genuinely informative prior** rather than either free gradient descent or a
numerical-safety-only floor. This reverses an earlier draft of this plan
(see Design) that concluded the prior should be non-informative.

The current public configuration is top-level `q_scale` (positive Q
variance) and `q_mstep` (default `true`). `q_scale` initializes Q and, when
`q_mstep=true`, centers its M-step prior; the prior pseudocount is derived as
`state_dim + 1`. `q_mstep=false` leaves Q SGD-managed. The default joint R/Q
Normal training accumulates pre-SGD minibatch R/Q statistics and finalizes
both at each epoch boundary without an additional inference pass. The old
configuration names and cadence discussion appearing below are retained only
as historical design record.

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
training dynamics, never the pass/fail criterion. Confirmed directly by
experiment (see Validation plan): the condition with the *worst* `Q`
accuracy of the three tested was not the condition with the worst `f`
accuracy, and vice versa.

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
   encoder/dynamics/readout jointly. So the design goal is **pacing** `Q`'s
   descent to track `f`'s genuine improvement — not preventing descent
   altogether, and not chasing an absolutely "correct" `Q` value either
   (see the success-metric section above).

### Empirical finding: the dominant observed failure mode is overestimation, not collapse

Both experiments (known-z, `--log-q0 0.0` giving `Q` init `≈0.69`; latent-z
Lorenz, `Q` init `1.0`) show joint gradient optimization of `f` and `Q`
converging toward a **substantially overestimated** `Q` (`~0.37`–`0.70` vs.
true `0.01`), slowly and monotonically, with **no sign of the runaway
collapse-toward-zero this Problem section originally focused on**, within
realistic training budgets (30–150 epochs / rounds tested).

The mechanism (conjectured, matches both experiments): **explaining
residual variance via a free noise parameter is a much easier optimization
move than improving a whole dynamics network** — increasing `Q` is a
trivial one-parameter fit that immediately reduces the joint NLL; improving
`f` requires real gradient progress through a harder, higher-dimensional
problem. Free joint optimization takes the easy path by default.

This does not mean the collapse pathology below is fictitious — it remains
a documented, general phenomenon in the classical EM-for-SSM literature,
and in the deep-learning-era literature for this exact model class (see
below) — only that it was not what was observed to dominate in these
specific experiments, at these training budgets. Both risks (collapse and
overestimation-driven stalling) are real reasons free joint gradient
optimization of `Q` is unreliable; the validated mechanism (see Design)
addresses both by construction, not by choosing one to worry about.

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
imposed by `Q`" at each step — small in low-SNR regimes. This is a
documented phenomenon in the classical linear-Gaussian SSM/EM literature:
EM-based `Q` estimation for Kalman-filter-type models is known to be able
to converge to a degenerate/singular `Q` under uninformative observations —
not a hypothetical specific to this repo. More recent evidence from the
deep-learning-era literature on this same class of models (latent neural
SDEs trained via ELBO/KL objectives) documents the opposite-sounding but
same-underlying-phenomenon finding: **systematic underestimation of the
diffusion/process-noise term is a known, general issue** in this model
class (Heck, Gelbrecht, Schaub & Boers, "Improving the noise estimation of
latent neural stochastic differential equations," *Chaos* 2025,
arXiv:2412.17499) — addressed there via an explicit additive loss penalty
on small diffusion size, a *one-sided* regularizer (only ever pushes `Q`
up). **Not adopted here**, and the experiments give a concrete reason
beyond the original objection (that it fights the wanted bias
unconditionally): a one-sided penalty can't help with the overestimation
failure mode that turned out to dominate in practice, only the
underestimation one.

**Even with fully known, non-latent `z`, jointly gradient-optimizing `f`
and `Q` has the same fundamental degeneracy** — a separate, more basic
justification for residual-based estimation that doesn't depend on any
latent-inference subtlety. A sufficiently flexible `f` can drive residuals
toward zero by memorizing specific training transitions; jointly-
gradient-optimized `Q` then chases that residual down — structurally
identical to factor analysis's Heywood case, just with `f`/`Q` playing the
role `C`/`R` play for observations. A residual-based M-step fixes this part
the same way it fixed `R`: `Q` can't *decouple* from the residual it's
supposed to equal, because it's computed as that residual by construction.
The latent-`z` complication is a *distinct, additional* risk on top of
this — unlike `R`'s exogenous `y`, the "data" (smoothed `z`) an M-step for
`Q` computes from is itself shaped by `Q` through the inference machinery.
**Tested directly (see Validation plan)**: both the known-z and latent-z
experiments show the same qualitative pattern (joint optimization stalls at
an overestimated `Q`; the M-step-based alternating approach recovers a
better `Q` and, more importantly, better `f`), suggesting the latent-z
entanglement doesn't qualitatively change the picture, at least at the
scale tested.

## Design

### The validated mechanism: alternating EM, `Q` kept in the loss, MAP-shrunk toward an informative prior

**This reverses an earlier draft's conclusion.** An earlier version of this
section decided the prior/floor should be non-informative (numerical-safety
only, `_MIN_VARIANCE`-scale), reasoning that a genuinely informative prior
would need a substantial blend weight that could itself anchor early
training too tightly. That reasoning wasn't wrong in the abstract, but the
actual experiments show the informative-prior mechanism, applied
correctly, wins:

- **Known-z** (`benchmarks/mstep_known_z_baseline.py`, `q_true=0.01`,
  150 epochs / 5 rounds): plain joint MLE (`Approach A`) reaches
  `Q≈0.371`, flow-field RMSE `0.0257`. Fully decoupling `Q` from `f`'s loss
  (`Approach B`: plain MSE for `f`, then a floor-only M-step) reaches a
  more accurate `Q≈0.0113` but a *worse* flow-field RMSE `0.0356`.
  **Alternating EM** (`Approach C`: `Q` stays in the joint NLL loss the
  whole time, scaling `f`'s gradient by `1/Q` exactly as in A, but `Q`
  itself is replaced every 30 epochs by `(n·raw_stat + prior_dof·prior)/
  (n+prior_dof)` with `prior=1.0, prior_dof=0.1n`) reaches `Q≈0.101` (3.7x
  closer to truth than A) **and** the best-or-near-best flow-field RMSE
  `0.0266` (nearly matching A, clearly beating B).
- **Latent-z Lorenz** (`benchmarks/mstep_lorenz_latent.py`, real XFADS
  model with posterior inference, `q_true=0.01`, 100 epochs / 5 rounds):
  A reaches `Q≈0.703`, flow-field RMSE `0.542`. B (`Q` frozen constant at
  its init value for all 100 epochs, M-step applied once at the end)
  reaches `Q≈1.056`, flow-field RMSE `0.638` (worse on both counts — the
  single-shot-at-the-end design applies the M-step to an undertrained `f`,
  and freezing `Q` constant denies `f` any of A's gradual-easing dynamic).
  **Alternating EM** (same design as known-z's C, `prior=1.0,
  prior_dof_frac=0.1`, 5 rounds of 20 epochs) reaches `Q≈1.321` (worse
  than A on this axis, with a genuinely interesting per-dimension split —
  the x/y dimensions shrink over rounds while z grows, plausibly reflecting
  Lorenz's z-coordinate having the largest local range/fastest dynamics,
  not yet independently confirmed) but the **best flow-field RMSE of all
  three, `0.318`** — roughly 40% better than A, 50% better than B.

Both experiments point the same direction: **decoupling `Q` from the loss
entirely (B) is worse than keeping it in the loss (A, C) on the metric that
matters, even though B's own `Q` estimate can be more numerically
accurate.** And **controlling `Q`'s value via periodic MAP-shrunk updates
(C) beats free gradient descent (A) on `f`-accuracy**, in the latent-z case
clearly, in the known-z case by a smaller margin while getting a much
better `Q` too. Caveats, not yet resolved: single-seed runs only; the
`prior=1.0, prior_dof_frac=0.1` hyperparameters were picked once, not
swept; only Lorenz has been tested in the latent-z setting.

### A converging conjecture: free `Q` benefits SGD as an extra parameter, independent of its value

Motivated by the B-vs-C gap in both experiments: B removes `Q` from `f`'s
loss entirely (plain MSE), while C keeps it in the loss (still scaling
`f`'s gradient by `1/Q`) but controls its value differently. C beats B on
`f`-accuracy in both experiments, despite A/C's `Q` not obviously being
"correct" in either case. This suggests **the benefit isn't specifically
about `Q`'s value being right — it's about `Q` being a live, present
parameter in the optimization at all**, consistent with the general
observation that free/adaptive per-parameter scaling (of which a trainable
noise term is one instance) often helps gradient-based optimization
independent of what that scaling factor converges to. Not independently
verified beyond these two experiments; stated here as a working hypothesis
the alternating-EM design is consistent with, not a proven mechanism.

### Final design: one combined `Approx.shrink`, `XFADS.mstep` as the composing entry point

This design went through four iterations before landing here, each
changed for a concrete, checked-against-precedent (or explicit
preference) reason, not arbitrary churn:

1. `Approx.mstep_transition_stat`/`mstep_noise_shrink` as two methods,
   plus a standalone driver function analogous to `mstep_gaussian_cov` --
   rejected once `Approx.mstep` was found to conflate distribution-family
   math with knowledge of `noise_free`'s name.
2. `Approx.shrink`/standalone `mstep_transition_stat` (a `core.py`
   function) -- rejected once `mstep_transition_stat` was found to call
   `approx.unpack`, which is *not* part of the `Approx` ABC and is
   MVN-specific (unlike `core.expected_predictive_moment`, the analogy
   used to justify a standalone function, which only calls genuine ABC
   methods). A standalone function assuming one specific subclass's
   internals isn't actually family-agnostic.
3. `Approx.shrink`/`Approx.mstep_transition_stat` as two separate
   *methods* (both on `MVN`) -- rejected once checked against
   `Observation.mstep`/`Gaussian.mstep`'s actual precedent:
   `GLM.mstep` calls exactly *one* method (`Gaussian.mstep`);
   `Gaussian.mstep_stat` exists only as `Gaussian.mstep`'s own internal
   helper, never called externally by the orchestrator. The two-method
   split required `XFADS.mstep` to sequence both calls itself *and* to
   pre-slice `moment`/`u`/`c` into aligned `(t, t-1)` pairs before calling
   the first one -- both symptoms of the same problem: knowledge that
   belonged inside the method leaking out to the orchestrator. The
   "`shrink` might be independently reusable for other covariance-shaped
   quantities" justification for keeping them separate was also purely
   speculative -- no second use case for it ever existed. Merged into one
   method, named `mstep_transition_noise` at this point (a new name
   describing the combined behavior, since neither `shrink` nor
   `mstep_transition_stat` alone described it).
4. `Approx.mstep_transition_noise` (the merged method) renamed back to
   `Approx.shrink` -- explicit naming preference: reuse the shorter,
   already-established name for the combined method rather than
   introduce a new, longer one. A name doesn't need to describe every
   internal step (the method still computes the statistic *and* shrinks
   it) -- "shrink" describes the salient operation/purpose well enough,
   matching how e.g. `Observation.mstep` doesn't spell out everything
   `Gaussian.mstep` does internally either.
5. The per-pair statistic's `Cov[transition_fn(z_{t-1})]` term: originally
   a first-order Taylor/Jacobian linearization (`J @ P @ J.T`, `J =
   jacrev(transition_fn)(m)`) -- replaced with the same weighted
   point-set propagation (Monte Carlo, or `MVN`'s current default,
   deterministic unscented-transform sigma points) already used for the
   prediction step's `core.expected_predictive_moment`, once a direct
   question ("why not reuse the sampling/UT machinery instead of a
   Jacobian?") was checked against the actual math: the Jacobian is only
   a first-order approximation to how `z_{t-1}`'s uncertainty propagates
   through a possibly-nonlinear `transition_fn`, while UT/MC capture the
   true nonlinear propagation (exact to the 3rd moment for UT, exact in
   the MC limit) -- the same accuracy argument that already motivated
   making UT the default propagation policy elsewhere in this codebase.
   Reusing `transition_points` here also keeps this statistic's
   propagation accuracy consistent with the rest of the model's own
   choice, instead of a second, separately-unvalidated approximation
   scheme coexisting alongside it. `core.expected_predictive_moment`
   itself could not be called directly (it bakes `Q` into the propagated
   moment via `approx.predictive_moment` -- the filtering-time question,
   "propagate forward before seeing `y_t`" -- but `Q` is exactly what
   `shrink` is estimating, so it must be excluded); instead
   `expected_predictive_moment` was split, extracting its noise-free
   prefix (`approx.transition_points` + push through `f`) into a new,
   standalone `core.propagate_transition_points`, shared by both. Also
   surfaced a latent bug while implementing this: the previous
   Jacobian-based code reused one single outer `key` across every
   `(batch, time)` pair (harmless there, since `jacrev`/`jax.linearize`
   don't consume randomness) -- for the new sampling-based statistic this
   would have correlated the point-propagation's own randomness across
   every pair whenever `use_sigma_points=False`. Fixed by splitting a
   distinct key per pair, matching `core.filter`/`core.smooth`'s own
   established per-timestep `jrnd.split` + `vmap` convention.
6. `Approx.shrink`'s own propagation call was itself removed, given a
   direct question ("the forward pass already computes this same
   propagation for its own noise-included predictive moment -- can
   `shrink` reuse it instead of computing it again?"). Traced against
   the actual mode dispatch (`core._site_filter`/`nofilt`/`causal`):
   whatever `q(z_{t-1})` the forward pass propagates through the
   transition for its own ELBO/KL term is, for `filter`/`smooth`/`nofilt`
   modes, *exactly* the same distribution `shrink` would later
   re-propagate -- initially thought to differ (filtering vs. smoothed
   posterior), corrected once it became clear this codebase's
   "smoothing" is a single forward recursion with richer per-timestep
   sites (`alpha + beta`, `beta` already whole-sequence-informed), not a
   separate forward-filter/backward-smooth algorithm with two genuinely
   different marginals. Only `causal` mode differs (its *returned*
   `moment` is a post-hoc `beta`-reconstruction, computed *after* its
   internal `filter()` call already ran the propagation on the
   filtering-only marginal) -- accepted as a deliberate trade: `causal`'s
   reused statistic is the filtering statistic, not one over its own
   returned `moment`, since the latter was never computed by any
   propagation step to reuse in the first place. `_site_filter`/`nofilt`
   retain the raw propagated point set from the exact same
   `propagate_transition_points` call already made for the
   noise-included predictive moment -- unconditionally, not gated behind
   a flag (an initial `collect_transition_stat` flag was added, then removed
   once checked against the actual marginal cost: retaining an
   already-materialized point set is small next to evaluating the
   transition at each point in the first place, so gating it added
   complexity for a saving not worth it). This permanently widens
   `XFADS.__call__`'s return from a 3-tuple to `(nature, moment,
   moment_p, transition_stat)` -- a breaking change, accepted, with every
   call site across the codebase updated. `Approx.shrink`'s contract
   shrinks accordingly to `shrink(self, moment, transition_stat, prior)`: no
   more `u`, `c`, `transition_fn`, `mc_size`, or `key` -- it purely
   consumes the already-propagated `transition_stat`, doing only the
   residual + MAP-shrinkage math.
7. An intermediate version of step 6 had `_site_filter`/`nofilt`
   themselves reduce the raw point set to `(mean, cov)` via a
   `core.weighted_moments` helper before passing it along as
   `transition_stat` -- rejected once checked directly against `core.py`'s
   own stated invariant (every algorithm in `core.py` is `Approx`-
   subclass-agnostic): reducing a point set to a weighted mean and
   covariance *presumes* a Gaussian-shaped sufficient statistic. A
   different `Approx` family need not summarize its own transition-noise
   statistic that way at all (e.g. a family whose natural sufficient
   statistics aren't a mean/covariance pair) -- `core.py` baking in that
   assumption, even just for `transition_stat`'s construction, is exactly the
   kind of family-specific knowledge leaking into the supposedly-generic
   recursions that this whole design has otherwise been careful to keep
   out (mirroring, e.g., why `mstep_transition_stat` as a standalone
   `core.py` function was rejected earlier -- see iteration 2 above --
   once it was found to call `approx.unpack`, an MVN-specific method).
   Fixed by having `_site_filter`/`nofilt`/`causal` pass `transition_stat`
   through **unreduced** -- the raw `(zs, weights)` point set, exactly as
   `propagate_transition_points` already produces it, with zero
   assumption about how it should be summarized. The reduction moved
   into `MVN`'s own module (`distributions/mvn.py`'s private
   `_weighted_moments`, called from inside `MVN.shrink` only) -- the
   generic point-set linear algebra didn't change, only which module is
   allowed to assume its output means "mean and covariance." As a
   consequence, no docstring in `base.py`/`core.py` (the family-agnostic
   layer) should name a concrete `Approx` subclass or presume a specific
   sufficient-statistic shape when describing `shrink`/`transition_stat` --
   audited and fixed throughout this section and the ABC docstring.
8. Passing the *raw* point set all the way through `core.py`'s
   `scan`/`vmap` (step 7's fix) keeps `core.py` agnostic, but has a real
   cost: it forces every timestep's stacked carry to hold the full
   `(n_points, state_dim)` point set rather than a much smaller reduced
   summary -- for Monte Carlo propagation with `mc_size` large, this is
   asymptotically bigger than a `(state_dim,)`/`(state_dim, state_dim)`
   mean/covariance pair would be. Fixed by adding a new `Approx` ABC
   method, `transition_stat(self, zs, weights) -> Any`, mirroring
   `transition_points`'s own pattern (`Approx` produces the points to
   propagate; `Approx` also reduces them back) -- `core.py`'s recursions
   call this method polymorphically per (batch, time) pair and stack
   *its* return value across time steps, never assuming or interpreting
   the reduced shape themselves (this stays agnostic for exactly the
   same reason `expected_predictive_moment`'s calls into `approx.
   predictive_moment` already do). Default: identity, `return zs,
   weights` unchanged -- since this method is called **unconditionally**
   for every model (not gated by whether `noise_prior`/`shrink` are
   configured), a non-overriding subclass must behave exactly as if this
   method didn't exist. `MVN.transition_stat` overrides it to call
   `_weighted_moments` (moving that reduction out of `MVN.shrink` and
   into `MVN.transition_stat`, called once per pair during the forward
   pass, not once per pair inside `shrink`) -- `MVN.shrink` goes back to
   directly unpacking `mean_f, cov_f = transition_stat`, since by the time it
   receives `transition_stat` the reduction has already happened, upstream,
   via its own class's `transition_stat` override.
9. The value itself was renamed from `shrink_stat` throughout
   (`XFADS.__call__`'s 4th return value, `Approx.shrink`'s 2nd
   parameter, `core.py`'s internal variable names) -- `shrink_stat` named
   the value after its *one current consumer* (`Approx.shrink`), which is
   narrower than what the value actually is: a general-purpose, per-pair
   statistic that `XFADS`'s forward pass computes once and could, in
   principle, feed any future consumer beyond `Q` estimation (e.g. some
   other diagnostic or M-step that also wants a reduced summary of the
   propagated points), not something intrinsically `shrink`-specific. An
   intermediate candidate, `aux_stat`, was considered and briefly used --
   rejected once checked against what it actually communicates: "aux"
   (auxiliary) says nothing about what the value *is*, only that it's
   "extra," which is less useful than naming it for its actual origin.
   Settled on `transition_stat`, matching `Approx.transition_stat` (the
   *method* that produces it) exactly -- mirroring this same file's own
   established pattern of naming a value after its producing method's
   returned contents (e.g. `zs, weights = approx.transition_points(...)`).
   This keeps the name general-purpose with respect to *consumers*
   (nothing about `transition_stat` presumes `shrink` is the only thing
   that will ever read it) while still being descriptive about
   *provenance* (it is, definitionally, whatever `Approx.transition_stat`
   computed) -- the narrowness complained about was coupling to a
   specific downstream consumer, not to the value's own computational
   origin, so this fully addresses it without trading away clarity for
   vagueness.

**Final**: one method, consuming an already-propagated *and already
reduced* statistic rather than computing or reducing it itself -- the
reduction itself lives in a separate, polymorphic `Approx.
transition_stat` method that `core.py` calls without interpreting its
result, keeping `core.py` agnostic while still avoiding storing the full
raw point set across time steps:

```python
class Approx:
    def transition_stat(self, zs, weights) -> Any:
        """Reduces a propagated, noise-free point set (zs, weights) to
        whatever family-specific statistic this subclass's own shrink
        needs. Called once per (batch, time) pair, unconditionally, by
        core.py's forward pass -- core.py stacks whatever this returns
        across time steps without interpreting it, keeping core.py itself
        agnostic to what a family's transition statistic even means.

        Default: identity, `return zs, weights` unchanged -- a
        non-overriding subclass behaves exactly as if this method didn't
        exist."""
        return zs, weights

    def shrink(self, moment, transition_stat, prior) -> Array:
        """Computes the per-(batch,time) statistic from smoothed moments
        and an already-propagated, already-reduced transition_stat (this
        same class's own transition_stat output), and MAP-shrinks it
        toward `prior`, in one call, returning a free-form array. Does
        its own pair-alignment slicing of `moment` internally -- the
        caller should not need to know this needs shifted, aligned pairs
        at all. Does NOT itself propagate anything through the
        transition, and does NOT itself reduce a raw point set -- both
        already happened upstream, in XFADS's own forward pass (the
        latter via this same class's transition_stat). Mirrors
        Observation.mstep/Gaussian.mstep's shape exactly: the orchestrator
        (XFADS.mstep) calls exactly one method here, never sequencing a
        separate raw-statistic step itself.

        Deliberately opaque about transition_stat's shape/meaning and prior's
        structure -- defined entirely by the subclass (via its own
        transition_stat pairing); not every Approx family need define
        this meaningfully. prior is external (owned by XFADS.noise_prior,
        not by Approx -- see below).

        Default: not supported, raises NotImplementedError. Callers
        (XFADS.mstep) only reach this when a prior has been explicitly
        configured (opt-in), so a loud failure is preferable to silently
        returning something the wrong shape."""
        raise NotImplementedError(f"{type(self).__name__} does not implement shrink")


class MVN(Approx):
    def transition_stat(self, zs, weights):
        """Reduces to this Gaussian family's own sufficient statistic --
        a weighted mean/covariance pair -- via _weighted_moments."""
        return _weighted_moments(zs, weights)

    def shrink(self, moment, transition_stat, prior):
        """Slices moment_t = moment[:, 1:, :] (aligned with transition_stat's
        own per-pair (mean_f, cov_f), already propagated *and* reduced,
        via this class's own transition_stat, for source steps t-1),
        then per (batch,time) pair:

            m', P' = self.unpack(moment_t)
            mean_f, cov_f = transition_stat  # already reduced, upstream
            r = m' - mean_f
            raw_stat = outer(r, r) + P' + cov_f

        (v1 approximation: no cross-covariance term -- see below), then
        MAP-shrinks the mean of that statistic toward prior = (value,
        prior_dof): (n * mean(raw_stat) + prior_dof * value) / (n +
        prior_dof), n = total pair count. Re-encodes via
        self.canon_to_free(MVNParam(loc=zeros, chol=cholesky(shrunk))) --
        NOT free_from_kw, which only accepts a diagonal/scalar scale for
        initialization and can't round-trip an arbitrary full shrunk
        covariance. loc is preserved as zero, matching how
        MVN.predictive_moment already discards noise_free's loc component
        entirely. Not required to be gradient-free on its own terms -- see
        Cadence/gradient-freedom discussion below."""
        ...


class XFADS:
    # noise_prior: Any = eqx.field(static=True) -- class-level field,
    # alongside noise_free. Set once in __init__ from conf.noise_prior/
    # conf.noise_prior_dof (both optional, default None -> self.noise_prior
    # = None, the structural no-shrinkage default, same None/not-None
    # pattern as self.beta_encoder). Static, and storing plain Python
    # values rather than jnp arrays, is load-bearing, not stylistic: a
    # non-static field holding jnp.asarray(...) values would be a genuine
    # float-array pytree leaf, which train()'s own eqx.filter(model, eqx.
    # is_inexact_array) would silently pick up as trainable with no
    # freeze_paths entry protecting it (caught after the fact by checking
    # this against train()'s actual filtering code, not assumed).

    def mstep(self, t, y, u, c, *, key) -> "XFADS":
        """Composes both components' updates in one call -- the single,
        discoverable entry point (model = model.mstep(data, key=...))
        instead of separate driver functions per parameter. The only
        place that knows noise_free is an attribute name -- self.approx.
        shrink knows it's computing/shrinking a
        transition-noise statistic, but not that the result gets stored
        as noise_free. One combined call to self.approx, mirroring
        GLM.mstep calling exactly one method, not two separately-sequenced
        steps.

        No `prior` call argument at all -- least-knowledge: this method's
        caller (and, eventually, train()) should not need to know a prior
        exists. Instead, prior/prior_dof are read once at construction
        into self.noise_prior (not on Approx, since Approx (self.approx)
        is a stateless property freshly reconstructed from
        conf.approx_kwargs on every access -- not a natural home for a
        specific model's chosen hyperparameter value). Skips the
        noise_free update entirely if self.noise_prior is None."""
        approx = self.approx
        _, moment, _, transition_stat = self(t, y, u, c, key=key)
        new_observation = self.observation.mstep(t, moment, y, approx)
        model = eqx.tree_at(lambda m: m.observation, self, new_observation)

        if self.noise_prior is None:
            return model

        new_noise_free = approx.shrink(moment, transition_stat, self.noise_prior)
        return eqx.tree_at(lambda m: m.noise_free, model, new_noise_free)
```

`transition_stat` is `self(...)`'s own 4th return value, always computed --
reusing the forward pass's own propagation rather than having `shrink`
repeat it (see revision 6 above).

**Resolved implementation concern from the earlier design**:
`Approx.mstep_frozen_paths` is still intentionally absent because it would
require `Approx` to know the external attribute name `noise_free`. This is
not needed: `XFADS` owns both `noise_prior` and `noise_free`, and
`trainer.py` derives the `noise_free` freeze path at the model/trainer
level whenever a Q prior is configured. Thus `Approx` remains unaware of
storage details while `train()` can safely compose the R and Q M-steps.

**Deliberate contract difference from `Observation.mstep`, stated
explicitly so it isn't mistaken for an inconsistency**: `Observation.mstep`
returns a *new `Observation`* because `R` is stored inside `Gaussian`/
`GLM`'s own pytree state. `Q` is not stored inside `Approx` at all --
`Approx` (`MVN`) holds no trainable state; `noise_free` lives directly on
`XFADS` and is only interpreted via `approx.canon_to_moment(approx.
free_to_canon(...))` at the point of use. So `Approx.shrink` returns the
raw updated array, and `XFADS.mstep` is what writes it back onto
`model.noise_free` -- consistent with `Observation.mstep`'s underlying
reasoning (the component that owns the storage returns/produces the
updated value in its own storage's shape), not an inconsistency.

**Where `prior` actually lives**: `XFADS`'s own `conf` (`conf.noise_prior`/
`conf.noise_prior_dof`), not `Approx`'s constructor and not `XFADS.mstep`'s
call signature. An earlier pass of this design put it on `MVN`'s
constructor (`MVN(..., noise_prior=..., noise_prior_dof=...)`), matching
how `use_sigma_points` already works -- rejected once it was checked
against `XFADS.approx`'s actual implementation: `approx` is a computed
property, freshly reconstructed from `conf.approx_kwargs` on *every*
access, not persistent stored state, so it's not a natural home for a
specific model's chosen hyperparameter (as opposed to `use_sigma_points`,
which genuinely is exponential-family-structural config, not a
model-specific numeric choice). It also isn't where the already-tracked
`dyn_conf.state_noise` relocation fix says this kind of thing belongs (a
top-level `XFADS`-owned config field, not nested in `approx_kwargs`).
`Approx.shrink(moment, transition_stat, prior)` still takes `prior` as a call
argument (opaque, as originally designed) -- `XFADS.mstep` is what
supplies it, as `self.noise_prior`.

**`self.noise_prior` is read from `conf` once, at construction, into a
proper `eqx.field(static=True)`** -- not re-read via `conf.get(...)`
inside `mstep()` on every call (an earlier pass did this; harmless but
needlessly indirect), and *not* stored as `jnp.asarray(...)` values on an
ordinary field. That second detail is load-bearing, not stylistic: caught
by checking against `train()`'s actual parameter-filtering code, not
assumed -- `train()` selects trainable leaves via `eqx.filter(model, eqx.
is_inexact_array)`. A non-static field holding float-array values would be
a genuine trainable leaf, silently receiving gradient updates with no
`freeze_paths` entry protecting it (nothing adds `"noise_prior"` to any
freeze list). Marking it `eqx.field(static=True)` (same as `conf` itself)
and storing plain Python values (via a small recursive helper converting
OmegaConf `ListConfig`/lists to hashable tuples, since static fields
participate in the pytree's aux data/treedef, which must be hashable for
JIT-cache correctness) excludes it from trainability *structurally* --
verified via `eqx.filter(model, eqx.is_inexact_array)` leaving it
untouched, and via two independently constructed models with identical
`noise_prior` config producing equal treedefs (confirming hashability).

`prior` is a **required argument to `Approx.shrink`, no
default value** (deliberately not even `(1.0, 0.1n)`, the `(value,
prior_dof)` pair that worked in the experiments), since tuning this is
explicitly out of scope and those values were picked once, not validated
as good defaults. Baking them in as a default would silently imply
recommendation; callers must supply `prior` explicitly, in whatever form
their concrete `Approx.shrink` expects.

### Deferred, conditional follow-up: drop `noise_free`'s free-form storage once `mstep` is the permanent mechanism

The free-form encoding (`noise_free`, requiring a `free_to_canon`/
`canon_to_moment` round-trip -- a real matrix inversion for the full-rank
layout, not a trivial transform -- on every forward pass) exists
specifically to give an unconstrained gradient optimizer something safe to
optimize. If `Q` is never gradient-optimized once `mstep` is the
*permanent* update mechanism (not just the current experimental one --
`Q` frozen from gradient updates via an explicit `freeze_paths=
["noise_free"]`, only touched by periodic `XFADS.mstep` calls), that
round-trip becomes pure overhead paid every forward pass for a capability
nothing uses anymore.

**Not scoped into the current implementation plan** -- it's conditional on
`mstep` proving out as the permanent mechanism (not yet certain), and it
touches more than `Approx.shrink`'s own signature: `noise_free`'s
initialization (`dyn_conf.state_noise` -> `free_from_kw`), every consumer
in `core.py` (`_site_filter`/`nofilt`/`_bismooth`'s `approx.
canon_to_moment(approx.free_to_canon(model.noise_free))`), and
`trainer.noise_schedule`, which explicitly anneals in "the attribute's
natural (constrained) space, converting to free-form only at the end"
(`docs/training.md`) -- a representation change needs that updated too, or
it breaks silently. **Nothing about the current design blocks this later**:
`MVN.shrink`'s internal computation already happens in canon/native space
(the statistic and shrinkage are covariance-shaped quantities, not
free-form ones) -- the free-form conversion is only the last line before
returning, a one-line change to remove if/when `noise_free`'s storage
format changes.

### Cadence: validated at round-based, shipped at `R`'s existing cadence (a deliberate, flagged gap)

The validated experiments use a **round-based cadence**: `N` epochs of
ordinary training with `Q` held fixed (via `freeze_paths=["noise_free"]`,
so it still scales `f`'s gradient but isn't itself gradient-updated),
then one M-step-shrinkage update, repeated for a fixed number of rounds
(`N` was 8-20 in the benchmarks).

`train()`-integration has since landed (see Steps toward implementation,
item 6) -- `model.mstep(...)`, composing both `R` and `Q`, is now called
at whatever cadence `mstep_mode` gives (`"minibatch"` default, or
`"epoch"`, i.e. every step or every epoch), reusing `R`'s existing
cadence machinery rather than building a dedicated round-based one. This
is **more frequent** than what was validated for `Q` specifically --
an explicit, known gap, not an oversight: the LL/KL independence
argument (see Steps toward implementation) means the mechanism is
mathematically well-defined at any cadence, since each M-step update is
computed from a fixed E-step posterior regardless of how often that
happens -- but whether frequent re-shrinkage from small per-call
statistics behaves the same as the validated infrequent shrinkage from
large ones has not been tested. `conf.noise_prior`/`conf.noise_prior_dof`
remain opt-in (`None` by default), so this only affects models that
explicitly configure them.

A genuine round-based cadence (a dedicated "every `N` epochs" knob,
distinct from the existing minibatch/epoch binary) remains unimplemented
and would be needed to reproduce the *exact* cadence validated in the
benchmarks, rather than the coarser approximation `mstep_mode` currently
gives.

### The cross-covariance omission — untested in isolation, but present in every validated result

The classical closed-form M-step for `Q` (Shumway & Stoffer 1982, linear
case) needs the *smoothed cross-covariance* `Cov(z_t, z_{t-1})`. **This
repo's smoothing algorithm does not currently expose it** (`core.py`'s
`smooth()` returns only marginal per-timestep moments); deriving it is
genuinely new, nontrivial, XFADS-specific machinery.

**v1 (used in all experiments above): omit the cross-covariance term.**
Dropping it means the approximation computes `Var(z_t) + A^2Var(z_{t-1})`
instead of the true (smaller) `Var(z_t) + A^2Var(z_{t-1}) -
2A\,Cov(z_t,z_{t-1})` — systematically overestimating the residual
statistic, with the overestimation growing as `z_t`/`z_{t-1}` become more
correlated (i.e. as the posterior narrows toward determinism) — a
self-regulating counter-pressure against collapse specifically, requiring
no hand-tuned strength parameter. **Not tested in isolation**: every result
in this doc uses this v1 approximation; there is no with/without-
cross-covariance ablation yet, so its specific contribution (as opposed to
the MAP-shrinkage/alternating-EM structure as a whole) is unconfirmed.

### Tracked, related fix (not part of this plan's core scope, but should land alongside it)

`dyn_conf.state_noise` (a `Dynamics`-config field) currently seeds
`model.noise_free`, even though `noise_free`'s parameterization is entirely
owned by `Approx`, not by whichever `Dynamics` subclass is plugged in.
Config-ownership leak: swapping `Dynamics` implementations shouldn't
implicitly carry a noise-init parameter unrelated to the dynamics plugin.
Fix: relocate the init hyperparameter to a top-level `Approx`/`XFADS`-owned
config field (sibling to `conf.state_dim`), documented explicitly (README,
`AGENTS.md`, changelog), not silently bundled into a code diff. Not
blocking for further validation work — the prototype scripts use direct
arguments, bypassing this entirely.

## Required safeguards

- **`Q` should stay "in the loss" during training, never fully decoupled**
  — the single clearest, most consistently validated finding. Full
  decoupling (Approach B in both experiments) underperformed on `f`-
  accuracy in both tests, even when its own `Q` estimate was more accurate.
- **MAP-shrunk periodic updates toward a genuinely informative prior**,
  not a numerical-safety-only floor — reverses the earlier draft's
  conclusion (see Design). `prior=1.0, prior_dof_frac=0.1` validated as *a*
  working choice, not necessarily a good one; not swept.
- **Round/epoch-based cadence**, matching what was actually tested — do
  not assume continuous per-minibatch updates are safe without testing
  that specifically (see Open questions).
- **A hard numerical floor is still worth keeping as an independent,
  narrow guard** (`_MIN_VARIANCE`-style) against literal numerical failure,
  even though the validated mechanism's prior is now informative rather
  than floor-only — the floor and the informative prior serve different
  purposes and aren't mutually exclusive.
- **Monitor the trend**: track `||Q||`, posterior variance, and `f`'s own
  held-out forecast accuracy across rounds — per-dimension divergence (as
  seen in the Lorenz z-coordinate) is worth watching, not necessarily a red
  flag by itself.

## Steps toward implementation

**Steps 1-6 below are done, implemented and merged** (`src/jaxfads/base.py`,
`core.py`, `distributions/mvn.py`, `smoother.py`; tests in
`test_distribution.py`, `test_algorithm.py`, `test_smoother.py`). The
original Step 7 (automatic Q freeze-path derivation and `train()`
integration) is also complete; the implementation now has 133 passing
library tests. The historical numbering is retained below because it
records how the design evolved. Two corrections surfaced during
implementation, worth recording rather than silently fixing:

- `MVN.shrink` returns its result via
  `self.canon_to_free(MVNParam(loc=zeros, chol=cholesky(shrunk)))`,
  **not** `self.free_from_kw(...)` as an early sketch said --
  `free_from_kw` only accepts a diagonal/scalar `scale` for
  initialization and cannot round-trip an arbitrary full shrunk
  covariance matrix. Caught by checking the sketch against the actual
  code before writing it, not after.
- The statistic computation needs `u`/`c` as explicit arguments (aligned
  to the *source* time step of each pair, matching `filter()`'s own
  `u[:-1], c[:-1]` convention) -- `transition_fn` cannot be evaluated
  without them. An early sketch omitted them; not a design change, just
  an incomplete first draft of the signature.
- `self.noise_prior` (where `prior` actually lives -- see Design's "Final
  design" subsection) must be `eqx.field(static=True)` storing plain
  Python values, not an ordinary field holding `jnp.asarray(...)` values --
  the latter would make it a genuine trainable leaf under `train()`'s own
  `eqx.filter(model, eqx.is_inexact_array)`, with no `freeze_paths` entry
  protecting it. Caught by checking against `train()`'s actual filtering
  code, not assumed; verified via `eqx.filter`, and via treedef equality
  across independently constructed models (confirming hashability).
- `mstep_transition_stat`/`shrink` were merged into one
  `Approx.shrink` method (see Design's "Final design"
  subsection, revision 3) -- the two-method split required `XFADS.mstep`
  to pre-slice `moment`/`u`/`c` into aligned pairs and to sequence two
  separate calls itself, both symptoms of knowledge leaking out of the
  method that should have owned it, caught once checked against
  `Observation.mstep`/`Gaussian.mstep`'s actual precedent (one combined
  method, not two the orchestrator must call).

Sequencing mirrored `mstep_gaussian_cov`'s own precedent: ship the core,
correctness-tested mechanism first, then integrate it into `train()` and
validate it on benchmark/example workflows. Those implementation steps
are complete; the remaining work is scientific validation of cadence,
robustness, and generalization, not missing core functionality.

1. **Placement is settled**: one combined `Approx.shrink`
   method (concrete `NotImplementedError` default), and `XFADS.mstep` as
   the composing entry point that alone knows about `noise_free`'s name
   and that this is about transition noise (see Design's "Final design"
   subsection for the full reasoning and all three rejected alternatives).
2. **Implement `MVN.shrink`**: its own pair-alignment
   slicing of the full `moment`/`u`/`c` inputs, the v1 statistic exactly
   as used in the experiments (`r = m' - transition_fn(m)`, `raw_stat =
   outer(r,r) + P' + J@P@J.T`, `J = jax.jacrev(transition_fn)(m)`, no
   cross-covariance term, decoupled from `Approx.transition_points`
   entirely) -- **superseded**: the `J@P@J.T` term was later replaced by
   `Approx.transition_points`-based (UT/MC) point-set propagation, see
   Design's "Final design" subsection, revision 5 -- then the
   MAP-shrinkage blend
   `(n·raw_stat + prior_dof·value)/(n+prior_dof)` for a `prior = (value,
   prior_dof)` pair (only `MVN.shrink` asserts this
   specific structure for `prior` -- see Design), returning the result as
   a free-form array via `self.canon_to_free(...)` (see correction above;
   not `free_from_kw`).
3. **Implement `XFADS.mstep`**: composes `self.observation.mstep(...)`
   and `self.approx.shrink(...)` into one call each, as
   sketched in Design -- this *is* the usable entry point for an
   alternating-EM loop (`for round in range(n_rounds): model = train(model,
   data, conf=freeze_q_conf); model = model.mstep(data, key=...)`,
   `conf.noise_prior`/`conf.noise_prior_dof` set once at construction),
   matching exactly what was validated, without needing deeper
   `train()`-integration first (that's the separate, deferred
   cadence-control work below).
4. **Unit tests**, correctness-focused (mirroring the 3 existing
   `mstep_gaussian_cov` tests' shape, adapted): `MVN.shrink`'s combined
   formula (statistic + shrinkage) matches an independently computed
   reference on a small linear case, for both diag and full layouts,
   exercising its own slicing with multi-batch/multi-timestep input;
   `Approx.shrink`'s base default raises `NotImplementedError` for a
   non-overriding
   `Approx`; `XFADS.mstep` composes both correctly (both the
   `noise_prior=None`-skips-`Q` case and the configured case). Skip
   exhaustive edge-case/negative-path tests per this repo's established
   test-purge philosophy.
5. **Scoped test run + lint** on touched files before considering this
   landed (`tests/test_distribution.py`, `tests/test_algorithm.py` if
   `core.py`/`base.py` are touched, plus the new test file) -- full suite
   only before push, per `AGENTS.md`.
6. **Done, not deferred after all**: automatic freeze-path derivation for
   `Q`, and `train()`-integration at `R`'s existing cadence, both landed.
   `train()`'s `train_step`/`apply_mstep` now call `model.mstep(...)`
   (composing both `R` and, when `conf.noise_prior`/`conf.
   noise_prior_dof` are set, `Q`) instead of `model.observation.mstep(...)`
   alone, at whatever cadence `mstep_mode` already gives (`"minibatch"`
   default, or `"epoch"`) -- reusing `R`'s existing cadence machinery
   rather than building a separate one. `noise_free` is auto-excluded
   from gradient descent whenever `conf.noise_prior` is set, mirroring
   `Observation.mstep_frozen_paths()`'s exact pattern, done entirely at
   the `XFADS`/`trainer.py` level (`XFADS` already owns both
   `noise_prior` and the `noise_free` name, so this needed no `Approx`-
   level `mstep_frozen_paths` at all -- the earlier "casualty" concern
   was about an `Approx`-level equivalent specifically, which still
   doesn't exist and still isn't needed).

   This reverses the earlier claim that automatic cadence needed
   trainer-internal surgery beyond what shipped for `R` -- once
   `XFADS.mstep` existed as a single composing call, wiring it into
   `train_step`/`apply_mstep` in place of `model.observation.mstep(...)`
   was a small, mechanical change, not new surgery. What's still
   genuinely open (see Open questions): the cadence used here
   (`"minibatch"`/`"epoch"`, i.e. every step or every epoch) is *more
   frequent* than what was validated for `Q` specifically (round-based,
   every 8-20 epochs in the benchmarks) -- the LL/KL independence
   argument (below) means the mechanism is mathematically well-defined
   at any cadence, but whether *frequent* re-shrinkage from small
   per-call statistics behaves the same as the validated *infrequent*
   shrinkage from large ones is not yet tested.

   **Why freezing `noise_free` doesn't conflict with anything, resolving
   an over-cautious earlier objection**: a concern was raised that
   gradient descent on `noise_free` and the closed-form shrinkage update
   might "fight" if both applied at the same cadence without freezing.
   They don't need to, and freezing is not a compromise: given a *fixed*
   posterior (this round's E-step), the ELBO's expected-log-likelihood
   term depends only on `ψ`/`R` and its KL term depends only on
   `θ`/`Q` -- maximizing each is an independent, well-defined operation
   on the same fixed posterior, exactly the classical M-step
   decomposition. The only real issue was mechanical (an un-frozen
   gradient step on `noise_free` would be silently overwritten by the
   next shrinkage call, wasting compute) -- solved by freezing, exactly
   mirroring `R`'s own already-shipped, already-relied-upon pattern.
   - The tracked `dyn_conf.state_noise` config-relocation fix.
   - The remaining validation gaps below (multi-seed, more systems,
     ablations, cadence sweep, prior sensitivity, the SNR correction) --
     real, but follow-up work on the shipped mechanism, not prerequisites
     to shipping it, mirroring how `mstep_gaussian_cov`'s own further
     real-data validation happened after the core mechanism landed.

## Validation plan

**Completed** (see Design for full results):
- Known-z clean baseline (`benchmarks/mstep_known_z_baseline.py`): joint
  MLE vs. fully-decoupled vs. alternating-EM comparison, Lorenz, single
  seed.
- Latent-z (`benchmarks/mstep_lorenz_latent.py`): same three-way
  comparison against a real XFADS model with posterior inference, Lorenz,
  single seed.

**Remaining validation work, tracked as Open questions below, is not
blocking the implemented mechanism**: multi-seed replication;
VDP/oscillator-bank coverage; a with/without cross-covariance-term
ablation; a cadence sweep (round-based behavior was validated, while
`train()` also supports minibatch/epoch cadence); a prior/prior_dof_frac
sensitivity check; correcting the SNR mismatch (intended `1`, actually
run at `~0.1`); and a real downstream Lorenz campaign without hand-tuned
annealing. The library implementation itself is covered by the current
full suite (133 tests), and `train()` integration has already landed.

## Open questions

- **Multi-seed replication** — both experiments are single-seed; the
  magnitude (and even direction, in edge cases) of the A/B/C gap needs
  confirming before treating this as settled.
- **Broader system coverage** — only Lorenz has been tested; VDP and the
  oscillator-bank benchmark (both already used elsewhere in this repo) are
  not yet run through either harness script.
- **Whether the cross-covariance omission's specific contribution matters**
  — an explicit with/without ablation hasn't been run; all current results
  include it.
- **Whether continuous per-minibatch cadence is safe** — only round-based
  cadence has been tested; do not assume `R`'s per-minibatch safety
  transfers. No longer purely hypothetical: `train()`'s `mstep_mode`
  default (`"minibatch"`) now applies to `Q` too whenever `conf.
  noise_prior` is configured (see Design's Cadence subsection) -- opt-in,
  but live, not just a documented risk.
- **Sensitivity to `prior`/`prior_dof_frac`** — picked once (`1.0`, `0.1`),
  not swept; how much of the C-beats-A/B result depends on this specific
  choice is unknown.
- **The intended `SNR=Q_true/R=1` starting point was not actually hit** —
  both experiments used `q_true=0.01` with observation noise implying
  `SNR≈0.1`, not `1`. Not corrected yet; worth checking whether results
  hold at the originally-intended SNR.
- Whether the design generalizes cleanly beyond `MVN` once a non-Gaussian
  noise family is attempted.
- Whether/when the deferred `noise_free` free-form-storage simplification
  (see Design) becomes worth doing -- contingent on `mstep` proving out as
  the permanent update mechanism, not yet certain.
- Whether the exogenous k-step-ahead diagnostic (proposed earlier as a
  backstop) is still needed given how well the alternating-EM mechanism
  performed without it in these experiments, or can be dropped.

## Implemented configuration migration: canonical Q scale and epoch-level joint M-step

**Implemented.** The following records the agreed migration and its final
semantics. It makes the Q-prior scale canonical while
retaining an explicit switch for whether Q is currently MAP-updated or
SGD-updated. It is a breaking configuration/checkpoint migration: old
`dyn_conf.state_noise`, `noise_prior`, and `noise_prior_dof` settings are
removed rather than translated.

1. **Introduce one canonical top-level Q-scale option.** Replace
   `dyn_conf.state_noise`, `conf.noise_prior`, and
   `conf.noise_prior_dof` with required top-level `conf.q_scale`, a
   positive process **variance**. Its semantics are

   $$
   Q_{\mathrm{init}} = Q_0 = q_{\mathrm{scale}} I_d,
   \qquad
   \nu_0 = d + 1,
   $$

   where $d = \texttt{state_dim}$. Thus the initialized Q and the center
   of its MAP shrinkage prior always agree. `d + 1` is the implemented
   update's dimension-aware pseudocount/prior weight, not a claim that the
   code implements a literal inverse-Wishart distribution.

2. **Add an explicit Q-M-step switch.** Introduce top-level
   `conf.q_mstep: bool = true`. Initialize `noise_free` from `q_scale` in
   either mode and remove the static `XFADS.noise_prior` field. When
   `q_mstep=true`, `XFADS.mstep` constructs `(q_scale, state_dim + 1)` and
   calls `Approx.shrink(moment, transition_stat, prior)`; `noise_free` is
   auto-frozen from SGD. When `q_mstep=false`, `XFADS.mstep` skips Q
   shrinkage entirely and `noise_free` remains SGD-managed. Consequently,
   `Approx.shrink`'s `NotImplementedError` default is reached only when a
   user enables `q_mstep` for an Approx family that has not implemented
   Q estimation; disabling Q shrink remains a valid path for such a family.

3. **Make the joint `XFADS.mstep` epoch-level by default.** Change the
   trainer's default `mstep_mode` from `"minibatch"` to `"epoch"`.
   After each completed epoch, run one fresh inference pass over the whole
   training set and call one `model.mstep(...)`, updating R and, when
   `q_mstep=true`, Q from the same posterior, data scope, and cadence.
   Preserve the existing stale-update guard so a normal final epoch does
   not duplicate that final full-dataset M-step. Keep `"minibatch"` as an
   explicit experimental override, not the recommended default.

4. **Retain `transition_stat` unchanged.** Epoch cadence does not make it
   redundant. The full-dataset inference pass already propagates transition
   points through $f$ to form predictive moments; `transition_stat` reuses
   those same propagated points for Q estimation. Therefore the M-step does
   not evaluate $f$ a second time inside `shrink`. `core.py` continues to
   call `approx.transition_stat(zs, weights)` polymorphically and only
   stacks/returns its result; `MVN.transition_stat` retains the
   MVN-specific weighted mean/covariance reduction.

5. **Temporarily disable the dedicated Q scheduling path.** Remove
   `trainer.noise_schedule`, its tests, and its documentation rather than
   attempting to support it alongside MAP updates. The generic
   `param_schedule` mechanism remains available for unrelated attributes.
   In the temporary `q_mstep=false` mode, Q is simply SGD-managed without
   the built-in scheduling helper. A later explicit public choice, e.g.
   `q_update_mode: "sgd" | "map"`, can restore scheduled SGD-Q behavior
   with unambiguous ownership semantics; do not add that broader mode API
   in this change.

6. **Apply the breaking config migration everywhere.** Update test
   fixtures, examples, benchmarks, README/config snippets, `AGENTS.md`,
   `docs/training.md`, and design documentation from

   ```yaml
   dyn_conf:
     state_noise: 1.0
   noise_prior: ...
   noise_prior_dof: ...
   ```

   to

   ```yaml
   q_scale: 1.0
   q_mstep: true
   ```

   Do not preserve old config or checkpoint compatibility.

7. **Update the discriminative tests.** Verify that `q_scale=s` initializes
   $Q=sI_d$; the direct Q M-step reference uses center $sI_d$ and
   pseudocount $d+1$; `q_mstep=true` auto-freezes `noise_free` and makes
   the default training path perform one full-dataset joint M-step per
   epoch; and the existing epoch/final-update deduplication behavior remains
   correct. Verify separately that `q_mstep=false` skips `Approx.shrink`,
   leaves `noise_free` SGD-managed, and lets an unsupported Approx train
   without raising. Retain a test that the epoch path produces and consumes
   `transition_stat` rather than repeating transition propagation. Remove
   the Q-specific scheduling-helper tests along with that helper.

8. **Validate in order.** During implementation, run the scoped
   distribution, smoother, trainer, and configuration-bearing tests; before
   committing, run the full CPU suite. Then measure whether epoch cadence is
   stable enough relative to the validated 8--20-epoch round cadence before
   considering a separate `mstep_every_n_epochs` control. This cadence
   question remains an empirical follow-up, not a reason to retain the
   minibatch default.

## Superseded proposal: epoch-local accumulated Q MAP without a second full-data pass

**Status: superseded before implementation.** This proposal correctly
identified that Q should accumulate minibatch statistics and update once per
epoch without a second full-data inference pass. Its proposed asymmetric
cadence—minibatch R replacement updates but epoch-accumulated Q updates—is
now rejected: the expensive operation is obtaining the minibatch posterior
and propagating transition points, while deriving either R or Q's additive
statistic once those quantities exist is comparatively cheap. Maintaining
separate R/Q cadence machinery would add complexity without a commensurate
compute benefit. The unified replacement plan below supersedes this section.

### Rationale and interpretation

The training dataset is replayed every epoch while dynamics, encoders, and
observation parameters change through SGD. Q statistics from a previous
epoch were therefore computed under a different approximate model and must
not be carried forward as new independent evidence. Each epoch is treated as
a new approximate E/M iteration for the current evolving model.

At the same time, resetting the *accumulator* must **not** reset the learned
Q parameter. Let $Q_{e,0}=Q_{e-1,\mathrm{end}}$ be the Q value carried into
epoch $e$. Only the evidence accumulator is reset to the fixed canonical
MAP prior:

$$
n_{e,0}=\nu_0=d+1,
\qquad
S_{e,0}=\nu_0 Q_0,
\qquad
Q_0=q_{\mathrm{scale}}I_d.
$$

For each minibatch $b$ in the epoch, collect its additive transition-noise
sufficient statistic $(n_b,S_b)$, where $n_b=B_b(T_b-1)$ and, for MVN's
current v1 approximation,

$$
S_b=
\sum_{i,t}
\left[
(m_t-m_f)(m_t-m_f)^\mathsf{T}+P_t+P_f
\right].
$$

Accumulate only within the epoch:

$$
n_{e,b+1}=n_{e,b}+n_b,
\qquad
S_{e,b+1}=S_{e,b}+S_b.
$$

Then, after the final minibatch, apply one Q update:

$$
Q_{e,\mathrm{end}}
=
\frac{S_{e,\mathrm{end}}}{n_{e,\mathrm{end}}}.
$$

Thus the previous Q remains the model state used for inference at the start
of the new epoch, but its old *confidence* is discarded. The fixed
$q_{\mathrm{scale}}I_d$ prior remains the explicit MAP regularizer each
epoch. This is not exact posterior Bayes over the whole neural-training
trajectory; it is an epoch-local stochastic/generalized MAP-EM
approximation. It is preferable to indefinite accumulation, which would
repeatedly double-count replayed data and make Q increasingly inert as the
model changes.

### Implementation plan

1. **Restore minibatch R cadence.** Change the default `mstep_mode` back to
   `"minibatch"`; it controls the existing inexpensive observation M-step.
   Keep `"epoch"` as an explicit R-only full-data option. Do not use
   `mstep_mode` to trigger a second full-data Q pass.

2. **Split the trainer-facing operations by statistical scope.** Preserve
   `XFADS.mstep` as a convenient manual full-data operation if useful, but
   give the trainer a model-level path that, after each gradient update,
   performs the existing minibatch R update and returns a Q batch statistic
   from that same forward pass. The trainer must remain `Observation`-
   agnostic and must not import `observations.py`.

3. **Factor the Q family math into additive statistics.** The current
   `Approx.shrink(moment, transition_stat, prior)` combines three jobs:
   derive the family-specific Q statistic, blend the prior, and encode Q.
   Refactor it so the concrete Approx can produce an additive
   transition-noise statistic from `(moment, transition_stat)` and convert
   an accumulated prior-plus-data statistic into `noise_free`. The exact
   method names may be chosen during implementation, but their contract
   must keep `core.py` Approx-agnostic and let the trainer add only
   array-valued statistic pytrees. For MVN, the statistic is full scatter
   matrix plus effective pair count; its initial prior statistic is
   `((d+1) q_scale I_d, d+1)`.

4. **Keep transition-stat reuse.** The minibatch R M-step already needs a
   post-gradient minibatch inference pass. Derive Q's batch statistic from
   that same call's `transition_stat`; do not add another transition
   propagation or another model inference pass. This preserves the value
   reuse established by `Approx.transition_stat`.

5. **Make epoch boundaries explicit in trainer state.** Store the running
   Q statistic in ephemeral trainer state, not in `XFADS` or checkpoints.
   At each epoch start, initialize it from the fixed `(q_scale I_d,d+1)`
   prior; tree-add each batch statistic; at epoch end, update only
   `noise_free` if `q_mstep=true`, then discard the accumulator. With
   `q_mstep=false`, do not create a Q accumulator and leave Q SGD-managed.

6. **Update tests and benchmarks.** Add discriminative coverage that:
   (a) the Q accumulator resets while Q itself persists across epochs;
   (b) accumulated minibatch scatter/count matches an independent
   whole-epoch sum under frozen model parameters; (c) exactly one Q update
   occurs per epoch, while R retains minibatch updates; (d) no second
   full-dataset inference pass is made for Q; and (e) `q_mstep=false`
   remains pure SGD-Q. Re-run the Lorenz latent benchmark and VDP example,
   comparing wall-clock time, Q trace, flow RMSE, and posterior RMSE against
   the current epoch-full-pass and SGD-Q baselines.

7. **Keep further online variants out of scope.** Do not recenter the prior
   at the preceding Q, carry counts indefinitely, introduce within-epoch Q
   updates, or add a forgetting hyperparameter. Those are different
   proximal/online-EM algorithms and require their own ablations.

## Implemented: unified epoch-local accumulated R/Q MAP without a second full-data pass

**Status: implemented and unit-tested.** This supersedes both the former
full-dataset epoch-Q path and the asymmetric proposal above. It uses one
shared epoch-local accumulator and one joint R/Q update while retaining
normal minibatch SGD throughout the epoch.

### Rationale and interpretation

Each minibatch already performs the expensive work: approximate inference for
its latent trajectory and, for transition terms, propagation through the
dynamics. From those outputs, both the observation-noise and transition-noise
sufficient statistics are inexpensive reductions. Accumulating them avoids a
second full-dataset inference pass, so the desired shared cadence has nearly
the same heavy-lifting cost as ordinary minibatch training.

The training dataset is replayed every epoch while dynamics, encoders,
readout, and latent posteriors change through SGD. Statistics from a prior
epoch therefore belong to an older approximate model and must not be carried
forward as new independent evidence. At each epoch boundary, reset the
**accumulators** but retain the learned model parameters, including R and Q.
Thus the next epoch starts from the prior epoch's learned R/Q values for
inference, but accumulates fresh evidence only from its own minibatches.

For minibatch $b$, let the component-owned additive statistics be
$(n_R^{(b)}, S_R^{(b)})$ and $(n_Q^{(b)}, S_Q^{(b)})$. The trainer treats them
as opaque compatible array pytrees and adds them over the epoch:

$$
n_R \leftarrow n_R + n_R^{(b)},
\qquad
S_R \leftarrow S_R + S_R^{(b)},
$$

$$
n_Q \leftarrow n_Q + n_Q^{(b)},
\qquad
S_Q \leftarrow S_Q + S_Q^{(b)}.
$$

At epoch end, update both components once:

$$
R \leftarrow \operatorname{finalize}_R(S_R,n_R),
$$

and, when `q_mstep=true`,

$$
Q \leftarrow
\frac{S_Q+(d+1)q_{\mathrm{scale}}I_d}
{n_Q+d+1}.
$$

Then discard both accumulators. The Q prior stays fixed at
$q_{\mathrm{scale}}I_d$ with pseudocount $d+1$; do **not** recenter it at the
previous Q, carry confidence indefinitely, or introduce forgetting. The
previous Q is retained only as the model parameter used during the next
epoch's inference.

This is an epoch-local stochastic/generalized MAP-EM approximation, not an
exact final-model batch M-step: each minibatch statistic is evaluated under
that minibatch's **pre-SGD** model parameters, and those parameters drift
through the epoch. That convention is deliberate: it lets the trainer reuse
the exact forward inference already required to evaluate the ELBO, rather
than perform any post-SGD or epoch-end inference pass. The approximation
applies symmetrically to R and Q and is preferable to either repeatedly
replacing parameters with independent minibatch estimates or paying for a
second full pass each epoch.

### Implementation plan

1. **Use an explicit XFADS-owned statistic lifecycle.** The private,
   trainer-facing methods are

   ```python
   batch_stat = model._collect_minibatch_stat(
       t, y, posterior_moment, transition_stat
   )
   epoch_stat = model._accumulate_minibatch_stat(epoch_stat, batch_stat)
   model = model._apply_mstep_stat(epoch_stat)
   ```

   `_collect_minibatch_stat` returns one additive **minibatch delta**; it
   never stores history or mutates a collection. The trainer owns exactly one
   ephemeral `epoch_stat` accumulator and resets it at each epoch boundary.
   `_accumulate_minibatch_stat` is model-owned even though its present
   operation is tree addition: it defines the additive-statistic contract at
   the model/component boundary instead of teaching the trainer which
   component leaves exist. `_apply_mstep_stat` delegates each final update to
   the relevant concrete component and writes any top-level model-owned
   result, such as `noise_free`, back into `XFADS`.

   Rename the current public full-data convenience method to
   `XFADS.mstep_from_data(t, y, u, c, *, key)`. It explicitly performs
   `data -> inference -> minibatch statistic -> _apply_mstep_stat` once over
   supplied data. Do **not** overload `mstep` between data and statistic
   inputs; their domains differ. The normal trainer never calls
   `mstep_from_data`.

2. **Use one symmetric component statistic lifecycle.** `Observation` and
   `Approx` each provide the *same* methods called by XFADS:

   ```python
   component.collect_minibatch_stat(...)
   component.accumulate_minibatch_stat(total, delta)
   component.mstep(epoch_stat, *, prior=None)
   ```

   The first returns one fixed-shape additive delta; the second combines
   same-component deltas; the third finalizes one epoch accumulator. The
   common method name is intentional: both components perform an M-step from
   their own accumulated statistic. Their return types differ naturally:
   `Observation.mstep(...) -> Observation`, while
   `Approx.mstep(...) -> noise_free`; XFADS is the only layer that knows
   where those results are stored. `prior` is ignored by observation
   components and supplied by XFADS only to the Approx branch.

   Rename the current `Approx.shrink` to `Approx.mstep` as part of this
   cleanup. Calling it `mstep` does **not** make Approx aware of the
   `noise_free` attribute: it returns an opaque free-form update, while XFADS
   writes that value into its own storage. A no-op component returns/retains
   `None`. The trainer remains Observation/Approx-family agnostic: it calls
   only XFADS private lifecycle methods and never imports `observations.py`
   or assumes mean/covariance layouts. Gaussian R owns masked residual sums
   and per-feature valid counts; MVN Q owns transition-scatter sums and pair
   counts.

3. **Reuse the existing pre-SGD loss forward pass—never run inference
   post-SGD.** Extend the current `batch_loss` path to return the component
   statistics as auxiliary output from the exact `model(...)` call that
   already produces posterior moments, predictive moments, and
   `transition_stat` for ELBO evaluation. Use
   `eqx.filter_value_and_grad(..., has_aux=True)` so autodiff differentiates
   only the scalar loss while the trainer receives those already-materialized
   statistics after the gradient calculation. Apply the SGD update normally,
   then pass the returned pre-update delta to
   `_accumulate_minibatch_stat`. `transition_stat` therefore avoids a second
   dynamics propagation for Q; deriving R's residual scatter from the same
   posterior is similarly only a reduction, not new inference.

4. **Keep trainer state ephemeral and reset it at each epoch.** Store the
   single R/Q `epoch_stat` only in `_run_training_loop`, never in `XFADS`,
   optimizer state, or checkpoints. Begin from the model-owned zero statistic
   (not from a prior). Apply Q's fixed prior `((d+1)q_scale I_d, d+1)` exactly
   once inside its component finalization, after all minibatch evidence has
   been accumulated. At the epoch boundary call `_apply_mstep_stat`, discard
   `epoch_stat`, and begin the next epoch with fresh evidence but the updated
   model parameters. A partial interrupted epoch is intentionally discarded.

5. **Preserve Q ownership modes.** With `q_mstep=true`, `noise_free` remains
   auto-frozen from SGD and is updated only by the epoch-final accumulated
   MAP step. With `q_mstep=false`, XFADS returns no Q delta, performs no Q
   finalization, and leaves Q entirely SGD-managed. R remains M-step-owned
   under both modes.

6. **Validation.** Unit tests cover public scalar `batch_loss`, frozen-model
   accumulated-statistic finalization against `mstep_from_data`, exactly one
   `_apply_mstep_stat` call per epoch with no automatic
   `mstep_from_data` call, callback visibility of finalized R, Q ownership
   modes, and full suite regression coverage. Add direct tests that one
   minibatch delta is returned by `_collect_minibatch_stat`, accumulation
   stores one reduced epoch statistic rather than a list, and the Q prior is
   absent from the accumulator but applied exactly once at finalization. A
   small Lorenz smoke benchmark completes the accumulated path without a
   full-data M-step. The remaining empirical work is a full VDP rerun and
   larger benchmark comparison against the former full-data epoch path and
   SGD-Q baseline, reporting wall-clock time, R/Q trajectories, flow RMSE,
   and posterior RMSE.

7. **Keep alternative cadences out of scope.** Do not retain separate R/Q
   cadences, direct replacement-style minibatch M-steps, within-epoch Q
   updates, indefinite confidence accumulation, prior recentering, or
   forgetting parameters unless a later ablation establishes a need.

## Dependencies

- `transition_points.md`: **done** — shipped and `MVN` now defaults to
  `use_sigma_points=True`, closing Problem 1 by default. This plan's own
  M-step statistic is explicitly built on its point-propagation mechanism
  through `transition_stat`, while keeping `core.py` family-agnostic.
- The former `dyn_conf.state_noise` config-ownership leak is resolved by
  the planned top-level `q_scale` migration above.
- Deriving the exact lag-one/cross-covariance term (if a future ablation
  shows the v1 approximation insufficient) is new, XFADS-specific
  machinery, not a dependency on anything already planned elsewhere.

## Future generalization (noted, out of scope here)

The top-level XFADS coordination and component M-step vocabulary converge
under the accumulated-statistic lifecycle above: `Observation.mstep` and
`Approx.mstep` each finalize their own component statistic, while XFADS
writes their different return types into component-owned observation storage
or top-level `noise_free` storage respectively. This does not make Approx
aware of `noise_free`. What remains genuinely future/out-of-scope is a
separate round-based cadence control beyond the fixed epoch boundary,
extending the `transition_stat`/Approx-M-step contract to a genuinely
non-MVN family, and the deferred `noise_free` storage-representation
simplification once the accumulated mechanism is confirmed as permanent.
