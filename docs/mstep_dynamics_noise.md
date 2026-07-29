# Plan: M-step for transition/process noise (`mstep` on the noise-owning component)

Status: **empirically validated on synthetic data (known-z and latent-z
Lorenz), not yet implemented in the library.** Two independent experiments
(`benchmarks/mstep_known_z_baseline.py`, `benchmarks/mstep_lorenz_latent.py`)
both support the same mechanism: keep `Q` inside the training loss (never
fully decoupled), update it periodically via a **MAP-shrunk M-step toward a
genuinely informative prior** rather than either free gradient descent or
a numerical-safety-only floor. This reverses an earlier draft of this plan
(see Design) that concluded the prior should be non-informative. Remaining
work before implementation: multi-seed replication (only single-seed runs
so far), broader system coverage (only Lorenz tested; VDP/oscillator-bank
not yet run), and the actual `train()` integration.

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

**Final**: one method, taking the *full*, un-sliced `moment`/`u`/`c` and
doing everything internally:

```python
class Approx:
    def shrink(self, moment, u, c, transition_fn, prior, *, key) -> Array:
        """Computes the per-(batch,time) statistic from smoothed moments
        and MAP-shrinks it toward `prior`, in one call, returning a
        free-form array. Does its own pair-alignment slicing internally --
        the caller should not need to know this needs shifted, aligned
        pairs at all. Mirrors Observation.mstep/Gaussian.mstep's shape
        exactly: the orchestrator (XFADS.mstep) calls exactly one method
        here, never sequencing a separate raw-statistic step itself.

        Deliberately opaque about raw_stat's shape/meaning and prior's
        structure -- defined entirely by the subclass; not every Approx
        family need define this meaningfully. transition_fn is external
        (dynamics are not an Approx concept, same pattern Observation.
        mstep already uses for approx); prior is also external (owned by
        XFADS.noise_prior, not by Approx -- see below).

        Default: not supported, raises NotImplementedError. Callers
        (XFADS.mstep) only reach this when a prior has been explicitly
        configured (opt-in), so a loud failure is preferable to silently
        returning something the wrong shape."""
        raise NotImplementedError(f"{type(self).__name__} does not implement shrink")


class MVN(Approx):
    def shrink(self, moment, u, c, transition_fn, prior, *, key):
        """Slices moment_tm1 = moment[:, :-1, :], moment_t = moment[:, 1:, :],
        u_tm1 = u[:, :-1, :], c_tm1 = c[:, :-1, :] (control/covariate at
        the *source* time step of each pair, matching core.filter()'s own
        u[:-1], c[:-1] convention), then per (batch,time) pair:

            r = m' - transition_fn(m)
            J = jacrev(transition_fn)(m)
            raw_stat = outer(r, r) + P' + J @ P @ J.T

        (v1 approximation: no cross-covariance term -- see below), then
        MAP-shrinks the mean of that statistic toward prior = (value,
        prior_dof): (n * mean(raw_stat) + prior_dof * value) / (n +
        prior_dof), n = total pair count. Re-encodes via
        self.canon_to_free(MVNParam(loc=zeros, chol=cholesky(shrunk))) --
        NOT free_from_kw, which only accepts a diagonal/scalar scale for
        initialization and can't round-trip an arbitrary full shrunk
        covariance. loc is preserved as zero, matching how
        MVN.predictive_moment already discards noise_free's loc component
        entirely. Never carries gradients: wraps its result in
        jax.lax.stop_gradient defensively, matching Gaussian.mstep's same
        convention."""
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
        _, moment, _ = self(t, y, u, c, key=key)
        new_observation = self.observation.mstep(t, moment, y, approx)
        model = eqx.tree_at(lambda m: m.observation, self, new_observation)

        if self.noise_prior is None:
            return model

        new_noise_free = approx.shrink(
            moment, u, c, self.transition, self.noise_prior, key=key,
        )
        return eqx.tree_at(lambda m: m.noise_free, model, new_noise_free)
```

**Casualty carried forward from the earlier design, unaffected by this
merge**: `Approx.mstep_frozen_paths` (declaring `["noise_free"]`) still
doesn't survive cleanly -- it would require `Approx` to know that external
attribute name. Since automatic `train()`-integration/cadence-control is
already deferred (see Steps toward implementation), auto-derived
freeze-paths for `Q` are deferred alongside it, not solved with a
workaround now. Callers pass `freeze_paths=["noise_free"]` to `train()`
explicitly for now, exactly as the validated prototype scripts already do.

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
`Approx.shrink(moment, u, c, transition_fn, prior, *, key)` still takes
`prior` as a call argument (opaque, as originally designed) --
`XFADS.mstep` is what supplies it, as `self.noise_prior`.

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
`test_distribution.py`, `test_algorithm.py`, `test_smoother.py`). Step 7
remains deferred, as planned. Two corrections surfaced during
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
correctness-tested mechanism as a standalone utility first;
train()-integration and further real-world validation follow as separate,
later steps, not blockers.

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
   entirely), then the MAP-shrinkage blend
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

**Still pending, tracked as Open questions below, not blocking initial
implementation**: unit tests for the actual library implementation (Step 4
above, distinct from the ad hoc prototype scripts); multi-seed replication;
VDP/oscillator-bank coverage; a with/without cross-covariance-term
ablation; a cadence sweep (round-based only tested so far); a
prior/prior_dof_frac sensitivity check; correcting the SNR mismatch
(intended `1`, actually run at `~0.1`); a real downstream Lorenz campaign
without hand-tuned annealing, once `train()` integration exists.

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

## Dependencies

- `transition_points.md`: **done** — shipped and `MVN` now defaults to
  `use_sigma_points=True`, closing Problem 1 by default. This plan's own
  M-step statistic is explicitly *not* built on top of it (see Design).
- The tracked `dyn_conf.state_noise` config-relocation fix: real, but
  explicitly deferred.
- Deriving the exact lag-one/cross-covariance term (if a future ablation
  shows the v1 approximation insufficient) is new, XFADS-specific
  machinery, not a dependency on anything already planned elsewhere.

## Future generalization (noted, out of scope here)

`XFADS.mstep` converging onto `Observation`'s `mstep` vocabulary at the
top-level entry point (see [mstep_gaussian_cov](mstep_gaussian_cov.md)) is
no longer future work -- it's the finalized design (see Design's "Final
design" subsection). Deliberately **not** converged, and not planned to
be: `Approx.shrink` keeps its own name rather than being called
`Approx.mstep`, and there is no `Approx.mstep_frozen_paths` -- both were
rejected mid-design specifically to keep `Approx` unaware of
`noise_free`'s existence (see Design's "casualty" note). What remains
genuinely future/out-of-scope: automatic cadence control inside `train()`
and auto-derived freeze-paths for `Q` (Steps toward implementation, item
6), extending the `shrink` contract to non-`MVN` `Approx` families, and
the deferred `noise_free` storage-representation simplification (see
Design) once `mstep` is confirmed as the permanent mechanism.
