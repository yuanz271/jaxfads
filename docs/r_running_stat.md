# Plan: automated R estimator — built into `train()`, not routed through hooks

Status: **implemented and unit-tested** (steps 1-7 of this plan; see
`Observation.mstep`/`mstep_frozen_paths` in `base.py`, `Likelihood.mstep`/
`GLM.mstep` dispatch and `mstep_observation_cov` in `observations.py`, and
`mstep_every_n_epochs` in `trainer.py`). Not yet validated against the
real downstream 27-job campaign (final item in the Validation plan below).
Lower risk than the Q plan — `R`'s residual statistic is anchored to
exogenous data (`y`), so it has no feedback/collapse loop analogous to
`Q`'s. Builds directly on the already-implemented and validated
`mstep_gaussian_cov`/`Gaussian.mstep_stat` (see
[mstep_gaussian_cov](mstep_gaussian_cov.md)), left completely untouched by
this plan.

## Problem

`mstep_gaussian_cov` is a standalone function; every caller currently
hand-rolls the EM alternation loop (freeze `unconstrained_cov`, alternate
`train()`/`mstep_gaussian_cov()` full-dataset passes). This is open
questions #1 and #2 in `mstep_gaussian_cov.md`: no native trainer
integration, and round-cadence tuning is a real, non-trivial burden that
currently lives only in a downstream script.

The underlying *correctness* problem (the Heywood-case exploit on `R`) is
already fixed and shipped. What remains is purely a UX/cadence-convenience
problem — a single periodic, built-in recompute fully solves it. Don't
reach for more machinery than that.

## Design

Two new ABC methods (both no-op by default) and one new `train()` kwarg.
No hooks, no `has_aux`, no `eqx.nn.State`, no minibatch EMA.

### `Observation`/`Likelihood` interface

```
class Observation:
    def mstep(self, t, moment, y, approx) -> "Observation":
        """Closed-form, non-SGD parameter update computed from a full
        forward pass over the given data. Default: no-op, return self
        unchanged."""
        return self
```

No `readout` parameter — mirrors `Observation.eloglik`'s existing
signature, since `readout` is owned internally by `GLM`, not passed in
from outside. `approx` stays explicit, owned by `XFADS`, not by
`Observation`/`GLM`/`Likelihood` at any level, exactly like `eloglik`.

### `Likelihood` stays a `Protocol` — not the framework's plugin surface

**Decision reversed from an earlier draft of this plan** (which proposed
converting `Likelihood` to a `SubclassRegistryMixin`-based ABC, mirroring
`Approx`/`Dynamics`/`Observation`). `Observation` is the actual
framework-level plugin surface a user swaps out; `Likelihood`/`readout`
are `GLM`'s own private internal composition, not a framework-level
concept in their own right — a different `Observation` subclass wouldn't
need a "likelihood"/"readout" split at all. Formally registering
`Likelihood` the same way as the real extension points would incorrectly
promote it to a status it was never meant to have. `GLM.__init__`'s
hardcoded `if likelihood_name == "Poisson": ... elif == "Gaussian": ...`
dispatch is not a gap to fix: it's a small, closed enumeration `GLM` owns
and can extend by editing its own source; a user wanting fundamentally
different observation-model internals should write a new `Observation`
subclass (the real extension point), not extend `GLM`'s private
composition.

This means only the narrower thing is actually needed — an inherited
no-op default so `GLM.mstep` doesn't fail on `Poisson` — which the
original (pre-escalation) design already handled: an internal
`hasattr(self.likelihood, "mstep")` check inside `GLM`, fully
encapsulated, never exposed to the generic trainer:

- `Likelihood.mstep(self, t, moment, y, approx, readout) -> "Likelihood"`
  — `Gaussian`'s own contract; needs `readout` explicitly (like
  `Likelihood.eloglik`), since `Gaussian` doesn't own the readout module.
  `Gaussian.mstep` internally reuses the existing, unchanged `mstep_stat`:
  `jnp.mean(jax.vmap(jax.vmap(partial(self.mstep_stat, approx=approx,
  readout=readout)))(t, moment, y), axis=(0,1))`, then the same
  `cov()`-inversion + `unconstrained_cov` write-back `mstep_gaussian_cov`
  already does. `Poisson` does not, and does not need to, implement
  `mstep` at all — `Likelihood` being a `Protocol` provides no runtime
  default-method behavior, so there is nothing to inherit.
- `Observation.mstep` on `GLM` supplies the no-op behavior itself, via the
  internal `hasattr` check: if `hasattr(self.likelihood, "mstep")`,
  delegate and reassemble (`eqx.tree_at(lambda m: m.likelihood, self,
  self.likelihood.mstep(t, moment, y, approx, self.readout))`, the same
  pattern `GLM.eloglik` already uses); otherwise return `self` unchanged.
  This mirrors exactly how the already-shipped `mstep_gaussian_cov`
  already dispatches, for the same underlying reason.

**No capability flag, no `mstep_stat` at the ABC level, no chunking
support on this path, no minibatch EMA** — all considered and rejected in
this plan's history for not being demonstrated needs:

- A capability flag / ABC-level `mstep_stat` / chunking: solve problems
  that aren't demonstrated at the documented dataset scale (~50-500
  trials); `mstep_gaussian_cov` remains the answer for anyone who
  genuinely needs `batch_size`-chunked scanning.
- Minibatch EMA / momentum (an earlier draft's "cadence 2"): two distinct
  possible justifications, both deferred, not dismissed. (1) *Compute
  savings*: a *self-contained* per-minibatch update (a separate forward
  pass on each minibatch, no reuse of the training step's own pass) costs
  the same total compute over an epoch as one big periodic pass — it
  doesn't save anything on that front; only a `has_aux`-based reuse of the
  training step's own forward pass would (see Escalation path below). (2)
  *Stabilization*: independent of compute, `mstep_every_n_epochs`'s hard
  replacement could plausibly cause a visible transient right after each
  recompute (a discontinuous jump in `R` feeding straight into `eloglik`,
  especially early in training while the encoder/dynamics/readout are
  still changing a lot) — exactly the kind of thing momentum smooths over
  in SGD, and this benefit doesn't require compute savings to be real.
  Deferred pending evidence that the hard-replace jumps actually cause
  visible instability in practice, not because the idea lacks merit.

**`mstep` must never carry gradients.** The driver below runs its forward
pass outside `eqx.filter_value_and_grad(batch_loss_fun)` entirely —
architectural, not conventional, isolation from the optimizer's gradient
tape. The write-back is a plain functional pytree replacement
(`eqx.tree_at`), outside `apply_updates`. Defensively, `Gaussian.mstep`'s
return value should still be wrapped in `jax.lax.stop_gradient` — cheap,
local insurance documenting the invariant.

### Driver: periodic exact recompute (full dataset, hard replace)

```
def mstep_observation_cov(model, data, *, key) -> model:
    t, y, u, c = data
    _natural, moment, _predicted = model(t, y, u, c, key=key)
    new_observation = model.observation.mstep(t, moment, y, model.approx)
    return eqx.tree_at(lambda m: m.observation, model, new_observation)
```

A standalone public function (not a rename or replacement of
`mstep_gaussian_cov`, which stays exactly as-is).

### Wiring: a dedicated `train()` kwarg, not a hook

**Why not `on_epoch_end`, and why a dedicated kwarg is actually the right
call, not a workaround:** `on_epoch_end`/`on_step_end`-style slots are
single-callable — a user who already uses `on_epoch_end` for
checkpointing/early-stopping (`train()`'s own documented canonical use
case) would have to hand-compose it with an `mstep` closure, reintroducing
exactly the manual-wiring boilerplate this plan exists to eliminate.
Categorically, `train()`'s docstring reserves `on_epoch_end` for a
specific class of concerns — "no notion of *validation, checkpointing,
best models, or early stopping*" — external, per-user meta-loop policy.
M-step estimation isn't that: it's a modification to *how the model's
parameters get updated during optimization*, the same category
`regularizer`, `optimizer`, and `param_schedule` are already in, and those
are already dedicated, direct `train()` kwargs, not hooks.

`mstep_every_n_epochs: int | None = None` — when set, at each epoch
boundary where `epoch % mstep_every_n_epochs == 0`, `train()` calls
`mstep_observation_cov(model, train_data, key=<internal subkey>)` itself,
using the `train_data` it already has in scope. No closure needed. `None`
(default) is a complete no-op — zero footprint on existing behavior.
`on_epoch_end` remains completely free for the user's own policy, with
zero overlap.

**This parameter's flexibility (choosing `N`, not just an on/off switch)
is deliberately kept**, unlike the EMA momentum knob that got removed
above — the cost it lets a user amortize (the extra full-dataset forward
pass every time it fires) is a concrete, inherent property of this
design, not a speculative one, so giving users control over how often
they pay it is directly justified by that known cost, not "flexibility
for its own sake."

### Gradient exclusion is automatic, not a manual `freeze_paths` config step

**Revised.** `unconstrained_cov` cannot simply be made static
(`eqx.field(static=True)`): static fields are compile-time-constant,
hashable *auxiliary* data in JAX/equinox's pytree system, not array leaves
subject to numerical replacement during transformations. `mstep`'s own
write-back (`eqx.tree_at(lambda m: m.likelihood.unconstrained_cov, ...,
new_value)`) needs it to remain a genuine *dynamic* leaf to update it at
all — marking it static would very likely break `mstep`'s own update
mechanism, not just block gradients. This is a category mismatch, not a
style preference: static is for values that never change and must be
hashable for JIT caching (like `conf`), not for something updated by a
non-gradient computation. The field must stay an ordinary trainable
array, because `mstep_every_n_epochs` is opt-in — a user who doesn't set
it still gets the original, fully-gradient-trainable `R`.

Given that, exclusion from gradient updates has to be a per-training-run
decision, not a structural one — which is exactly what `conf.freeze_paths`
already is. But requiring the user to *manually* keep `conf.freeze_paths`
in sync with `mstep_every_n_epochs` is unnecessary risk for no benefit:
`train()` already builds an internal `freeze_mask` from `conf.freeze_paths`
(`optax.masked(optax.set_to_zero(), freeze_mask)`); it can just as easily
**derive that mask automatically** from `model.observation.
mstep_frozen_paths()` whenever `mstep_every_n_epochs` is set, folding
those paths into the mask alongside whatever the user's own
`conf.freeze_paths` already specifies. This removes the manual
configuration step entirely (not just the risk of forgetting it) — there
is nothing left for a user to misconfigure, and the earlier "raise if
`freeze_paths` doesn't include the expected path" enforcement step is no
longer needed, since `train()` no longer depends on the user having set it
correctly (or at all).

**`Observation.mstep_frozen_paths()` is still needed, just for a
different purpose than originally stated — automatic mask derivation
instead of manual-config validation:**

```
class Observation:
    def mstep_frozen_paths(self) -> list[str]:
        """Model-relative attribute paths that must be excluded from
        gradient updates whenever mstep-driven updates are active.
        train() folds these into its internal freeze mask automatically;
        no user-facing config entry is required. Default: []. """
        return []
```

`Likelihood.mstep_frozen_paths` gets no inherited default (same reasoning
as `mstep` above — `Likelihood` stays a `Protocol`). `Gaussian.mstep_frozen_paths`
returns `["unconstrained_cov"]`, relative to *itself* (`self.likelihood`,
one level nested inside `GLM`) — matching where the field actually lives.
`GLM.mstep_frozen_paths` supplies the no-op via the same internal
`hasattr` check, **and must add its own nesting prefix when delegating**:
if `hasattr(self.likelihood, "mstep_frozen_paths")`, return
`["likelihood." + p for p in self.likelihood.mstep_frozen_paths()]`;
otherwise `[]`. `train()` then prefixes with `"observation."` on top of
that, so the full path added to the internal freeze mask is
`"observation.likelihood.unconstrained_cov"` — matching exactly the path
already documented in `mstep_gaussian_cov.md`'s manual-alternation usage
pattern (`conf.freeze_paths = ["observation.likelihood.unconstrained_cov"]`),
but derived automatically here rather than requiring the user to write it.
Getting this nesting right matters: without the `GLM`-level prefix, the
wrong leaf would be excluded from gradients — either failing to protect
the real `R` parameter (if the derived path is wrong) or accidentally
freezing an unrelated leaf that happens to share the shorter path name.
`Poisson` implements neither method, and doesn't need to.

## Escalation path (not being built now — only if evidence justifies it)

- **Stabilize `mstep_every_n_epochs` itself with an EMA blend** (the
  low-cost escalation, if hard-replace jumps prove disruptive): reuse the
  already-designed generic pytree-blend trick (blend the whole
  `Observation` old-vs-new; untouched leaves are unaffected since
  `mstep`'s output equals the input everywhere but `unconstrained_cov`) to
  smooth the *periodic, full-dataset* update itself — no `has_aux`, no new
  forward pass, just replacing `mstep_every_n_epochs`'s hard
  `eqx.tree_at` write with a momentum-weighted blend against the previous
  `R`. Only justified if the real-data validation run shows a visible
  training-loss transient right after an `mstep` recompute; not built
  speculatively.
- **Reuse the training step's own forward pass** (the only form of
  per-minibatch update that would actually reduce compute): thread
  `mstep`'s per-instance contribution back via
  `eqx.filter_value_and_grad(..., has_aux=True)` at the `vi.elbo`/
  `batch_loss` level, computed once per step from the *same* forward pass
  already used for the differentiated loss. This eliminates two nested
  layers of duplication at once, not just one: the expensive recurrent
  encoder/dynamics filtering-smoothing scan that `mstep_observation_cov`
  currently re-runs from scratch via its own `model(...)` call, *and* the
  cheap `readout(t, mean_z)`/residual evaluation that `Gaussian.eloglik`
  and `Gaussian.mstep`/`mstep_stat` each separately compute today (the
  latter is a rounding error next to the former, so this escalation is
  only worth it for the recurrent-pass savings, not the residual-math
  savings on its own). Only justified if profiling shows
  `mstep_every_n_epochs`'s periodic extra forward pass materially matters
  *and* eliminating it (not just amortizing it further via a larger `N`)
  is the actual goal — a separate question from the stabilization one
  above, since this one is about compute rather than smoothness. Note
  this fix does not change the optimization dynamics at all (`has_aux`
  differentiates the same scalar loss, produces the same gradients; the
  aux value is a side-channel JAX excludes from the gradient by
  construction) — the real cost is purely to existing code structure
  (`batch_loss`/`vi.elbo`/`eloglik` signatures all need to carry the aux
  value through), not to training behavior.
- **Full `eqx.nn.State`**: only if either of the above still isn't
  enough, or another stateful component independently needs it. `Q` (the
  natural second consumer) has already been recommended *against*
  continuous per-step cadence (see `mstep_dynamics_noise.md`'s
  runaway-loop analysis), so there's no near-term second use case to
  amortize this larger cost against.

## Steps

1. Add `Observation.mstep(t, moment, y, approx) -> Observation` and
   `Observation.mstep_frozen_paths(self) -> list[str]` to the ABC
   (`base.py`), concrete, no-op defaults (`return self` / `return []`).
2. Add the corresponding `Likelihood.mstep(t, moment, y, approx, readout)`/
   `Likelihood.mstep_frozen_paths()` to the `Likelihood` `Protocol`'s
   documented shape (no runtime default — `Protocol` provides none).
   Implement `GLM.mstep`/`GLM.mstep_frozen_paths` with the internal
   `hasattr(self.likelihood, ...)` fallback described above, mirroring
   `GLM.eloglik`'s delegation — note `GLM.mstep_frozen_paths` must prepend
   `"likelihood."` to whatever `self.likelihood.mstep_frozen_paths()`
   returns (see nesting detail above), not just pass it through unchanged.
   No changes needed to `Poisson`.
3. Implement `Gaussian.mstep` (internally reusing its existing, unchanged
   `mstep_stat`: mean over the given data, then the same
   `cov()`-inversion + `unconstrained_cov` write-back `mstep_gaussian_cov`
   already does) and `Gaussian.mstep_frozen_paths` (returns
   `["unconstrained_cov"]`).
4. Add `mstep_observation_cov(model, data, *, key)` as a public
   standalone function. Do **not** touch, rename, or deprecate
   `mstep_gaussian_cov`.
5. Add `mstep_every_n_epochs: int | None = None` as a direct `train()`
   kwarg (alongside `regularizer`/`optimizer`/`param_schedule`, the same
   category of thing). Wire it into the existing epoch-boundary logic,
   calling `mstep_observation_cov` internally with `train_data` already
   in scope — no `on_epoch_end` involvement, no closures.
6. When `mstep_every_n_epochs` is set, have `train()` fold
   `model.observation.mstep_frozen_paths()` (prefixed with
   `"observation."`) into its internal `freeze_mask` automatically,
   alongside whatever `conf.freeze_paths` already specifies — fully
   generic, no hardcoded path string in `trainer.py`, and no user-facing
   config step required for this to be correct.
7. Update `docs/training.md`/`docs/mstep_gaussian_cov.md` to document
   `mstep_every_n_epochs` and `mstep_gaussian_cov` as the fallback for
   chunked/large-dataset needs (no `freeze_paths` documentation burden
   for the automated path, since it's now handled internally).

## Validation plan

- **`Poisson` no-op test**: confirm `GLM.mstep`/`GLM.mstep_frozen_paths`
  correctly fall back to their no-op behavior when `self.likelihood` is a
  `Poisson` (via the `hasattr` check), without `Poisson` implementing
  either method.
- **Consistency test**: `Gaussian.mstep`, accessed via the ABC-declared
  method, produces the same result as calling `mstep_gaussian_cov` on the
  same dataset (same math, different orchestration).
- **No-op test**: `mstep_observation_cov` on a `Poisson`-observation model
  runs its forward pass but returns the model unchanged.
- **Automatic gradient-exclusion test**: confirm `train()` correctly
  excludes `observation.likelihood.unconstrained_cov` from gradient
  updates whenever `mstep_every_n_epochs` is set, *without* the user
  setting `conf.freeze_paths` at all, and that gradient descent no longer
  drifts `R` between `mstep` corrections as a result.
- **User `freeze_paths` non-interference test**: confirm a user's own
  unrelated `conf.freeze_paths` entries (for other parameters) still work
  correctly alongside the automatically-derived `mstep_frozen_paths()`
  entries — the two sources of frozen paths compose, neither overwrites
  the other.
- **No-flag regression test**: `train()`'s existing behavior (loss curves,
  `on_epoch_end`/other kwargs, final `R` under gradient descent) is
  completely unaffected when `mstep_every_n_epochs` is left at its
  default (`None`).
- **`on_epoch_end` non-interference test**: a user-supplied `on_epoch_end`
  (e.g. checkpointing/early-stopping) works unmodified and independently
  of whether `mstep_every_n_epochs` is also set — no composition
  required.
- **Cadence test**: confirm `mstep_every_n_epochs=1` vs. a larger value
  fires at the expected epoch boundaries and converges to the same final
  `R` (just at different wall-clock cost), consistent with the existing
  manual alternation loop.
- **Real-data comparison**: re-run (or spot-check) the same downstream
  27-job-style campaign referenced in `mstep_gaussian_cov.md` using
  `mstep_every_n_epochs`; compare final losses / `cov_min` against the
  already-validated results.

## Open questions

- Default `mstep_every_n_epochs` (`None`, i.e. off by default, is the
  safe choice absent evidence for a different default).
- Whether real-data validation surfaces a visible training-loss transient
  right after an `mstep_observation_cov` recompute — the trigger for the
  EMA-blend stabilization escalation (a smoothness concern, independent
  of compute).
- Whether, separately, evidence emerges that the periodic-recompute cost
  itself matters enough to justify the `has_aux` escalation (a compute
  concern, independent of smoothness).

## Dependencies

None. Independent of `transition_points.md` and `mstep_dynamics_noise.md`;
lowest cost and risk of the three plans; can be implemented immediately.

## Future generalization (noted, out of scope here)

The `mstep` method introduced here for `Observation`/`Likelihood` is a
candidate for a component-uniform contract across every pluggable piece of
the generative model (`Approx`, and possibly `Dynamics` for subclasses
whose map is linear-in-parameters) — the same generalization
`eloglik`/`predictive_moment`/`initialize` already have. If pursued,
`mstep_dynamics_noise.md`'s `mstep_transition_stat`/`mstep_noise_shrink`
would naturally converge to this same `mstep` vocabulary — but `Q`'s
continuous per-step update should **not** be adopted by default given its
runaway-collapse risk (see that doc); only `R` has been argued safe for
continuous cadence, and even for `R` a self-contained per-minibatch
version was rejected above as not actually saving compute. Deliberately
out of scope for this plan; revisit once `R`'s `mstep` has shipped.
