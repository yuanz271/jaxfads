# Plan: MAP-shrunk M-step for transition/process noise (`mstep_dynamics_noise`)

Status: proposed, not implemented. **Higher risk than the R estimator** — read
the runaway-loop analysis below before implementing; do not skip the required
safeguards.

See also: [mstep_gaussian_cov](mstep_gaussian_cov.md), [transition_points](transition_points.md).

## Problem

`model.noise_free` (process noise `Q`) is always gradient-trained jointly
with the rest of the model (dynamics, encoder, readout), via
`approx.canon_to_moment(approx.free_to_canon(model.noise_free))` in
`core.py`. This is subject to two distinct pathologies:

1. **Rank-deficiency exploit** (same family as the observation-`R` Heywood
   case): when `mc_size < state_dim`, the MC "spread" term in the predictive
   covariance is rank-deficient, giving gradient descent a route to shrink
   `Q` in the directions that term happens to cover. Mitigated by
   `transition_points.md` (deterministic, full-rank-by-construction sigma
   points).
2. **Posterior collapse (a distinct, more serious failure mode)**: unlike
   `R`, which is checked against *fixed, exogenous* data `y`, the KL term
   comparing `q(z_t)` to the `Q`-derived predictive prior has **both sides
   jointly trainable**. Gradient descent can shrink `q(z_t)`'s posterior
   width and `Q` together, driving KL toward zero without ever consulting
   real data — especially in low-SNR regimes, where the per-step
   observation evidence pulling the posterior mean is weak. `Q = 0`
   (deterministic dynamics) is a **self-consistent fixed point** of this
   process, not just a numerical floor artifact.

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
in the classical linear-Gaussian SSM/EM literature (EM for Kalman-filter-type
models can converge to near-deterministic dynamics under uninformative
observations), not a hypothetical specific to this repo.

**Consequence for cadence**: unlike `R`, do not default to a fast,
continuous per-minibatch running-stat/EMA update for `Q` — that would
tighten the feedback loop rather than damp it. Keep a slower, deliberate
cadence (block rounds, and/or an explicit annealing schedule) by default.

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

Default `prior`/`prior_dof`: reuse the noise-init hyperparameter (currently
`dyn_conf.state_noise`, to be relocated per the tracked config-ownership fix
below) as the shrinkage anchor unless the caller overrides it.

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

- **Q-annealing / KL-beta warmup must run first.** Start `Q` large, anneal
  down slowly (existing `beta` warmup in `vi.elbo` is the direct analogue).
  Only switch to (or blend in) the M-step estimator after a warmup period or
  an explicit gating diagnostic — never from the start of training.
- **MAP shrinkage toward a floor is mandatory, not optional** — never use
  the raw statistic as a plain MLE, precisely because the raw MLE has a
  reachable, self-consistent `Q = 0` fixed point.
- **Prefer an exogenous diagnostic to gate/monitor the update**: a k-step-
  ahead forecast, decoded through the observation model and checked against
  held-out `y`, is the only way to anchor `Q` to something outside the
  self-referential loop (analogous to what `y` already provides for `R`
  directly).
- **Slow/round cadence, not per-minibatch EMA** — do not apply the
  BatchNorm-style continuous-update pattern planned for `R` to `Q`.
- **Monitor the trend**: track `||Q||` and posterior variance across
  rounds/epochs; a monotonically shrinking `Q` with no corresponding
  improvement in held-out ELBO/eloglik is the collapse signature to watch
  for, distinct from genuine convergence.

## Steps

1. **Build a synthetic validation harness first** — simulate a known
   (linear or mildly nonlinear) SSM with a known true `Q`, at controlled
   SNR levels including a deliberately low-SNR case. This is essential given
   the identified risk that the estimator could reinforce collapse rather
   than correct it; internal consistency checks alone are not sufficient
   evidence of correctness here (unlike the `R` case).
2. Implement `mstep_transition_stat` / `mstep_noise_shrink` for `MVN`,
   reusing `Approx.transition_points` for the spread term.
3. Implement the `mstep_dynamics_noise` driver, prior/prior_dof plumbing,
   and the tracked config-relocation fix.
4. Run the synthetic harness across SNR levels: does the estimator recover
   the true `Q`, or collapse without annealing? Re-run with annealing
   enabled — does it recover then? Report both outcomes explicitly.
5. Only after the synthetic harness passes, validate on a real downstream
   dataset with alternation + annealing (mirroring the `mstep_gaussian_cov`
   validation precedent).
6. Document results, including any observed collapse cases at low SNR and
   which safeguard did or did not prevent them.

## Validation plan

- Unit tests for `mstep_transition_stat`/`mstep_noise_shrink` matching an
  independently computed statistic (mirroring the 3 existing
  `mstep_gaussian_cov` tests: exact-match, `NotImplementedError` for
  non-participating `Approx`, duck-typed dispatch).
- **Synthetic-SSM recovery test across SNR levels** — the key discriminative
  test, since it's the only way to check against ground truth rather than
  internal self-consistency.
- Real-data campaign re-run only after the synthetic test passes.

## Open questions

- Exact default `prior`/`prior_dof` values and how sensitive results are to
  them.
- Whether the shrink-combinator design generalizes cleanly beyond `MVN`
  once a non-Gaussian noise family is attempted.
- How to define/measure "low SNR" precisely for the stress test (e.g. ratio
  of observation Fisher information to `Q^{-1}` at a given time step).
- Whether the exogenous k-step-ahead diagnostic should be a hard gate
  (blocking the update) or a soft monitor (logged only).

## Dependencies

- Benefits from `transition_points.md` (shared spread-term computation);
  should land after or alongside it.
- Depends on the tracked `dyn_conf.state_noise` config-relocation fix.

## Future generalization (noted, out of scope here)

`Observation.mstep`/`Observation.mstep_frozen_paths` have now shipped (see
[mstep_gaussian_cov](mstep_gaussian_cov.md)). This plan's
`mstep_transition_stat`/`mstep_noise_shrink` are candidates to converge on
that same `mstep`/`mstep_frozen_paths` vocabulary, extending it to `Approx`
(and possibly `Dynamics` for subclasses whose map is linear-in-parameters).
Note `Q`'s own continuous per-step cadence should **not** be adopted by
default given its runaway-collapse risk (see this doc's analysis above);
only `R` has been shown safe for continuous cadence. Deliberately deferred,
not part of this plan's scope.
