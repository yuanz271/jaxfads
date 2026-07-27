# Plan: `Approx.transition_points` — pluggable propagation policy for the prediction step

Status: proposed, not implemented. Lower risk than the Q plan (no feedback loop);
implement first since the Q plan (`mstep_dynamics_noise.md`) wants to reuse this.

See also: [mstep_gaussian_cov](mstep_gaussian_cov.md), [mstep_dynamics_noise](mstep_dynamics_noise.md).

## Problem

`core.expected_predictive_moment` moment-matches the one-step-ahead predictive
distribution by drawing `mc_size` i.i.d. samples of `z_{t-1}` via
`approx.sample_by_moment`, pushing each through the transition `f`, and
averaging sufficient statistics. The implied predictive covariance is

```
Cov_p(z_t) = Q + Cov_over_samples[f(z^(1)_{t-1}), ..., f(z^(S)_{t-1})]
```

i.e. a diagonal term (`Q`, dimension `state_dim`) plus a between-sample
"spread" term of the same ambient dimension. That spread term's rank is
bounded by `mc_size - 1`. When `mc_size < state_dim` (e.g. `mc_size=4` in
`examples/vdp_example.py`, plausibly less than `state_dim` for many models),
this reproduces the same structural precondition (diagonal + rank-deficient
term, jointly optimized) that enabled the Heywood-style exploit against the
observation noise `R` — this time against the process noise `Q`.

Separately: since `q(z_{t-1})` is already assumed Gaussian (that's the point
of the MVN exponential-family approx), the prediction step doesn't need
random sampling at all — it's fundamentally a deterministic moment-matching
computation. Random MC sampling adds pure sampling variance to something
that could be computed exactly (to first two moments) via a deterministic
transform, e.g. the unscented transform (UKF sigma points).

## Design constraint

The framework is designed so `Observation`, `Dynamics`, and `Approx` are
user-pluggable, and `core.py` must stay blind to which concrete
implementation it's talking to. A sigma-point transform is inherently
Gaussian-specific (relies on a Cholesky factor of a covariance matrix), so it
must not be hardcoded into `core.py` or added as a required abstract method
on `Approx`.

## Interface

Add a **concrete, non-abstract, overridable** method on the `Approx` base
class (`base.py`), following the same backward-compatible-default pattern
already used for `free_size`/`free_to_natural`:

```
class Approx:
    def transition_points(self, key, moment, mc_size) -> tuple[Array, Array]:
        """Representative (points, weights) approximating q(z_{t-1}) for
        propagation through the transition. Default: mc_size i.i.d. samples
        via sample_by_moment, uniform weights 1/mc_size (today's behavior,
        bit-for-bit)."""
```

`MVN` overrides it with deterministic sigma points: `2*dim + 1` points at the
mean and at `mean +/- sqrt((dim + lambda) * cov)` columns of the Cholesky
factor, with standard unscented-transform weights (ignoring `key`).

`core.expected_predictive_moment` changes in exactly one place: replace
`approx.sample_by_moment(...)` with `approx.transition_points(...)`, and
generalize the current uniform `sum(safe)/n_valid` reduction to a weighted
one (`sum(w * safe) / sum(w)`, with `w` renormalized over finite entries).
This is a strict generalization — uniform weights reduce to today's formula
exactly.

Implementer / caller summary:
- Implements the default: `Approx` base class (`base.py`).
- Overrides it: `MVN` only (for now).
- Calls it: `core.expected_predictive_moment`, single call site. No other
  module (`vi.py`, `trainer.py`, `smoother.py`) needs to change.

## Steps

1. Extract today's inline MC-sampling call in `expected_predictive_moment`
   into the new `Approx.transition_points` default — pure refactor, no
   behavior change.
2. Generalize the reduction to a weighted average; add a regression test
   confirming bit-identical output vs. current behavior for uniform weights.
3. Implement `MVN.transition_points` (sigma points), gated behind an
   explicit opt-in (constructor flag or `approx_kwargs`) so existing model
   configs/checkpoints are unaffected until validated. Do not silently
   change the default.
4. Handle `rank=0` (diagonal) vs `rank>0` (full) `MVN` layouts explicitly —
   decide whether the diagonal case needs a cheaper point set.
5. Verify the non-finite masking logic (used for numerically unstable `f`)
   still makes sense for a small, deterministic point set (fewer points
   means each non-finite point is a larger fraction of the total weight
   than in a large MC batch — check this doesn't make the safety net too
   coarse).
6. UT tuning parameters (alpha/beta/kappa) live on `MVN`'s constructor /
   `approx_kwargs`, not threaded generically through `core.py`.

## Validation plan

- **Exact-linear-transition unit test** (discriminative): for `f(z) = A z +
  b`, the unscented transform's predicted mean/covariance has a known closed
  form (matches the true linear-Gaussian pushforward exactly). Assert
  `MVN.transition_points` + `expected_predictive_moment` reproduces it, and
  that plain MC does not (or only approximately, with residual sampling
  noise) at matched cost.
- **Nonlinear integration test**: compare predictive-moment error against a
  high-`mc_size` MC reference on a nonlinear system (e.g. Van der Pol, as
  used in `examples/vdp_example.py`).
- **Training comparison**: loss-curve variance/stability comparison between
  MC (`mc_size` scaled to `state_dim` per the interim heuristic) and sigma
  points on the same example.

## Open questions

- Whether to keep `mc_size` as an ignored/no-op config field for `MVN` once
  sigma points are the default path, or repurpose it (e.g. select among UT
  variants).
- Exposure surface for UT tuning parameters (alpha/beta/kappa).
- Whether sigma points should eventually become `MVN`'s *default* propagation
  policy or stay opt-in indefinitely (depends on validation results above).

## Dependencies

None (self-contained). `mstep_dynamics_noise.md` (the Q plan) wants to reuse
this same mechanism for its "spread of `f(z_{t-1})`" sufficient-statistic
term, so should land after or alongside this plan.
