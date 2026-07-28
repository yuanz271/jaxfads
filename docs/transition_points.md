# Plan: `Approx.transition_points` — pluggable propagation policy for the prediction step

Status: implemented, unit-tested, and empirically validated (real
nonlinear-system + full-training comparison, see Validation plan).
**`MVN` now defaults to `use_sigma_points=True`** — sigma points are no
longer opt-in; validated to be as-good-or-better than MC at every scale
tested (D=8, 16, 32), never worse on the accuracy metric that matters
most (latent recovery), and often cheaper or comparable in wall-clock
cost despite doing more point-evaluations. Lower risk than the Q plan (no
feedback loop); implemented first since the Q plan
(`mstep_dynamics_noise.md`) wants to reuse this.

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

**Architectural invariant** (applies uniformly, at every level, decided
over the course of this plan): distribution classes (`Approx`, `MVN`) own
*parameters and the dispatch decision*; standalone module-level functions
own *algorithms*. No class has an algorithm's actual math sitting inline
inside an overridable/dispatching method — composition over inheritance,
applied consistently rather than only where it was first raised.

### The MC default (`base.py`)

```python
def _monte_carlo_transition_points(approx: "Approx", key, moment, mc_size):
    """Default: mc_size i.i.d. samples via sample_by_moment, uniform
    weights 1/mc_size (today's behavior, bit-for-bit), plus a
    rank-deficiency warning. Standalone, composable, independently
    testable — Approx.transition_points merely delegates to it."""
    points = approx.sample_by_moment(key, moment, mc_size)
    dim = points.shape[-1]  # ambient state dim, read off sample_by_moment's
                             # own output shape -- no new abstract Approx
                             # property needed (param_size() isn't dim; e.g.
                             # full-rank MVN has param_size = dim + dim**2).
    if mc_size <= dim:
        logger.warning(
            "transition_points: mc_size=%d <= state_dim=%d; the MC spread "
            "term is rank-deficient (rank <= mc_size-1 < state_dim), the "
            "same structural precondition behind the Heywood-style exploit "
            "against R (see 'Problem' above). Use mc_size >= state_dim + 1, "
            "or a deterministic policy (e.g. MVN(use_sigma_points=True)) to "
            "avoid this entirely.",
            mc_size, dim,
        )
    weights = jnp.full((mc_size,), 1.0 / mc_size)
    return points, weights


class Approx(SubclassRegistryMixin, ABC):
    def transition_points(self, key, moment, mc_size) -> tuple[Array, Array]:
        """Representative (points, weights) approximating q(z_{t-1}) for
        propagation through the transition. Default: plain Monte Carlo."""
        return _monte_carlo_transition_points(self, key, moment, mc_size)
```

`mc_size` and `points.shape[-1]` are both static Python ints even under
`jit` (array shapes are always static), so the warning's `if` is a plain
Python check resolved once per trace, not a per-call runtime cost, and
won't spam logs during training. Uses the project's existing
`get_logger(__name__)` convention (`smoother.py`/`trainer.py`), not
`warnings.warn`, which isn't used anywhere in this codebase.

**Considered and rejected: an `mc_size=0` "auto" sentinel** that would
internally pick a safe `mc_size` (e.g. `dim + 1`) instead of just warning.
Rejected because there's no well-established heuristic to auto-select:
`dim + 1` is the bare mathematical *minimum* to avoid literal rank
deficiency (linear algebra, certain), not a validated *good-accuracy*
choice (a materially different, much weaker claim) — an ensemble/sample
count just above the rank threshold is a classically noisy, poorly
conditioned covariance estimate, not a target anyone recommends aiming
for. Checked via web search: the unscented transform has a fixed,
dimension-driven point-count formula ($2n+1$; e.g.
[Wikipedia: Unscented transform](https://en.wikipedia.org/wiki/Unscented_transform),
[Särkkä's UKF preprint](https://users.aalto.fi/~ssarkka/pub/ukf-preprint.pdf)),
but plain Monte Carlo sample-size selection has no analogous
dimension-linked formula in the literature — it's determined empirically
(accuracy target, variability, compute budget), not by a formula anyone
cites the way $2n+1$ is cited for UT. An "auto" feature is only as
trustworthy as the heuristic behind it; presenting a bare rank-deficiency
minimum as "auto" would overstate what that number actually guarantees.
The warning states the qualitative structural risk honestly instead, and
points at the actual principled fix (deterministic sigma points).

### The sigma-point alternative, composed into `MVN` (`distributions/mvn.py`)

**Not a subclass.** `MVN` gains one constructor flag and holds a reference
to whichever policy is selected; the sigma-point *algorithm* lives in its
own standalone function, exactly mirroring the MC default above:

```python
class MVN(Approx):
    def __init__(self, dim, rank, *, use_sigma_points: bool = True,
                 ut_alpha: float = 1.0, ut_kappa: float = 0.0):
        ...  # existing init, unchanged
        self._use_sigma_points = use_sigma_points
        self._ut_alpha = ut_alpha
        self._ut_kappa = ut_kappa

    def transition_points(self, key, moment, mc_size):
        if self._use_sigma_points:
            return _unscented_transition_points(
                self, key, moment, mc_size,
                alpha=self._ut_alpha, kappa=self._ut_kappa,
            )
        return super().transition_points(key, moment, mc_size)
```

`use_sigma_points`/`ut_alpha`/`ut_kappa` are keyword-only (`*,`) — checked
every existing `MVN(...)` call site in the repo (tests, benchmarks): all
already use `dim=`/`rank=` keywords, none positional beyond that, so this
is zero-risk and just defensive practice against future positional calls.

**`ut_alpha=1.0, ut_kappa=0.0`, not the `alpha=1e-3` UKF-literature
convention** (a correction from an earlier draft of this plan). This
codebase runs in **float32** throughout (no `enable_x64` anywhere), and
small-`alpha` scaled UT is a known numerical trap there: at, say, `D=8`,
`alpha=1e-3, kappa=0` gives $\lambda \approx -8$, $c = \alpha^2 D \approx
8\times10^{-6}$, so $w_0 \approx -10^6$ and $w_i \approx 6\times10^4$ —
weights that still sum to exactly 1 algebraically, but only via massive
cancellation between huge opposite-sign terms (a finite-difference-style
computation: tiny point spread, $\sqrt{c}\approx0.003$, compensated by huge
weights). float32 has ~7 significant digits to absorb ~6 orders of
magnitude of cancellation — too little margin. `alpha=1.0, kappa=0.0` is
the classical (unscaled) UT instead: $\lambda=0$, $c=D$, $w_0=0$,
$w_i=1/(2D)$ — all non-negative, zero cancellation risk, point spread
scales sensibly as $\sqrt{D}$. Both parameters stay user-overridable for
anyone who specifically wants tighter, more locally-linear sigma points
and is willing to accept the precision tradeoff (or runs with `x64`
enabled).

`MVN`'s fallback calls `super().transition_points(...)` rather than
`_monte_carlo_transition_points(...)` directly, so if `Approx`'s own default
ever evolves beyond a one-line delegation, `MVN` stays in sync via
inheritance instead of needing a second call site tracked by hand.

**Both standalone functions are private** (`_monte_carlo_transition_points`,
`_unscented_transition_points`), not public API — a correction from an
earlier draft of this plan that argued they needed to be public for
testability. Checking the actual planned test/benchmark usage (Steps and
Validation plan, below) shows that claim doesn't hold: every planned test
calls `approx.transition_points(...)` or `expected_predictive_moment(...)`
(the method/public-API surface), never the standalone functions by name,
and the benchmark extension only ever constructs `MVN(...,
use_sigma_points=True)` via `approx_kwargs`. There is no real need to reach
the functions directly. Matches this file's own existing convention for
internal algorithm helpers backing public methods — `_damping_inv`,
`_constrain_chol_full`, `_constrain_chol_diag`, etc. are all
underscore-prefixed already.

This went through two rejected drafts before landing here, both worth
recording as decision notes since the reasoning generalizes:

1. *Rejected: a separate `MVNSigmaPoints(MVN)` subclass* overriding only
   `transition_points`. Unambiguous ("does this class implement its own
   version" has a crisp yes/no answer per class) and matches this
   codebase's existing name-registered-subclass idiom for `Dynamics`/
   `Integrator`/`Likelihood` (different *algorithms* for the same role).
   But composition still wins here: a whole subclass for one overridden
   method is disproportionate (everything else would be pure pass-through
   inheritance), and — the deciding factor — **the graduation path is much
   worse**. If sigma points validate well and should become `MVN`'s own
   default, a subclass forces an awkward choice: `MVN` inheriting from
   `MVNSigmaPoints` (backwards), or copying the logic up into `MVN` and
   deprecating `MVNSigmaPoints` (disruptive for anyone who wrote
   `conf.approx = "MVNSigmaPoints"` explicitly). With composition,
   "becoming the default" is just flipping `use_sigma_points`'s default
   value on `MVN` itself — no class to retire, no config migration.
2. *Rejected: a flag on `MVN` with the sigma-point math written inline* in
   the `if` branch of the overridden method. This is what motivated
   factoring the algorithm out into `_unscented_transition_points` as its
   own standalone unit in the first place — a flag alone doesn't fix the
   "algorithm baked into a dispatch method" problem, it just relocates
   which class has that problem. The architectural invariant at the top of
   this section (classes dispatch, functions compute) is what actually
   resolves it, at every level, including `Approx`'s own MC default (hence
   `_monte_carlo_transition_points` also being factored out, not just
   `_unscented_transition_points`).

Selection stays a plain `bool`, not a named-policy registry (mirroring
`SubclassRegistryMixin` but for policy objects) or a general
callable-injection API. There is exactly one alternative policy today;
building a generic multi-policy plugin surface for a two-choice decision
would be exactly the kind of speculative complexity this repo's own
conventions say to defer until a concrete third option shows up.

### Sigma-point construction (`_unscented_transition_points`)

With $D$ = `dim`, mean $m$ and covariance $P$ from `self.unpack(moment)`,
and tunable scalars $\alpha$ (`ut_alpha`), $\kappa$ (`ut_kappa`):

$$
\lambda = \alpha^2(D+\kappa) - D, \qquad c = D + \lambda
$$

$$
X_0 = m,\quad X_i = m + \sqrt{c}\,L_{:,i},\quad X_{D+i} = m - \sqrt{c}\,L_{:,i}
\quad (i=1,\dots,D)
$$

where $L$ is a square root of $P$: `jnp.linalg.cholesky(P + _EPS*I)` for the
full layout (`rank>0`), `jnp.sqrt(jnp.diag(P))` for the diag layout
(`rank=0`) — cheaper, no $O(D^3)$ decomposition needed when `P` is already
diagonal. This resolves Step 4 below concretely: the diag/full branch lives
inside `_unscented_transition_points`, keyed on `approx._layout.is_diag`.

**Weights — a single set, not the classic UKF mean/covariance pair:**

$$
w_0 = \lambda/c,\qquad w_i = 1/(2c)\ \ (i=1,\dots,2D)
$$

This is a deliberate departure from the doc's original "alpha/beta/kappa"
framing (see Open questions): classic UKF needs *separate* mean-weights and
covariance-weights (with `beta` correcting the covariance reconstruction for
kurtosis) because it reconstructs the output covariance from the sigma
points' own spread around their weighted mean. This framework doesn't do
that — `approx.predictive_moment(z_i, noise)` already returns the *exact*
conditional sufficient statistics (mean **and** $E[zz^T]$, including the
transition noise term) for each point, so
$\sum_i w_i \cdot \text{predictive\_moment}(f(X_i), noise)$ directly
estimates $E_q[\mu_\theta(z_{t-1})]$ by the law of total expectation, with
one weight set. `beta` plays no role here and is intentionally **not**
exposed.

**Known limitation, discovered while flipping `MVN`'s default (see below),
not yet mitigated:** with `alpha=1.0, kappa=0.0`, $w_0 = \lambda/c = 0$
exactly — the center point $X_0$ contributes *zero* weight. If every
non-center point is masked out as non-finite (e.g. `f` diverges away from
the mean but is well-behaved at it), the reduction's total weight is
exactly `0` even though the center point's value was perfectly finite and
usable, and `expected_predictive_moment` correctly-per-its-own-contract
returns NaN (see the `w_sum > 0` guard) — discarding a good value instead
of using it. This is a real gap in the non-finite-masking safety net that
plain MC (uniform, always-positive weights) never has, uncovered by
`_bismooth`'s pre-existing negative-covariance fragility surfacing under
the new default (see `tests/test_algorithm.py::test_bismooth_shapes_and_
finite`'s docstring) rather than by a targeted test. Not fixed here
(`_bismooth` itself is unrelated, documented pre-existing/incomplete
code); worth a follow-up decision — e.g. redistributing $X_0$'s weight
proportionally when it's masked out, or accepting this as a rare-enough
corner case (all `2D` spread points invalid simultaneously, center still
valid) not worth the added complexity.

**Compute-cost tradeoff — theoretically expected, empirically did not
materialize.** Sigma points cost `2D+1` evaluations of `f` per step vs.
`mc_size` for MC (e.g. `65` vs. a safe `mc_size=dim+1=33` at `D=32`,
roughly 2x). The *validated* result (see Validation plan) is that UT is
never meaningfully slower than safe MC across `D=8/16/32`, and is often
modestly *faster* (up to ~12%) despite doing more point-evaluations —
real wall-clock cost is dominated by the encoder/dynamics network's own
forward+backward pass, not by how many points the prediction step
evaluates. The one exception: combined with LoRa (`use_sigma_points=True,
rank<dim`), UT costs modestly more (~12%) than LoRa+MC. This asymmetry
is part of why `use_sigma_points=True` is now `MVN`'s default rather than
an unconditional replacement of `sample_by_moment` inside `Approx`
itself: the cost profile isn't uniformly favorable, even though accuracy
is never worse.

`core.expected_predictive_moment` changes in exactly one place: replace
`approx.sample_by_moment(...)` with `approx.transition_points(...)`, and
generalize the current uniform `sum(safe)/n_valid` reduction to a weighted
one (`sum(w_valid * safe) / sum(w_valid)`, with `w_valid` the weights
zeroed at non-finite entries). This is a strict generalization — uniform
weights reduce to today's formula exactly. Broadcasting of `u`/`c` must use
`z.shape[0]` (the actual point count), not `mc_size` — sigma points always
return `2*dim + 1` points regardless of the `mc_size` argument.

**Why this reduction doesn't need to move into `Approx`** (resolved
discussion, recorded as a decision note): the reduction is mathematically
generic for *any* valid `(points, weights)` pair, precisely because
`predictive_moment` (already `Approx`-owned, pre-existing) is what makes the
per-point moment computation family-specific — the averaging step only
needs weights, not family knowledge. Moving the whole
`expected_predictive_moment` computation into `Approx` would additionally
require giving `Approx` the dynamics function `f`, `u`/`c`, `noise`, and
`key` — coupling `Approx` to "how transitions get applied," which today it
correctly knows nothing about (that's `core.py`'s orchestration job,
uniform across every `Approx`). No concrete `Approx` family today needs a
reduction other than weighted averaging, so this coupling is not adopted.

Implementer / caller summary:
- Default algorithm: `_monte_carlo_transition_points` (`base.py`), delegated
  to by `Approx.transition_points`.
- Alternative algorithm: `_unscented_transition_points` (`distributions/
  mvn.py`), delegated to by `MVN.transition_points` only when
  `use_sigma_points=True`; every other `Approx` subclass inherits the MC
  default untouched.
- Calls `transition_points`: `core.expected_predictive_moment`, single call
  site. No other module (`vi.py`, `trainer.py`, `smoother.py`) needs to
  change.

## Steps (executable, in order)

1. `base.py`: add `_monte_carlo_transition_points` (standalone function)
   and `Approx.transition_points` (delegates to it). Requires adding
   `logger = get_logger(__name__)` to `base.py` (not currently imported
   there; matches `smoother.py`/`trainer.py`'s existing convention). Pure
   addition, no call sites changed yet, no behavior change to any
   returned array — only a new, trace-time-only log line for the
   already-mathematically-risky case.
2. `core.py`: swap `approx.sample_by_moment(subkey, moment, mc_size)` for
   `approx.transition_points(subkey, moment, mc_size)`; broadcast `u`/`c`
   against `z.shape[0]`; generalize the reduction to the weighted-average
   form above. **Checkpoint**: run `tests/test_algorithm.py
   tests/test_core.py tests/test_smoother.py` — must be 100% unchanged
   (this is the "bit-identical for uniform weights" claim, made
   falsifiable before touching `mvn.py` at all).
3. `distributions/mvn.py`: add the standalone `_unscented_transition_points`
   function, and extend `MVN.__init__` with `*, use_sigma_points: bool =
   False, ut_alpha: float = 1.0, ut_kappa: float = 0.0` plus the
   `transition_points` override described above (delegates to
   `_unscented_transition_points` or falls back to `super()`), branching on
   `approx._layout.is_diag` inside the function for the diag/full
   square-root computation (resolves the old Step 4).
4. Non-finite masking (old Step 5): already generalized to weight-based
   masking in step 2 above — no separate reduction logic needed for the
   coarser, deterministic `2D+1`-point set. Add one targeted test (see
   Validation plan) confirming it degrades correctly at this coarser
   granularity, rather than silently trusting the generalization.
5. Tests (new):
   - `tests/test_distribution.py`: `MVN(use_sigma_points=True).
     transition_points` returns `2D+1` points/weights summing to 1, for
     both diag and full layouts (`rank=0`/`rank=dim`), matching the
     existing `_RANK_CASES` parametrization convention in that file; and
     `MVN(use_sigma_points=False)` (explicit opt-out, no longer the
     default -- see below) still reproduces the base class's MC contract
     exactly (`test_transition_points_explicit_mc_matches_base_contract`).
   - `tests/test_algorithm.py`: the discriminative linear-transition test
     (see Validation plan) — this is the one test that actually proves
     the mechanism does something MC can't at matched point count.
   - `tests/test_algorithm.py`: one non-finite-masking test mirroring
     `test_expected_predictive_moment_partial_invalid`, but with
     `MVN(use_sigma_points=True)`'s deterministic points and a threshold
     radius chosen so exactly one of the `2D+1` points is invalid.
6. `docs/transition_points.md` (this file): mark implemented; resolve the
   open questions below concretely rather than leaving them open.

## Validation plan

- **Exact-linear-transition unit test** (discriminative, part of Step 5
  above): for `f(z) = A z + b`, the unscented transform's predicted
  mean/covariance has a known closed form (matches the true
  linear-Gaussian pushforward exactly). Assert `MVN(use_sigma_points=True).
  transition_points` + `expected_predictive_moment` reproduces it (tight
  tolerance, e.g. `atol=1e-5`), and that plain MC at a *matched point
  count* (`mc_size=2*dim+1`) does not (or only approximately, with
  residual sampling noise). This is the one test that proves the
  mechanism does something MC can't at equal cost.
- **Real nonlinear-system + full-training comparison** (performed):
  extended `benchmarks/benchmark_highd_oscillator.py` (nonlinear
  oscillator-bank, scalable latent dim, synthetic ground-truth latents)
  with explicit variants: `FullMVN`/`DiagMVN`/`LoRaMVN-r2` (MC, pinned
  `use_sigma_points=False` so this comparison stays meaningful
  independent of `MVN`'s own default), `FullMVN-mc4-unsafe` (deliberately
  below the `mc_size>=dim+1` threshold), `FullMVN-UT` and
  `LoRaMVN-r{r}-UT` (`use_sigma_points=True`). Ran `dims={8,16,32}`,
  3 seeds, 40 epochs, 64 trials, 200 timesteps (the benchmark's own
  defaults) — n=3 per cell.

  **`post_rmse` (Procrustes-aligned latent recovery vs. ground truth) —
  UT beats safe MC at every scale, margin shrinks with dimension:**

  | dim | FullMVN (safe MC) | FullMVN-UT | FullMVN-mc4-unsafe | UT edge over safe MC |
  |---|---|---|---|---|
  | 8  | 0.4415±0.0203 | **0.3755±0.0279** | 0.4902±0.0160 | ~15% |
  | 16 | 0.5792±0.0072 | **0.5687±0.0060** | 0.6130±0.0087 | ~1.8% |
  | 32 | 0.6733±0.0052 | **0.6653±0.0052** | 0.7044±0.0034 | ~1.2% |

  `obs_rmse` (observation reconstruction) shows the same shrinking-then-
  reversing pattern: UT clearly better at D=8 (~11%), essentially tied at
  D=16, slightly worse at D=32 (~1.8%) — UT's advantage is concentrated
  in latent recovery, not observation fit.

  **Compute cost**: UT is never slower than safe MC at any tested scale —
  ~12% faster at D=8, ~tied at D=16, ~6.6% faster at D=32 — despite doing
  more point-evaluations. The documented "2D+1 vs mc_size" cost-tradeoff
  assumption above does not hold empirically for `FullMVN` vs
  `FullMVN-UT`; real cost is dominated by the encoder/dynamics network,
  not the prediction step's point count (see also the D=16→D=32 CPU-
  scaling discussion below).

  **Stability across seeds — genuinely mixed, not a clean UT win.**
  `post_rmse` std is higher for UT at D=8, comparable-or-lower at D=16/32
  — the "UT removes prediction-step stochasticity, so should reduce
  outcome variance" hypothesis from the original plan is not cleanly
  confirmed; other seed-dependent sources (init, minibatch order)
  apparently dominate. A genuine, unexplained anomaly: at D=32, UT's
  `train_seconds` std jumped to 41.3s (vs `FullMVN`'s 2.1s) in one run
  (24.7s vs 13.9s in a repeat) — `LoRaMVN-r2` (similar parameter count)
  shows comparably high variance, but `FullMVN` itself (same parameter
  count as `FullMVN-UT`) stays low, so this isn't cleanly explained by
  model size either. Flagged as unexplained rather than rationalized.

  **LoRa + UT compose, and compound their benefits** (extension run,
  D=32 only, same 3 seeds): a `LoRaMVN-r2-UT` variant
  (`{"rank": 2, "use_sigma_points": True}`) was added since
  `_unscented_transition_points` branches only on `approx._layout.is_diag`
  (true only for `rank=0`), never on the specific rank value, so it was
  never technically restricted to full rank — just untested before.
  Result: `LoRaMVN-r2-UT` (0.6194±0.0011 post_rmse) beats *every other
  D=32 variant tested*, including plain `LoRaMVN-r2` (0.6336±0.0029) and
  `FullMVN-UT` (0.6653±0.0052) — LoRa's rank-regularization (helps in the
  data-limited high-D regime: 64 trials vs. 1056 full-rank free
  parameters) and UT's deterministic propagation address different
  problems and stack rather than compete. It also has the tightest
  `post_rmse` variance of any variant tested (±0.0011). The one cost
  caveat: `LoRaMVN-r2-UT` (577.7s) costs ~12% *more* than plain
  `LoRaMVN-r2` (515.8s) — unlike the `FullMVN`-vs-`FullMVN-UT` pair, the
  "UT isn't more expensive" finding does not hold for every baseline.

  **Interesting side-finding on CPU scaling** (not specific to UT, but
  discovered via this benchmark): D=8→D=16 scaled almost exactly 2x for
  both diag (32s→64s) and full-layout (79.6s→159.6s) runs, but D=16→D=32
  jumped ~2.86x for *both*, including the diagonal-layout variant (which
  has no `O(D^2)`/`O(D^3)` term in its EF math at all — `rank=0` uses
  `sqrt`, not Cholesky). This points to a hardware/system effect (cache
  or BLAS-kernel-selection threshold crossed at this scale) rather than a
  pure FLOP-complexity story, and is a large part of *why* the naive
  `2D+1`-vs-`mc_size` compute model above doesn't hold empirically:
  real wall-clock cost is not simply proportional to point count once
  other, less predictable, system-level effects dominate.

  Not yet done: a from-scratch (non-oscillator-bank) nonlinear system
  comparison and a qualitative check via `examples/vdp_example.py`
  (`state_dim=2` there is safely above the rank-deficiency threshold
  regardless of propagation method, so it would mainly serve as a
  regression sanity check, not a discriminative one) — low priority given
  the oscillator-bank results already span the regime of interest.
  `benchmarks/pca_dynamics_verification.py` remains unrelated (PCA/NOFILT
  gradient checks), not a fit for this comparison.

## Open questions (resolved)

- **`mc_size` when `use_sigma_points=True`**: stays in
  `transition_points`'s signature for interface parity with the base
  class, but is a documented no-op — point count is always `2*dim + 1`,
  never `mc_size`-controlled. Not repurposed; no evidence yet that
  selecting among UT variants via this argument is needed.
- **Exposure surface for UT tuning parameters**: `ut_alpha`/`ut_kappa`
  only, as `MVN` constructor kwargs (reachable via `approx_kwargs`, same
  mechanism as `rank`). `beta` is **not exposed** — see the weights
  derivation above; there is no covariance-from-spread reconstruction step
  for it to correct.
- **Selection mechanism**: a plain `bool` (`use_sigma_points`) on `MVN`,
  composed with a standalone `_unscented_transition_points` function, not a
  subclass and not a named-policy registry — see the Interface section's
  decision notes for why both alternatives were rejected.

## Open questions (resolved, continued)

- **Whether sigma points should become `MVN`'s default**: yes — done.
  `MVN.__init__`'s `use_sigma_points` default flipped from `False` to
  `True` based on the Validation plan results above (never worse on
  `post_rmse` at any tested scale, often cheaper). The composition-based
  design made this a one-line change, exactly as the graduation-path
  reasoning in the Interface section anticipated — no subclass to
  retire, no config migration. Existing callers that want the old
  behavior pass `use_sigma_points=False` explicitly (backward
  compatibility was not a design goal for this feature).

## Open questions (still open)

- **The zero-weight center point masking gap** (see the decision note
  above, in the weights derivation): not mitigated. Only surfaced by
  `_bismooth`'s pre-existing fragility, not a targeted test; unclear how
  likely it is to matter in real (non-`_bismooth`) filtering/smoothing
  paths. Worth a follow-up test specifically constructing this scenario
  (all `2D` spread points invalid, center valid) outside of `_bismooth`.
- **The D=32 UT `train_seconds` variance anomaly**: unexplained, not
  reliably reproduced in magnitude (41.3s in one run, 24.7s in a repeat,
  vs. `FullMVN`'s consistently low 2-14s). Not blocking, but worth
  watching if UT is used at large scale in real workloads.
- **The D=16→D=32 super-linear CPU scaling**: plausibly a cache/BLAS-
  kernel threshold effect, not confirmed via profiling. Not
  UT-vs-MC-specific (affects `DiagMVN` too), so out of scope for this
  plan, but noted here since it materially affects how compute-cost
  tradeoffs should be reasoned about for any `Approx` at this scale.

## Dependencies

None (self-contained). `mstep_dynamics_noise.md` (the Q plan) wants to reuse
this same mechanism for its "spread of `f(z_{t-1})`" sufficient-statistic
term, so should land after or alongside this plan.
