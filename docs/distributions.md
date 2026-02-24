# Distribution Design

## Overview

The distribution system follows a two-layer design:

1. **`Approx` ABC** — defines the exponential-family interface using opaque
   flat arrays.  It has no knowledge of how parameters are structured
   internally.
2. **Subclasses** (e.g. `MVN`) — handle the internal structure of parameters
   and provide conversions between natural, mean, and structured forms.

The core algorithm (`core.py`, `vi.py`) interacts only with the `Approx`
interface.  Distribution-specific callers (e.g. `observations.py`,
`smoother.noise_cov`) access subclass methods directly through the concrete
instance.

## Parameterizations

Every exponential-family distribution has three forms:

| Form | Level | Description |
|------|-------|-------------|
| **Natural** (η) | ABC | Canonical parameters of the exponential family.  Used in the filtering algorithm for additive updates. |
| **Mean** (μ) | ABC | Expected sufficient statistics `E[T(x)]`.  Stored as a flat array; used for KL, sampling, and storage. |
| **Structured** | Subclass | Human-readable parameters specific to the distribution family (e.g. `(loc, cov)` for MVN). |

The ABC treats natural and mean parameters as opaque vectors.  Only
subclasses know the internal layout.

## `Approx` ABC Interface

All methods operate on opaque flat arrays.

| Method | Signature | Role |
|--------|-----------|------|
| `natural_to_mean` | `(η) → μ` | Natural to mean conversion |
| `mean_to_natural` | `(μ) → η` | Mean to natural conversion |
| `sample_by_mean` | `(key, μ, n) → z` | Draw `n` samples from the distribution |
| `kl` | `(μ₁, μ₂) → scalar` | KL divergence `KL(p₁ ‖ p₂)` |
| `predict_mean` | `(locs, noise_μ) → μ` | Predicted mean from batch of dynamics locations |
| `constrain_mean` | `(unconstrained) → μ` | Map unconstrained params to valid mean params |
| `unconstrain_mean` | `(μ) → unconstrained` | Inverse of `constrain_mean` |
| `constrain_natural` | `(unconstrained) → η` | Map unconstrained params to valid natural params |
| `unconstrain_natural` | `(η) → unconstrained` | Inverse of `constrain_natural` |
| `param_size` | `(state_dim) → int` | Natural parameter vector size |
| `mean_size` | `(state_dim) → int` | Mean parameter vector size |
| `prior_natural` | `(state_dim) → η` | Default prior in natural form |
| `init_noise` | `(scale, state_dim) → unconstrained` | Initialize noise mean params |

### Notes

- `predict_mean` takes a batch of dynamics locations `(N, D)` and noise
  mean params.  Each subclass handles sufficient-statistic averaging
  internally.  The `noise_mean` parameter may be empty (`jnp.array([])`)
  for families without a separate dispersion parameter.
- `constrain_*` / `unconstrain_*` enable unconstrained optimization while
  ensuring parameters remain in the valid domain (e.g. positive-definite
  covariance).

## MVN Subclass

`MVN(dim, rank)` implements the multivariate normal with covariance
structure `Σ = diag(d) + U Uᵀ`.

### Mean Parameter Layout

```
[loc(D), cov_diag(D), cov_factor(D × r)]  →  total: D(2 + r)
```

### Natural Parameter Layout

- rank 0: `[η₁(D), η₂(D)]` → total: `2D`
- rank > 0: `[η₁(D), η₂_flat(D²)]` → total: `D + D²`

### Additional Methods (not on ABC)

| Method | Role |
|--------|------|
| `mean_to_canon(μ) → (loc, cov)` | Convert mean params to canonical `(loc, cov)` tuple |
| `canon_to_mean(loc, cov) → μ` | Convert canonical form to mean params |
| `full_cov(cov) → (D, D)` | Materialize full covariance matrix |

These are used by distribution-specific callers such as observation
models and the noise regularization loss.

### Implementation Details

- rank 0 uses element-wise vector operations (`O(D)`) for efficiency.
- rank > 0 uses matrix operations (`O(D³)`) via `jnp.linalg.solve`.
- Both paths are JIT-compatible: branching is on `self._rank` (a plain
  Python int), so only one path is traced.
- KL divergence uses `tfd.MultivariateNormalFullCovariance` as backend
  because `tfd.MultivariateNormalDiagPlusLowRankCovariance` does not
  register a KL kernel in TFP.
- `_decompose_cov` is a lossy operation for rank > 0: it zeros the
  diagonal before eigendecomposition, so the off-diagonal reconstruction
  is a best rank-r approximation.  The diagonal is preserved exactly.

## Adding a New Distribution

1. Subclass `Approx` in a new module under `src/jaxfads/distributions/`.
2. Implement all abstract methods from the ABC.
3. Define any distribution-specific methods (e.g. structured form
   conversions) as regular methods on the subclass.
4. Re-export from `src/jaxfads/distributions/__init__.py`.
5. Add tests in `tests/test_distribution.py`.
6. The subclass is automatically discoverable via
   `Approx.get_subclass("ClassName")` (from `SubclassRegistryMixin`).
