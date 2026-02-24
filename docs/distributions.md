# Distribution Design

## Overview

The distribution system follows a two-layer design:

1. **`Approx` ABC** — defines the exponential-family interface using opaque
   flat arrays.  It has no knowledge of how parameters are structured
   internally.
2. **Subclasses** (e.g. `MVN`) — handle the internal structure of parameters
   and provide conversions between free-form, structured, natural, and
   mean representations.

The core algorithm (`core.py`, `vi.py`) interacts only with the `Approx`
interface.  Distribution-specific callers (e.g. `observations.py`,
`smoother.noise_cov`) access subclass methods directly through the
concrete instance.

## Parameter Forms

Every exponential-family distribution has four representations:

| Form | Description | Stored? | Example (MVN) |
|------|-------------|---------|---------------|
| **Free-form** | Unconstrained array for SGD optimization | ✓ | Raw reals in ℝⁿ |
| **Structured param** | Valid structured parameters | | `[loc, cov_diag, cov_factor]` with `cov_diag > 0` |
| **Natural** (η) | Canonical exponential-family parameters | | `[η₁, η₂]` with `η₂` negative definite |
| **Mean** (μ) | Expected sufficient statistics `E[T(x)]` | | Used for sampling, KL, prediction |

All arrays stored on `XFADS` are in **free-form**.  Other forms are
derived at inference time through the conversion methods below.

## Conversion Flow

```
param_from_conf(**kwargs)
        │
        ▼
    free-form  ◄──── stored on XFADS, optimized by SGD
        │
   to_structured
        │
        ▼
  structured param  ◄── valid structured parameters
        │
        ├── structured_to_natural
        │           │
        │           ▼
        │       natural  ◄── additive updates in filtering (η_f = η_p + α)
        │           │
        │      natural_to_mean
        │           │
        └── structured_to_mean (shortcut, numerically stable)
                    │
                    ▼
                  mean  ◄── sampling, KL, predict_mean
```

Each arrow is an `Approx` method.  The reverse direction is available
where needed (`to_free`, `mean_to_natural`).

## `Approx` ABC Interface

All methods operate on opaque flat arrays.

### Initialization

| Method | Signature | Role |
|--------|-----------|------|
| `param_from_conf` | `(**kwargs) → free` | Create free-form array from serializable spec |

Each subclass defines which kwargs it accepts.  The returned array is
suitable for storage on `XFADS` and optimization by SGD.

### Structured transforms

| Method | Signature | Role |
|--------|-----------|------|
| `to_structured` | `(free) → param` | Free-form to valid structured parameters |
| `to_free` | `(param) → free` | Inverse of `to_structured` |

### Parameter conversions

| Method | Signature | Role |
|--------|-----------|------|
| `structured_to_natural` | `(param) → η` | Structured parameters to natural form |
| `structured_to_mean` | `(param) → μ` | Structured parameters to mean form (shortcut) |
| `natural_to_mean` | `(η) → μ` | Natural to mean conversion |
| `mean_to_natural` | `(μ) → η` | Mean to natural conversion |

`structured_to_mean` is a direct path from structured parameters
to mean form, avoiding the roundtrip through natural parameters
(`structured → natural → mean`) which may involve numerically unstable
operations (e.g. matrix inversion for MVN).

### Inference

| Method | Signature | Role |
|--------|-----------|------|
| `sample_by_mean` | `(key, μ, n) → z` | Draw `n` samples from the distribution |
| `kl` | `(μ₁, μ₂) → scalar` | KL divergence `KL(p₁ ‖ p₂)` |
| `predict_mean` | `(locs, noise_μ) → μ` | Predicted mean from batch of dynamics locations |
| `param_size` | `(state_dim) → int` | Natural parameter vector size |

### Usage in XFADS

```python
# Construction — all stored as free-form
self.prior_params = self.approx.param_from_conf(scale=1.0)
self.noise_params = self.approx.param_from_conf(scale=conf.state_noise)

# Inference — derive natural/mean on the fly
prior_natural = self.approx.structured_to_natural(
    self.approx.to_structured(self.prior_params)
)
noise_mean = self.approx.structured_to_mean(
    self.approx.to_structured(self.noise_params)
)

# Encoder outputs — network → structured → natural
alpha = self.approx.structured_to_natural(
    self.approx.to_structured(encoder(y))
)
```

### Notes

- `predict_mean` takes a batch of dynamics locations `(N, D)` and noise
  mean params.  Each subclass handles sufficient-statistic averaging
  internally.  The `noise_mean` parameter may be empty (`jnp.array([])`)
  for families without a separate dispersion parameter.

## MVN Subclass

`MVN(dim, rank)` implements the multivariate normal with covariance
structure `Σ = diag(d) + U Uᵀ`.

### Structured Parameter Layout

```
[loc(D), cov_diag(D), cov_factor(D × r)]  →  total: D(2 + r)
```

Where `cov_diag > 0` (enforced by softplus in `to_structured`).

### Natural Parameter Layout

- rank 0: `[η₁(D), η₂(D)]` → total: `2D`
- rank > 0: `[η₁(D), η₂_flat(D²)]` → total: `D + D²`

### Additional Methods (not on ABC)

| Method | Role |
|--------|------|
| `mean_to_canon(μ) → (loc, cov)` | Convert mean params to canonical `(loc, cov)` tuple |
| `canon_to_mean(loc, cov) → μ` | Convert canonical form to mean params |
| `full_cov(cov) → (D, D)` | Materialize full covariance matrix |
| `mean_size(state_dim) → int` | Mean parameter vector size |

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
