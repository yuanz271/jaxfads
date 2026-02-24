# Distribution Design

## Overview

The distribution system follows a two-layer design:

1. **`Approx` ABC** — defines the exponential-family interface.  Natural
   and mean parameters are **flat arrays** (required for array arithmetic).
   Structured and free-form parameters are **pytrees** (distribution-specific
   layout handled natively by JAX/Equinox/optax).
2. **Subclasses** (e.g. `MVN`) — define the concrete pytree structure and
   implement conversions between all four forms.

The core algorithm (`core.py`, `vi.py`) interacts only with the `Approx`
interface.  Distribution-specific callers (e.g. `observations.py`) access
subclass methods directly through the concrete instance.

### Subclass Registration

Subclasses are discovered via `Approx.get_subclass("MVN")` using
`SubclassRegistryMixin.__init_subclass__`.  Registration happens when
the subclass module is imported.  Built-in distributions are registered
by a side-effect import in `smoother.py`:

```python
from . import distributions  # registers Approx subclasses
```

Plugin distributions in external packages register on user import.

## Parameter Forms

Every exponential-family distribution has four representations:

| Form | Type | Reason | Example (MVN) |
|------|------|--------|---------------|
| **Free-form** | pytree | SGD via optax (handles pytrees natively) | `MVNParam(loc, diag_free, factor)` |
| **Structured** | pytree | Human-readable, constraints satisfied | `MVNParam(loc, cov_diag, cov_factor)` with `cov_diag > 0` |
| **Natural** (η) | flat array | Additive updates in filtering: `η_f = η_p + α` | `[η₁, η₂]` with `η₂` negative definite |
| **Mean** (μ) | flat array | Averaging in `predict_mean`, TFP for KL/sampling | `[loc, cov_diag, cov_factor]` packed flat |

**Why natural and mean must be flat:**
- Natural: additive updates `η_p + α_t` require element-wise addition
- Mean: averaging `(1/S) Σ μ_θ(z^s)` in `predict_mean`, passed to TFP

**Why structured and free-form are pytrees:**
- JAX and optax handle pytrees natively — no manual flatten/unflatten
- Each subclass defines its own pytree structure (e.g. `MVNParam` NamedTuple)
- Constraints (e.g. softplus) apply to individual leaves
- Equinox stores pytree fields on modules directly

## Conversion Flow

```
param_from_conf(**kwargs)
        │
        ▼
    free-form (pytree)  ◄──── stored on XFADS, optimized by optax
        │
   to_structured
        │
        ▼
  structured (pytree)  ◄── valid, human-readable parameters
        │
        ├── mean_to_natural ∘ structured_to_mean
        │           │
        │           ▼
        │       natural (flat)  ◄── additive updates in filtering
        │           │
        │      natural_to_mean
        │           │
        └── structured_to_mean (direct, numerically stable)
                    │
                    ▼
              mean (flat)  ◄── sampling, KL, predict_mean
                    │
              mean_to_structured
                    │
                    ▼
              structured (pytree)
```

Each arrow is an `Approx` method.  The reverse direction is available
where needed (`to_free`, `mean_to_natural`, `mean_to_structured`).

## `Approx` ABC Interface

### Initialization

| Method | Signature | Role |
|--------|-----------|------|
| `param_from_conf` | `(**kwargs) → pytree` | Create free-form pytree from serializable spec |
| `param_size` | `(dim) → int` | Natural parameter vector size |

### Structured ↔ free-form (pytree ↔ pytree)

| Method | Signature | Role |
|--------|-----------|------|
| `to_structured` | `(free_pytree) → structured_pytree` | Apply constraints (e.g. softplus) |
| `to_free` | `(structured_pytree) → free_pytree` | Inverse constraints |

### Structured ↔ mean (pytree ↔ flat)

| Method | Signature | Role |
|--------|-----------|------|
| `structured_to_mean` | `(structured_pytree) → μ_flat` | Pack pytree into flat mean array |
| `mean_to_structured` | `(μ_flat) → structured_pytree` | Unpack flat mean array into pytree |

### Natural ↔ mean (flat ↔ flat)

| Method | Signature | Role |
|--------|-----------|------|
| `natural_to_mean` | `(η_flat) → μ_flat` | Natural to mean conversion |
| `mean_to_natural` | `(μ_flat) → η_flat` | Mean to natural conversion |

### Inference (flat arrays)

| Method | Signature | Role |
|--------|-----------|------|
| `sample_by_mean` | `(key, μ, n) → z` | Draw `n` samples from the distribution |
| `kl` | `(μ₁, μ₂) → scalar` | KL divergence `KL(p₁ ‖ p₂)` |
| `predict_mean` | `(locs, noise_μ) → μ` | Predicted mean from batch of dynamics locations |

### Usage in XFADS

```python
# Construction — stored as free-form pytrees on the module
self.noise_free = self.approx.param_from_conf(scale=conf.state_noise)
self.unconstrained_prior_natural = self.approx.param_from_conf(scale=1.0)

# Inference — derive natural/mean on the fly
prior_natural = self.approx.mean_to_natural(
    self.approx.structured_to_mean(
        self.approx.to_structured(self.unconstrained_prior_natural)
    )
)
noise_mean = self.approx.structured_to_mean(
    self.approx.to_structured(self.noise_free)
)

# Encoder outputs are flat arrays → flat natural params
# (encoders output flat for additive updates in filtering)
def _free_to_natural(free_flat):
    free_pytree = self.approx.mean_to_structured(free_flat)
    return self.approx.mean_to_natural(
        self.approx.structured_to_mean(self.approx.to_structured(free_pytree))
    )
alpha = _free_to_natural(encoder(y))
```

### Notes

- `predict_mean` takes a batch of dynamics locations `(N, D)` and noise
  mean params (flat).  Each subclass handles sufficient-statistic
  averaging internally.
- Encoder outputs remain flat arrays (natural parameter updates for
  additive filtering).  They pass through `mean_to_structured` →
  `to_structured` → `structured_to_mean` → `mean_to_natural` to
  convert from unconstrained flat to valid natural parameters.

## MVN Subclass

`MVN(dim, rank)` implements the multivariate normal with covariance
structure `Σ = diag(d) + U Uᵀ`.

### MVNParam NamedTuple

```python
class MVNParam(NamedTuple):
    loc: Array        # (D,)
    cov_diag: Array   # (D,)
    cov_factor: Array # (D, r)
```

Both structured and free-form use `MVNParam`.  The difference:
- **Structured**: `cov_diag > 0` (constrained)
- **Free-form**: `cov_diag ∈ ℝ` (unconstrained, softplus maps to positive)

### Flat Layouts

**Mean** (flat array): `[loc(D), cov_diag(D), cov_factor_flat(D×r)]` → total `D(2+r)`

**Natural** (flat array):
- rank 0: `[η₁(D), η₂(D)]` → total `2D`
- rank > 0: `[η₁(D), η₂_flat(D²)]` → total `D + D²`

### Additional Methods (not on ABC)

| Method | Role |
|--------|------|
| `structured_to_natural(MVNParam) → η_flat` | Convenience: compose `mean_to_natural(structured_to_mean(...))` |
| `unpack(μ_flat) → (loc, cov)` | Extract `(loc, cov_diag_or_full)` from flat mean |
| `pack(loc, cov) → μ_flat` | Inverse of `unpack` (lossy for rank > 0 via `_decompose_cov`) |
| `full_cov(cov) → (D, D)` | Materialize full covariance matrix |
| `mean_size(dim) → int` | Mean parameter vector size |

### Implementation Details

- rank 0 uses element-wise vector operations (`O(D)`) for efficiency.
- rank > 0 uses matrix operations (`O(D³)`) via `jnp.linalg.solve`.
- Both paths are JIT-compatible: branching is on `self._rank` (a plain
  Python int), so only one path is traced.
- KL divergence uses `tfd.MultivariateNormalFullCovariance` as backend
  because `tfd.MultivariateNormalDiagPlusLowRankCovariance` does not
  register a KL kernel in TFP.
- `_decompose_cov` is lossy for rank > 0: it zeros the diagonal before
  eigendecomposition, so the off-diagonal reconstruction is a best
  rank-r approximation.  The diagonal is preserved exactly.

## Observation Likelihoods vs Latent Approximations

The codebase has two distinct distribution concerns:

| Concern | Class | Requirements | Constraint |
|---------|-------|-------------|------------|
| **Latent posterior** | `Approx` | Natural/mean parameterization, KL, sampling, predict | Must be exponential family |
| **Observation likelihood** | `Likelihood` | `eloglik(z, y)` | None — can be any distribution |

These are fundamentally different:

- **`Approx`** needs the exponential-family machinery (natural parameters
  for additive filtering updates, mean parameters for KL and sampling).
- **Observation likelihoods** (Poisson, Gaussian, Bernoulli, etc.) only
  need to evaluate `E_q[log p(y | z)]`.  They are not restricted to
  exponential families — any density or even a neural likelihood works.

### Why `approx` is passed to `eloglik`

The `approx` parameter enables subclasses to choose their evaluation
strategy:

- **Analytical** (e.g. Gaussian likelihood + Gaussian posterior):
  extract posterior moments via `approx` for closed-form
  `E_q[log p(y | z)]` — lower variance, more efficient.
- **Monte Carlo** (e.g. Poisson, or any complex likelihood):
  use `approx.sample_by_mean` to draw samples and average
  `log p(y | z)`.

This coupling is deliberate — it lets each `Observation` subclass
pick the most efficient evaluation path for its likelihood family
and the given posterior approximation.

## Adding a New Distribution

1. Subclass `Approx` in a new module under `src/jaxfads/distributions/`.
2. Define a pytree type (e.g. a `NamedTuple`) for structured/free-form params.
3. Implement all abstract methods from the ABC.
4. Define any distribution-specific methods as regular methods on the subclass.
5. Re-export from `src/jaxfads/distributions/__init__.py` (triggers registration).
6. Add tests in `tests/test_distribution.py`.
7. The subclass is automatically discoverable via
   `Approx.get_subclass("ClassName")` (from `SubclassRegistryMixin`).
