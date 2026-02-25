# Distribution Design

## Overview

The distribution system follows a two-layer design:

1. **`Approx` ABC** — defines the exponential-family interface.  Natural
   and moment parameters are **flat arrays** (required for array arithmetic).
   Canon and free-form parameters are **pytrees** (distribution-specific
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

Terminology note:
- **Moment parameters** (`μ`) mean the expected sufficient statistics
  `E[T(z)]` of the exponential family.
- `predictive_moment(z, noise)` returns conditional moments
  `E[T(z_t) | z_{t-1}]`.
- For the transition model:
  - **predictive moments** (Eq 4): `μ_θ(z_{t-1}) = E_{p(z_t|z_{t-1})}[T(z_t)]`
  - **expected predictive moments** (Eq 12): `E_{π(z_{t-1})}[μ_θ(z_{t-1})]`.

See `docs/notation.md` for mathematical naming/notation.

Every exponential-family distribution has four representations:

| Form | Type | Reason | Example (MVN) |
|------|------|--------|---------------|
| **Free-form** | pytree | SGD via optax (handles pytrees natively) | `MVNParam(loc, chol_free)` |
| **Canon** | pytree | Human-readable, constraints satisfied | `MVNParam(loc, chol)` with `diag(chol) > 0` |
| **Natural** (η) | flat array | Additive updates in filtering: `η_f = η_p + α` | `[h, J_flat]` (precision-like block) |
| **Moment** (μ) | flat array | Natural↔moment conversions, sampling, KL | `[E[z], E[-½ zzᵀ]_flat]` |

**Why natural and moment must be flat:**
- Natural: additive updates `η_p + α_t` require element-wise addition
- Moment: averaging (Eq 12) and passing flat vectors to sampling/KL backends

**Why canon and free-form are pytrees:**
- JAX and optax handle pytrees natively — no manual flatten/unflatten
- Each subclass defines its own pytree structure (e.g. `MVNParam` NamedTuple)
- Constraints (e.g. softplus) apply to individual leaves
- Equinox stores pytree fields on modules directly

## Conversion Flow

```
free_from_kw(**kwargs)
        │
        ▼
    free-form (pytree)  ◄──── stored on XFADS, optimized by optax
        │
   free_to_canon
        │
        ▼
  canon (pytree)  ◄── valid, human-readable parameters
        │
        ├── moment_to_natural ∘ canon_to_moment
        │           │
        │           ▼
        │       natural (flat)  ◄── additive updates in filtering
        │           │
        │      natural_to_moment
        │           │
        └── canon_to_moment (direct, numerically stable)
                    │
                    ▼
            moment (flat)  ◄── sampling, KL
                    │
              moment_to_canon
                    │
                    ▼
              canon (pytree)
```

Each arrow is an `Approx` method.  The reverse direction is available
where needed (`canon_to_free`, `moment_to_natural`, `moment_to_canon`).

## `Approx` ABC Interface

Quick reference (public API):

| Method | Input → Output | Notes |
|--------|----------------|------|
| `free_from_kw(**kw)` | `kw → free` | Create free-form parameters from a serializable spec. |
| `free_to_canon(free)` | `free → canon` | Apply constraints (e.g. softplus). |
| `canon_to_free(canon)` | `canon → free` | Inverse constraints. |
| `canon_to_moment(canon)` | `canon → moment` | Canon → moment conversion. |
| `moment_to_canon(moment)` | `moment → canon` | Moment → canon conversion. |
| `natural_to_moment(natural)` | `natural → moment` | Natural → moment conversion. |
| `moment_to_natural(moment)` | `moment → natural` | Moment → natural conversion. |
| `sample_by_moment(key, moment, n)` | `moment → samples` | Sampling from the distribution parameterized by `moment`. |
| `kl(moment1, moment2)` | `moment × moment → scalar` | KL between two distributions. |
| `predictive_moment(z, noise)` | `(z, noise) → moment` | Conditional moment parameters `E[T(z_t) | z_{t-1}]`. |

### Initialization

| Method | Signature | Role |
|--------|-----------|------|
| `free_from_kw` | `(**kwargs) → pytree` | Create free-form pytree from serializable spec |
| `param_size` | `(dim) → int` | Natural parameter vector size |

### Canon ↔ free-form (pytree ↔ pytree)

| Method | Signature | Role |
|--------|-----------|------|
| `free_to_canon` | `(free_pytree) → canon_pytree` | Apply constraints (e.g. softplus) |
| `canon_to_free` | `(canon_pytree) → free_pytree` | Inverse constraints |

### Canon ↔ moment (pytree ↔ flat)

| Method | Signature | Role |
|--------|-----------|------|
| `canon_to_moment` | `(canon_pytree) → μ_flat` | Pack pytree into flat moment array |
| `moment_to_canon` | `(μ_flat) → canon_pytree` | Unpack flat moment array into pytree |

### Natural ↔ moment (flat ↔ flat)

| Method | Signature | Role |
|--------|-----------|------|
| `natural_to_moment` | `(η_flat) → μ_flat` | Natural to moment conversion |
| `moment_to_natural` | `(μ_flat) → η_flat` | Moment to natural conversion |

### Inference (flat arrays)

| Method | Signature | Role |
|--------|-----------|------|
| `sample_by_moment` | `(key, μ, n) → z` | Draw `n` samples from the distribution |
| `kl` | `(μ₁, μ₂) → scalar` | KL divergence `KL(p₁ ‖ p₂)` |
| `predictive_moment` | `(z, noise) → μ_flat` | Conditional moment parameters `E[T(z_t) | z_{t-1}]` |

### Usage in XFADS

```python
# Construction — stored as free-form pytrees on the module
self.noise_free = self.approx.free_from_kw(scale=conf.state_noise)
self.unconstrained_prior_natural = self.approx.free_from_kw(scale=1.0)

# Inference — derive natural/moment on the fly
prior_natural = self.approx.moment_to_natural(
    self.approx.canon_to_moment(
        self.approx.free_to_canon(self.unconstrained_prior_natural)
    )
)
noise = self.approx.canon_to_moment(
    self.approx.free_to_canon(self.noise_free)
)

# Encoder outputs are flat arrays → flat natural params
# (encoders output flat for additive updates in filtering)
def _free_to_natural(free_flat):
    free_pytree = self.approx.moment_to_canon(free_flat)
    return self.approx.moment_to_natural(
        self.approx.canon_to_moment(self.approx.free_to_canon(free_pytree))
    )
alpha = _free_to_natural(encoder(y))
```

### Notes

- `predictive_moment(z, noise)` is **single-state**: it takes one dynamics output
  `z` with shape `(D,)` and transition noise parameters `noise` (flat array; use
  `jnp.array([])` if unused). It returns **moment parameters** (expected
  sufficient statistics) `E[T(z_t) | z_{t-1}]` in a flat vector form.
- Monte Carlo averaging across samples of `z_{t-1}` is handled outside the
  distribution (in `core.expected_predictive_moment`) via `jax.vmap` +
  `jnp.mean`.
- Encoder outputs remain flat arrays (natural parameter updates for additive
  filtering). They pass through `moment_to_canon` → `free_to_canon` →
  `canon_to_moment` → `moment_to_natural` to convert from unconstrained flat to
  valid natural parameters.

## MVN Subclass

`MVN(dim, structure=...)` implements either a full-covariance or diagonal
multivariate normal exponential family:

### MVNParam NamedTuple

```python
class MVNParam(NamedTuple):
    loc: Array   # (D,)
    chol: Array  # (D, D) lower-triangular Cholesky factor
```

Both canon and free-form use `MVNParam`. The difference:
- **Canon**: `chol` has positive diagonal entries.
- **Free-form**: `chol` has unconstrained diagonal entries.

### Flat layouts

**Moment** (flat array): moment parameters `μ = E[T(z)]` with
`T(z) = [z, -½ zzᵀ]`:

- `[E[z] (D), E[-½ zzᵀ] (D×D) flattened]` → total `D + D²`.

**Natural** (flat array): natural parameters `[h, J_flat]` where `J` is the
precision matrix:

- `[h (D), J (D×D) flattened]` → total `D + D²`.

### Gaussian transition sufficient statistics

For the transition:

```
z_t | z_{t-1}=z  ~  N( f(z), Q )
```

With `T(z_t) = [z_t, -½ z_t z_tᵀ]`, the conditional moment parameters are:

- `E[T₁(z_t) | z] = μ = f(z)`
- `E[T₂(z_t) | z] = -½ (Q + μ μᵀ)`

`MVN.predictive_moment(z, noise)` returns exactly these conditional moment
parameters in the flat moment layout.

### Diagonal MVN variant

`MVN(..., structure="diag")` uses diagonal sufficient statistics:

- `T(z) = [z, -½ (z ⊙ z)]`

and therefore stores flat moment/natural parameters of size `2D`.

For caller uniformity, `MVN.unpack(moment)` still returns a *full* covariance
matrix of shape `(D, D)` (diagonal in value).

### Additional methods (MVN-specific)

| Method | Role |
|--------|------|
| `pack(mean, cov) → moment` | Convert `(E[z], Cov(z))` into moment parameters `E[T(z)]`. |
| `unpack(moment) → (mean, cov)` | Extract `(E[z], Cov(z))` from moment parameters. |

## Observation Likelihoods vs Latent Approximations

The codebase has two distinct distribution concerns:

| Concern | Class | Requirements | Constraint |
|---------|-------|-------------|------------|
| **Latent posterior** | `Approx` | Natural/moment parameterization, KL, sampling, predict | Must be exponential family |
| **Observation likelihood** | `Likelihood` | `eloglik(z, y)` | None — can be any distribution |

These are fundamentally different:

- **`Approx`** needs the exponential-family machinery (natural parameters
  for additive filtering updates, moment parameters for KL and sampling).
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
  use `approx.sample_by_moment` to draw samples and average
  `log p(y | z)`.

This coupling is deliberate — it lets each `Observation` subclass
pick the most efficient evaluation path for its likelihood family
and the given posterior approximation.

## Adding a New Distribution

1. Subclass `Approx` in a new module under `src/jaxfads/distributions/`.
2. Define a pytree type (e.g. a `NamedTuple`) for canon/free-form params.
3. Implement all abstract methods from the ABC.
4. Define any distribution-specific methods as regular methods on the subclass.
5. Re-export from `src/jaxfads/distributions/__init__.py` (triggers registration).
6. Add tests in `tests/test_distribution.py`.
7. The subclass is automatically discoverable via
   `Approx.get_subclass("ClassName")` (from `SubclassRegistryMixin`).
