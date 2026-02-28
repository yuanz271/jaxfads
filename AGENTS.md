# JAXFADS - Project Knowledge Base

**Updated:** 2026-02-25  
**Branch:** feature/unify-noise-approx

## Overview

JAX library for variational Bayesian state-space modeling (XFADS). Implements
variational inference for nonlinear state-space models with exponential-family
approximate posteriors via Equinox/JAX.

## Structure

```
jaxfads/
├── src/jaxfads/                 # Core library
│   ├── base.py                  # ABCs + subclass registries (Approx/Dynamics/Observation)
│   ├── smoother.py              # XFADS orchestrator
│   ├── core.py                  # filtering/smoothing primitives
│   ├── vi.py                    # ELBO
│   ├── observations.py          # GLM observation model + Poisson/Gaussian likelihoods
│   ├── encoders.py              # Alpha/Beta encoders
│   ├── trainer.py               # training loop
│   ├── distributions/           # Approx implementations (currently MVN)
│   │   └── mvn.py
│   ├── nn.py                    # neural blocks (MLPs, readouts, etc.)
│   ├── constraints.py           # parameter constraints
│   ├── util.py                  # helpers
│   └── ilqr.py                  # iLQR utilities
├── tests/                       # Unit tests mirroring src/ structure
└── pyproject.toml               # Build config, deps, ruff settings
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Main model class | `src/jaxfads/smoother.py` | `XFADS` |
| Filtering primitives | `src/jaxfads/core.py` | `filter()`, `bismooth()`, `expected_predictive_moment()` |
| Approx posterior families | `src/jaxfads/distributions/` | currently `MVN` |
| Observation models | `src/jaxfads/observations.py` | `GLM`, `Poisson`, `Gaussian` |
| Encoders | `src/jaxfads/encoders.py` | `AlphaEncoder`, `BetaEncoder` |
| Training | `src/jaxfads/trainer.py` | `train()`, `batch_loss()`, `DEFAULT_TRAINER_CONFIG` |
| Abstract interfaces | `src/jaxfads/base.py` | `Approx`, `Dynamics`, `Observation` |

## Core Design Notes

### Exponential-family parameter forms

- **Natural parameters** (η): flat vectors for additive filtering updates.
- **Moment parameters** (μ): flat vectors of expected sufficient statistics
  `E[T(z)]`.

### MVN approximation

`MVN(dim, structure=...)` supports two exponential-family layouts:

- `structure="full"`:
  - natural size `D + D²`
  - moment size `D + D²` with `T₂(z) = -½ zzᵀ`
- `structure="diag"`:
  - natural size `2D`
  - moment size `2D` with `T₂(z) = -½ (z ⊙ z)`

Invariant for callers:
- `MVN.unpack(moment)` returns `(mean, cov)` where `cov` is **always** a full
  `(D, D)` covariance matrix (diagonal-valued in diag mode).

### Config invariants (avoiding duplication)

- `conf.state_dim` and `conf.observation_dim` are the **single source of truth**.
  `XFADS` injects these into `dyn_conf`/`obs_conf`/`enc_conf` at construction.
- `enc_conf` does **not** contain `approx` / `approx_kwargs`.
  Encoders are Approx-agnostic and only require:
  - `param_size` (natural-parameter size; injected by `XFADS`)
  - `free_size` (encoder free-form size; injected by `XFADS`)
  - `observation_dim` (injected by `XFADS`)
  - encoder hyperparameters (`width`, `depth`, `dropout`)

### Training regularization

- The trainer supports an optional user hook:
  - `trainer_conf.noise_regularizer: Callable[[XFADS], Array] | None`
- Parameter freezing is configured declaratively via:
  - `trainer_conf.freeze_paths: list[str]` (dot-separated model attribute paths,
    e.g. `["noise_free"]`)
- No built-in covariance-based noise regularization is applied by default.

## Commands

```bash
uv sync

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Tests
uv run pytest
```

## Dependency / workflow notes

- `jaxtyping` is a runtime dependency used for type annotations (e.g. the
  `PyTree` type in `src/jaxfads/base.py`).
- `prek` (dev dependency) is used to run the pre-commit hooks defined in
  `.pre-commit-config.yaml` (currently `ruff-format` and `ruff`).
  Use:
  - `uv run prek install` (one-time)
  - `uv run prek run --all-files`

#### Pre-commit checklist (before any `git commit`)

- [ ] Review `git status` and `git diff`
- [ ] Update docs per the documentation hard gates above
- [ ] Ensure todos are updated (and closed if complete)
- [ ] Ensure commit is run as an isolated command (no chained commands)
- [ ] Ask for confirmation before running `git commit`
- [ ] Run quick checks when applicable:
  - `uv run prek install` (one-time)
  - `uv run prek run --all-files`
  - `uv run pytest` (when code changes under `src/`)

## Documentation conventions

- When preparing a commit that changes **behavior, APIs, configuration schemas, or examples**, update the relevant documentation in the same PR/commit series.
- At minimum, consider whether updates are needed for:
  - `README.md`
  - `AGENTS.md`
  - `docs/*.md`
  - `examples/*` comments/config snippets

## Testing conventions

- Run tests (`uv run pytest`) when code changes under `src/`.
- Avoid long-running examples (e.g. `examples/vdp_example.py`) unless explicitly
  requested.
