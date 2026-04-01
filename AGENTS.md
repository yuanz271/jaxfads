# JAXFADS - Project Knowledge Base

**Updated:** 2026-03-01  
**Branch:** main

## Overview

JAX library for variational Bayesian state-space modeling (XFADS). Implements
variational inference for nonlinear state-space models with exponential-family
approximate posteriors via Equinox/JAX.

## Structure

```
jaxfads/
├── src/jaxfads/                 # Core library
│   ├── base.py                  # ABCs + subclass registries (Approx/StateMap/Stepper/Observation)
│   ├── smoother.py              # XFADS orchestrator
│   ├── core.py                  # filtering/smoothing primitives
│   ├── state_maps/              # built-in latent state maps (e.g. OU, function wrapper)
│   ├── steppers.py              # Euler/RK4/discrete steppers
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
| Filtering primitives | `src/jaxfads/core.py` | `filter()`, `smooth()`, `causal()`, `nofilt()`, `expected_predictive_moment()` |
| Approx posterior families | `src/jaxfads/distributions/` | currently `MVN` |
| Observation models | `src/jaxfads/observations.py` | `GLM`, `Poisson`, `Gaussian` |
| Encoders | `src/jaxfads/encoders.py` | `AlphaEncoder`, `BetaEncoder`; user-defined for NOFILT |
| Training | `src/jaxfads/trainer.py` | `train()`, `batch_loss()`, `DEFAULT_TRAINER_CONFIG` |
| Abstract interfaces | `src/jaxfads/base.py` | `Approx`, `StateMap`, `Stepper`, `Observation`, `Encoder` |

## Core Design Notes

### Mathematical priority for contributors/agents

When implementing or reviewing changes in this repo:

- Do **not** over-index on exotic edge cases by default.
- Prioritize, in order:
  1. **Mathematical correctness** (objective, parameter transforms, inference equations)
  2. **Mathematical consistency** across modules (Approx/encoders/core/trainer contracts)
  3. **Numerical stability** (well-conditioned transforms, finite outputs, stable parameterization)
- Prefer simple, theoretically consistent formulations before adding special-case logic.
- Add edge-case handling when it is required by a concrete failure, test, or user requirement.


### Exponential-family parameter forms

- **Natural parameters** (η): flat vectors for additive filtering updates.
- **Moment parameters** (μ): flat vectors of expected sufficient statistics
  `E[T(z)]`.

### MVN approximation

`MVN(dim, rank)` — a single `rank` parameter controls the exponential-family
layout and encoder precision parameterization:

- `rank=0`: diagonal EF layout — `param_size = 2D`, `free_size = 2D`
- `rank>0`: full EF layout — `param_size = D + D²`, `free_size = 2D + D·rank`

Encoder precision: `J = diag(softplus(d)) + L @ Lᵀ` for all ranks.

Invariant for callers:
- `MVN.unpack(moment)` returns `(mean, cov)` where `cov` is **always** a full
  `(D, D)` covariance matrix (diagonal-valued when rank=0).

### Config invariants (avoiding duplication)

- `conf.state_dim` and `conf.observation_dim` are the **single source of truth**.
  `XFADS` injects these into `dyn_conf`/`obs_conf`/`enc_conf` at construction.
- `enc_conf` does **not** contain `approx` / `approx_kwargs`.
  Encoders are Approx-agnostic and only require:
  - `param_size` (natural-parameter size; injected by `XFADS`)
  - `free_size` (encoder free-form size; injected by `XFADS`)
  - `state_dim` (latent dimensionality; injected by `XFADS`)
  - `observation_dim` (injected by `XFADS`)
  - encoder hyperparameters (`width`, `depth`, `dropout`)

### Training regularization

- The trainer supports an optional user hook:
  - `trainer_conf.noise_regularizer: Callable[[XFADS], Array] | None`
- Parameter freezing is configured declaratively via:
  - `trainer_conf.freeze_paths: list[str]` (dot-separated model attribute paths,
    e.g. `["noise_free"]`)
- No built-in covariance-based noise regularization is applied by default.

## Python rules

Python-specific development rules, tooling, and workflows are defined in:

- `PYTHON.md`

Use `PYTHON.md` as the source of truth for Python commands, lint/format/test
workflows, and related conventions.

#### Docs to consider updating

- `README.md`, `AGENTS.md`, `PYTHON.md`, `docs/*.md`, `examples/*` comments/config snippets
