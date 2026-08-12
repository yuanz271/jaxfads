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
│   ├── base.py                  # ABCs + subclass registries (Approx/Dynamics/Integrator/Observation)
│   ├── smoother.py              # XFADS orchestrator
│   ├── core.py                  # filtering/smoothing primitives
│   ├── dynamics/                # built-in latent dynamics (e.g. OU, functional wrapper)
│   ├── integrators.py              # Euler/RK4/discrete integrators
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
| Training | `src/jaxfads/training/` | `train()`, `batch_loss()`, `DEFAULT_TRAINER_CONFIG`, post-optimizer transforms |
| Abstract interfaces | `src/jaxfads/base.py` | `Approx`, `Dynamics`, `Integrator`, `Observation`, `Encoder` |

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

### Research-code implementation style

This is research code. Favor mathematical clarity, algorithmic fluency, and
readability over defensive abstraction or speculative compatibility:

- Keep core algorithms pure, direct, and minimal. Express the mathematical
  operation at the point where it is used rather than introducing wrappers,
  fallbacks, or layers with no concrete second use.
- Validate inputs at public boundaries and configuration resolution. Inside
  numerical kernels and JAX-transformed code, rely on documented input
  contracts and natural Python/JAX failures unless a guard is required for
  numerical stability, data integrity, or a demonstrated failure mode.
- Do not add speculative validation, fallback behavior, compatibility
  scaffolding, or general-purpose extension mechanisms without a concrete
  caller, requirement, or test.
- Prefer focused tests for mathematical identities, invariants, conditioning,
  numerical stability, and scientific behavior. Each test should distinguish a
  required behavior or regression rather than exercise implementation detail.


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

- The trainer supports an optional user hook passed as an argument (not config,
  since it is a callable):
  - `train(model, data, *, conf, on_epoch_end=None, regularizer=None,
    optimizer=None)` where `regularizer: Callable[[XFADS], Array] | None` adds a
    scalar penalty to the per-batch objective (`loss = -ELBO +
    regularizer(model)`).
  - `batch_loss` stays a pure objective; the trainer composes the penalty.
  - Penalize the intended quantity in its own space (e.g. decode
    `model.noise` through `model.approx` to regularize Q; do not penalize the
    raw free parameters).
- Optimizer policy is user-ownable. The default optimizer applies **no weight
  decay**: in a plugin framework the trainer cannot know which leaves are
  weight matrices vs variances/biases, so it makes no decay assumption. Pass
  `optimizer=` (an `optax.GradientTransformation`) to take full control, e.g.
  `optax.add_decayed_weights(wd, mask=...)` with a model-derived mask. There is
  no `weight_decay` config field.
- Parameter freezing is configured declaratively via:
  - `trainer_conf.freeze_paths: list[str]` (dot-separated model attribute paths,
    e.g. `["noise"]`); applied on top of the default or a supplied
    optimizer.
- M-step policy is trainer-owned and serializable through ordered
  `trainer_conf.post_optimizer_transforms` entries. Each entry has a symbolic
  name and child settings; `gaussian_observation` and `mvn_noise` are the
  built-in R/Q policies. `mvn_noise` owns Q initialization, fractional Q
  update, and `noise` freezing. Runtime `post_optimizer_transforms` objects
  remain the extension point for custom transformations. The default configured
  list enables both built-ins; an explicit empty list is optimizer-only
  training.
- Each batch uses one pre-SGD inference forward for loss, gradients, and the
  selected plugins' statistics; plugins update the post-SGD model with no
  extra inference pass. Model/component M-step APIs are unsupported.
- `state_noise`, `noise_prior`, and `noise_prior_dof` are unsupported. Q
  policy belongs to the serializable trainer configuration.
- No built-in covariance-based noise regularization is applied by default.

## Testing Workflow

- Run the full test suite (`pytest tests/`) **only** right before pushing, or
  when explicitly asked — not automatically after every commit/edit.
- During iteration, scope test runs to the files that exercise the changed
  code (e.g. `pytest tests/test_trainer.py tests/test_observations.py` for
  `trainer.py`/`observations.py` changes; include `tests/test_smoother.py`
  for changes touching `base.py`'s `Observation`/`Approx`/`Dynamics` ABCs,
  since `smoother.py`/`XFADS` sits directly on top of them).
- Do not run a scoped set and then the full suite back-to-back for the same,
  unchanged code — that pays both costs for no additional information. If a
  full-suite run is warranted (about to push, or asked for), it replaces the
  scoped run for that checkpoint, it doesn't follow it.

## Python rules

Python-specific development rules, tooling, and workflows are defined in:

- `PYTHON.md`

Use `PYTHON.md` as the source of truth for Python commands, lint/format/test
workflows, and related conventions.

#### Docs to consider updating

- `README.md`, `AGENTS.md`, `PYTHON.md`, `docs/*.md`, `examples/*` comments/config snippets
