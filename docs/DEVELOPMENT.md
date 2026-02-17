# Development Guide

## Overview

JAX library for variational Bayesian state-space modeling (XFADS). Exponential-family dynamical systems with neural parameterizations via Equinox.

## Structure

```
jaxfads/
├── src/jaxfads/      # Core library
├── tests/            # Unit tests mirroring src/ structure
└── pyproject.toml    # Build config, deps, ruff settings
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Filtering/smoothing algorithms | `src/jaxfads/core.py` | `filter()`, `bismooth()` |
| Main model class | `src/jaxfads/smoother.py` | `XFADS` orchestrator |
| State transitions | `src/jaxfads/dynamics.py` | `Dynamics`, `DiagGaussian` |
| Observation models | `src/jaxfads/observations.py` | `Poisson`, `DiagGaussian` |
| Distribution math | `src/jaxfads/distributions.py` | `DiagMVN`, `FullMVN`, `Approx` |
| Neural blocks | `src/jaxfads/nn.py`, `encoders.py` | MLP, RBF, encoders |
| Training loop | `src/jaxfads/trainer.py` | `train()`, `batch_elbo()` |
| Add tests | `tests/test_<module>.py` | Use `conftest.py` fixtures |

## Core Library Notes (`src/jaxfads`)

### Module Map

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `core.py` | Filtering primitives | `filter()`, `bismooth()`, `Mode` |
| `smoother.py` | Main orchestrator | `XFADS` class |
| `dynamics.py` | State transitions | `Dynamics`, `predict_moment()`, `sample_expected_moment()` |
| `observations.py` | Likelihoods | `Poisson`, `DiagGaussian`, `Likelihood` |
| `distributions.py` | Approx posteriors | `DiagMVN`, `FullMVN`, `LoRaMVN`, `Approx` |
| `nn.py` | Neural blocks | `make_mlp()`, `WeightNorm`, `RBFN`, `DataMasker` |
| `encoders.py` | Sequence encoders | `AlphaEncoder`, `BetaEncoder` |
| `trainer.py` | Training loop | `train()`, `batch_elbo()`, `batch_loss()` |
| `vi.py` | Variational inference | `elbo()` |
| `util.py` | Helpers | `vmap_with_key()` |
| `constraints.py` | Parameter constraints | - |
| `ilqr.py` | Iterative LQR | - |

### Architecture

```
XFADS (smoother.py)
├── forward: Dynamics        # State transition model
├── likelihood: Likelihood   # Observation model (Poisson/Gaussian)
├── alpha_encoder            # Forward pass encoder
├── beta_encoder             # Backward pass encoder (GRU-based)
└── approx: Approx           # Posterior family (DiagMVN/FullMVN)

Filtering (core.py)
├── filter()     # Forward filtering with Mode.PSEUDO or Mode.BIFILTER
└── bismooth()   # Bidirectional smoothing
```

### Class Hierarchy

**Distributions** (`distributions.py`):
```
Approx (ABC)
├── FullMVN      # Full covariance MVN
├── LoRaMVN      # Low-rank + diagonal MVN
└── DiagMVN      # Diagonal covariance MVN
```

**Observations** (`observations.py`):
```
Likelihood (ABC)
├── Poisson      # Poisson with neural readout
└── DiagGaussian # Gaussian with learned noise
```

**Dynamics** (`dynamics.py`):
```
Noise (ABC)
└── DiagGaussian # Diagonal state noise

Dynamics        # Transition model with forward()
```

### Conventions (Core Library)

- All modules use Equinox for PyTree-compatible neural layers
- Natural/moment parameterization pattern throughout distributions
- `__call__` typically wraps forward computation
- `set_static()` marks params non-trainable (e.g., readout freezing)

### Critical Constants

| Constant | Location | Purpose |
|----------|----------|---------|
| `MAX_LOGRATE` | `observations.py:33` | Clip log-rate for numerical stability |
| `MAX_EXP` | `nn.py:53` | Exp overflow guard |
| `EPS` | `nn.py:54` | Numerical epsilon |

### Adding New Components

**New observation model**:
1. Subclass `Likelihood` in `observations.py`
2. Implement `eloglik()` for expected log-likelihood
3. Register in `__init__.py` exports

**New distribution family**:
1. Subclass `Approx` in `distributions.py`
2. Implement natural↔moment conversions, sampling, KL
3. Add tests in `tests/test_distribution.py`

## Commands

```bash
# Environment
uv sync                              # Install deps

# Testing
uv run pytest                        # Full suite
uv run pytest tests/test_smoother.py -k "test_name"  # Specific

# Linting
uv run ruff check .                  # Check
uv run ruff check . --fix            # Auto-fix
```

## Conventions

- **Style**: PEP 8, 4-space indent, type annotations on public APIs
- **Docstrings**: NumPy format (`Parameters`, `Returns`, `Notes`) with shapes/dtypes
- **Imports**: stdlib → third-party → local
- **Naming**: `snake_case` functions/vars, `PascalCase` classes
- **Ruff ignores**: E501 (line length), F722 (forward refs)

## Branching Practices

- Branch from `main` for every change.
- Keep branches focused and short-lived.
- Open a pull request before merging.
- Rebase or merge latest `main` before final review.
- Squash or tidy commits before merge if requested.

## Anti-Patterns

- **NO** `as any`, `@ts-ignore` equivalents for type suppression
- **NO** empty exception handlers
- **NO** committing without running `uv run ruff check .`

## Testing Conventions

- Test files: `tests/test_<module>.py`
- Fixtures: Use `spec` from `conftest.py` for shared params
- Stochastic tests: Use `chex.assert_trees_all_close` with tolerances
- Randomness: Always `jax.random.PRNGKey(seed)` for reproducibility
- I/O tests: Use `tempfile.TemporaryDirectory`

## Dependencies

| Package | Purpose |
|---------|---------|
| `jax` | Autodiff, accelerators |
| `equinox` | Neural modules (PyTree-based) |
| `optax` | Optimizers |
| `tensorflow-probability[jax]` | Probability distributions |
| `omegaconf` | Config management |
| `gearax` | GitHub dependency pinned in `pyproject.toml` |

## Security Notes

- Dependabot reports a minor advisory for `fonttools` **4.59.0**, pulled in via the dev dependency `matplotlib`.
- This impacts development tooling only; no runtime dependency for the library.

## Known Issues

- TODO in `core.py:100`: Prior placement undecided (approx vs dynamics)
- No CI/CD pipelines yet (manual `pytest`/`ruff`)
- Several modules lack test coverage (`vi.py`, `constraints.py`, `encoders.py`, `observations.py`, `util.py`, `core.py`)
