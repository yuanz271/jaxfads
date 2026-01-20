# JAXFADS - Project Knowledge Base

**Generated:** 2026-01-20  
**Commit:** 0597b52  
**Branch:** main

## Overview

JAX library for variational Bayesian state-space modeling (XFADS). Exponential-family dynamical systems with neural parameterizations via Equinox.

## Structure

```
jaxfads/
├── src/jaxfads/      # Core library (see src/jaxfads/AGENTS.md)
├── tests/            # Unit tests mirroring src/ structure
├── gearax/           # Workspace submodule (editable dependency)
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
| `gearax` | Workspace component (local editable) |

## Known Issues

- TODO in `core.py:100`: Prior placement undecided (approx vs dynamics)
- No CI/CD pipelines yet (manual `pytest`/`ruff`)
- Several modules lack test coverage (`vi.py`, `constraints.py`, `encoders.py`, `observations.py`, `util.py`, `core.py`)
