# src/jaxfads - Core Library

## Overview

Core XFADS implementation: filtering, smoothing, neural parameterizations, and training.

## Module Map

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

## Architecture

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

## Class Hierarchy

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

## Conventions (This Module)

- All modules use Equinox for PyTree-compatible neural layers
- Natural/moment parameterization pattern throughout distributions
- `__call__` typically wraps forward computation
- `set_static()` marks params non-trainable (e.g., readout freezing)

## Critical Constants

| Constant | Location | Purpose |
|----------|----------|---------|
| `MAX_LOGRATE` | `observations.py:33` | Clip log-rate for numerical stability |
| `MAX_EXP` | `nn.py:53` | Exp overflow guard |
| `EPS` | `nn.py:54` | Numerical epsilon |

## Adding New Components

**New observation model**:
1. Subclass `Likelihood` in `observations.py`
2. Implement `eloglik()` for expected log-likelihood
3. Register in `__init__.py` exports

**New distribution family**:
1. Subclass `Approx` in `distributions.py`
2. Implement natural↔moment conversions, sampling, KL
3. Add tests in `tests/test_distribution.py`
