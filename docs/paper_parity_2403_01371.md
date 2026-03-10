# Parity Review: arXiv:2403.01371 (XFADS)

Reference paper: Dowling, Zhao, Park (2024), *eXponential FAmily Dynamical Systems (XFADS)*, arXiv:2403.01371.

This document evaluates implementation parity between the paper and this codebase.

Legend:
- `MATCH`: behavior is implemented consistently with the paper's method.
- `PARTIAL`: core idea is present, but with constraints/simplifications.
- `MISMATCH`: paper feature/claim is absent or materially different.

## Parity Matrix

| Paper component | Status | Code evidence | Notes |
|---|---|---|---|
| Variational filtering recursion (predict + additive natural update) | `MATCH` | `src/jaxfads/core.py:23`, `src/jaxfads/core.py:121`, `src/jaxfads/core.py:190`, `src/jaxfads/core.py:191` | Implements Eq. 12-style Monte Carlo predictive moment and Eq. 13 additive natural update. |
| Exponential-family natural/moment parameterization for Gaussian approx | `MATCH` | `src/jaxfads/distributions/mvn.py:252`, `src/jaxfads/distributions/mvn.py:268` | Uses Gaussian sufficient-statistics representation and conversions between natural and moment forms. |
| Approximate ELBO (expected log-likelihood minus KL to predictive) | `MATCH` | `src/jaxfads/vi.py:16`, `src/jaxfads/vi.py:84`, `src/jaxfads/vi.py:85` | Matches Eq. 17 structure; trainer supports KL warmup (`beta`) in `src/jaxfads/trainer.py:192`. |
| Alpha/Beta encoder architecture for pseudo-observation parameters | `MATCH` | `src/jaxfads/encoders.py:30`, `src/jaxfads/encoders.py:79`, `src/jaxfads/smoother.py:408`, `src/jaxfads/smoother.py:417`, `src/jaxfads/smoother.py:423` | Feedforward alpha + reverse-time recurrent beta, then additive composition into update sites. |
| Missing-observation handling via zero update | `MATCH` | `src/jaxfads/smoother.py:401`, `src/jaxfads/smoother.py:410` | Non-finite observations are masked and produce zero local update. |
| Low-rank pseudo-observation parameter output from encoders | `MATCH` | `src/jaxfads/distributions/mvn.py` (`MVN.free_to_natural`); smoke test in `tests/test_smoother.py:186` | Unified via `MVN(dim, rank=r)`. Encoder precision: `J = diag(softplus(d)) + L Lᵀ`. All ranks share the same code path. |
| Low-rank structured linear algebra complexity claims (Woodbury/Cholesky pipeline) | `MISMATCH` | `src/jaxfads/distributions/mvn.py:295`, `src/jaxfads/distributions/mvn.py:305`, `src/jaxfads/distributions/mvn.py:258`, `src/jaxfads/distributions/mvn.py:274` | Current inference/kl/sample paths materialize full covariance/precision operations (TFP full-cov + dense solves), so paper's scalable structured complexity is not realized end-to-end. |
| Streaming/causal inference recursion (paper Eq. 29 family) | `MATCH` | `src/jaxfads/core.py` (`causal`), `src/jaxfads/smoother.py` (`mode="causal"` branch) | Implemented as alpha-only filtering for `λ̆_t` followed by reconstruction `λ_t = λ̆_t + b_t` (code indexing, where `b_t` corresponds to paper `β_{t+1}`). The API also exposes `mode="filter"` for alpha-only filtering output directly. |

## Consistency Details

- The central VI algorithmic loop is consistent: encoder-derived pseudo-observation parameters are added in natural space after predictive-moment propagation.
- The code keeps the paper's exponential-family framing explicit (`Approx` abstraction + Gaussian natural/moment mechanics).
- The training objective uses the same ELBO decomposition with optional annealing.

## Inconsistency Details

- The paper's major scalability contribution (structured low-rank matrix identities through filtering and KL computation) is not implemented as the main compute path.
- The low-rank encoder parameterization (`MVN(dim, rank=r)`) is a compact layer, but downstream operations still use dense Gaussian algebra.
- Indexing difference alone (`β_t` vs `β_{t+1}`) does not imply mismatch; this implementation documents the code convention `b_t ↔ β_{t+1}` and checks recurrence form.
- The unified `free_to_natural` emits `h` independently of `J = diag(softplus(d)) + L Lᵀ`; the diagonal baseline prevents unbounded posterior means.

## Overall Assessment

Implementation parity is strong for the core XFADS variational filtering objective, encoder-driven pseudo-observation inference, and Eq. 29 causal recursion; the remaining major gap is the paper's scalability-focused structured linear algebra path.
