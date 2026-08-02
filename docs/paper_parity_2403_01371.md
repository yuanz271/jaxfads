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
| Approximate ELBO (expected log-likelihood minus KL to predictive) | `MATCH` | `src/jaxfads/vi.py:16`, `src/jaxfads/vi.py:84`, `src/jaxfads/vi.py:85` | Matches Eq. 17 structure; trainer supports KL warmup via an optax schedule built from `conf.kl_warmup_steps` and passed as the `beta` KL weight to `batch_loss` (`src/jaxfads/trainer.py`, `beta_schedule`). |
| Alpha/Beta encoder architecture for pseudo-observation parameters | `MATCH` | `src/jaxfads/encoders.py:30`, `src/jaxfads/encoders.py:79`, `src/jaxfads/smoother.py:408`, `src/jaxfads/smoother.py:417`, `src/jaxfads/smoother.py:423` | Feedforward alpha + reverse-time recurrent beta, then additive composition into update sites. |
| Missing-observation handling via zero update | `MATCH` | `src/jaxfads/smoother.py:401`, `src/jaxfads/smoother.py:410` | Non-finite observations are masked and produce zero local update. |
| Low-rank pseudo-observation parameter output from encoders | `PARTIAL` | `src/jaxfads/distributions/mvn.py` (`MVN.free_to_natural`); smoke test in `tests/test_smoother.py:186` | Unified via `MVN(dim, rank=r)`, but `h` is emitted independently of `J` (see detail below). Paper Eq. 19 couples them via `h = Kᵀb`, `J = KᵀK`. |
| Low-rank structured linear algebra complexity claims (Woodbury/Cholesky pipeline) | `MISMATCH` | `MVN.natural_to_moment`, `MVN.moment_to_natural`, `MVN.unpack`, `MVN.kl` in `src/jaxfads/distributions/mvn.py` | Current inference/kl/sample paths materialize full covariance/precision operations (TFP full-cov + dense solves), so paper's scalable structured complexity is not realized end-to-end. |
| Streaming/causal inference recursion (paper Eq. 29 family) | `MATCH` | `src/jaxfads/core.py` (`causal`), `src/jaxfads/smoother.py` (`mode="causal"` branch) | Implemented as alpha-only filtering for `λ̆_t` followed by reconstruction `λ_t = λ̆_t + b_t` (code indexing, where `b_t` corresponds to paper `β_{t+1}`). The API also exposes `mode="filter"` for alpha-only filtering output directly. |
| Generative-model parameter learning strategy (θ, ψ, including state-noise `Q_θ` and the observation model's own noise) | `MATCH` | `src/jaxfads/trainer.py` (`train`, gradient descent on the full model via `eqx.filter_value_and_grad`) | The paper explicitly considers classical vEM (alternating E-step/M-step over the approximate posterior and generative-model parameters) and rejects it for scalability, adopting VAE-style joint gradient-based training of θ/ψ/φ via stochastic backpropagation instead (paper: "vEM can be slow due to the need to fully optimize the variational parameters before taking gradient steps on parameters of the generative model... the VAE is better suited for large scale problems for its ability to simultaneously learn the generative model and inference network"). `train()` defaults to unconstrained gradient descent over the whole model, including `noise.free` and observation-noise parameters; no M-step/vEM machinery is used unless explicit trainer plugins are selected. |

## Consistency Details

- The central VI algorithmic loop is consistent: encoder-derived pseudo-observation parameters are added in natural space after predictive-moment propagation.
- The code keeps the paper's exponential-family framing explicit (`Approx` abstraction + Gaussian natural/moment mechanics).
- The training objective uses the same ELBO decomposition with optional annealing.

## Inconsistency Details

- The paper's major scalability contribution (structured low-rank matrix identities through filtering and KL computation) is not implemented as the main compute path.
- The low-rank encoder parameterization (`MVN(dim, rank=r)`) is a compact layer, but downstream operations still use dense Gaussian algebra.
- Indexing difference alone (`β_t` vs `β_{t+1}`) does not imply mismatch; this implementation documents the code convention `b_t ↔ β_{t+1}` and checks recurrence form.
- The unified `free_to_natural` emits `h` independently of `J = diag(softplus(d)) + L Lᵀ`. Paper Eq. 19 constrains the linear natural parameter via `h = Kᵀb` so that both `h` and `J = KᵀK` are determined by the same low-rank factor `K` and shift `b`. The current implementation decouples them: the encoder emits `h` as a free vector and constructs `J` separately from a diagonal baseline plus `L Lᵀ`. This gives the encoder strictly more degrees of freedom per update site; the diagonal baseline also prevents unbounded posterior means (`J⁻¹h` stays bounded even when `L ≈ 0`).

## Documented Extensions Beyond the Paper

These are opt-in library capabilities, not required for parity and not
claimed by the paper -- listed separately from Inconsistency Details
above so they aren't mistaken for gaps or failures to replicate the
paper's method.

- **Closed-form, non-SGD updates for observation noise (`R`) and
  transition/process noise (`Q`)**: the trainer-owned
  `GaussianObservationMstep` and `MVNNoiseMstep` provide an opt-in alternative
  to the paper's joint-gradient-descent treatment of `R`/`Q`. `R` and `Q` are
  independently selected through `train(..., post_optimizer_transforms=...)`; Q initialization
  and prior shrinkage are owned by `MVNNoiseMstep(q_scale=...,
  q_prior_fraction=...)`. This is
  narrower than the classical vEM the paper considers and rejects: only
  `R`/`Q` are ever updated this way, never `θ`/`ψ`/`φ` as a whole, and
  gradient descent on the remaining parameters continues throughout
  (`Q`, in particular, stays "in the loss" -- see
  `docs/mstep_dynamics_noise.md`'s "Required safeguards" -- scaling `f`'s
  gradient by `1/Q` even while `Q` itself is closed-form-updated, not
  decoupled the way a literal M-step's generative-model-only phase would
  imply).
- **Motivation, not paper fidelity**: these mechanisms exist because
  joint gradient descent on `R`/`Q` was empirically found to have real
  failure modes in this codebase's own testing -- a Heywood-case
  degeneracy for `R` (a component's fitted covariance reaching the
  numerical floor while its true residual variance was ~10⁶x larger, a
  known failure mode of factor-analysis-structured joint MLE, documented
  in `Gaussian.mstep_stat`'s docstring) and, for `Q`, an empirically
  dominant overestimation tendency under joint MLE plus better
  downstream dynamics-accuracy from MAP-shrunk alternating updates
  (`docs/mstep_dynamics_noise.md`'s Design section, both known-`z` and
  latent-`z` Lorenz experiments). The paper does not report or address
  either failure mode; these mechanisms are this codebase's own response
  to them, not a paper-parity gap.

## Overall Assessment

Implementation parity is strong for the core XFADS variational filtering objective, encoder-driven pseudo-observation inference, Eq. 29 causal recursion, and the paper's own choice of joint gradient-based (not classical vEM) parameter learning; the remaining major gap is the paper's scalability-focused structured linear algebra path. This codebase additionally offers opt-in, closed-form `R`/`Q` update mechanisms beyond anything the paper describes (see Documented Extensions above), motivated by failure modes found in this codebase's own testing, not by a paper claim.
