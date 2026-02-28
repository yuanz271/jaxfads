# XFADS Algorithm

Reference: Dowling, Zhao, Park (2024). *eXponential FAmily Dynamical Systems
(XFADS): Large-scale nonlinear Gaussian state-space modeling.* NeurIPS 2024.
[arXiv:2403.01371](https://arxiv.org/abs/2403.01371)

## Problem Setting

State-space model with latent states `z_t ∈ ℝ^L` and observations `y_t`:

```
p(y_{1:T}, z_{1:T}) = p(z_1) ∏_t p(y_t | z_t) p(z_t | z_{t-1})
```

The dynamics are **nonlinear Gaussian** (Eq 1 of paper):

```
p(z_t | z_{t-1}) = N(z_t | m_θ(z_{t-1}), Q_θ)
```

where `m_θ` is a neural network and `Q_θ` is learnable state noise.

## Exponential Family Formulation

The dynamics are written in exponential family form with sufficient
statistics `T(z) = [z, -½ zzᵀ]`:

- **Natural parameters** (Eq 3): `λ_θ(z_{t-1}) = [Q⁻¹ m_θ(z_{t-1}), -½ Q⁻¹]`
- **Moment parameters** (Eq 4): `μ_θ(z_{t-1}) = [m_θ(z_{t-1}), -½(m_θ m_θᵀ + Q)]`

## Variational Approximation

Instead of parameterizing `q(z_t | z_{t-1})` directly (as in DKF/DVBF),
XFADS uses **pseudo-observations** — Gaussian potentials that encode data
(Eq 9):

```
p(ỹ_t | z_t) ∝ exp(λ̃_t^T T(z_t)) = exp(k_t^T z_t - ½ ||K_t z_t||²)
```

These are combined with the prior via Bayes' rule (Eq 10):

```
q(z_{1:T}) = ∏ p(ỹ_t | z_t) · p_θ(z_{1:T}) / p(ỹ_{1:T})
```

This imposes the latent dependency structure of the generative model onto
the posterior, unlike black-box approaches.

## Variational Filtering (Core Algorithm)

Two-step recursion (Eqs 12-13):

### Step 1 — Variational Predict (Eq 12)

Moment-matching (forward KL minimization):

```
μ̄_t = E_{π(z_{t-1})}[μ_θ(z_{t-1})]

where the *predictive moment function* (Eq. 4) is

μ_θ(z_{t-1}) = E_{p(z_t|z_{t-1})}[T(z_t)]
```

Approximated via reparameterization trick with `S` samples (Eq 22):

```
μ̄_t ≈ (1/S) Σ_s μ_θ(z^s_{t-1})     where z^s_{t-1} ~ π(z_{t-1})

In code this is implemented by `core.expected_predictive_moment`:
- sample `z_{t-1}^s ~ π(z_{t-1})`
- compute dynamics mean `f(z_{t-1}^s)`
- compute conditional moments via `approx.predictive_moment(f(z_{t-1}^s), noise)`
- average across samples
```

This gives predictive mean `m̄_t = (1/S) Σ m_θ(z^s)` and covariance
`P̄_t = M̄_c M̄_cᵀ + Q` (Eq 23), where `M̄_c` is the centered sample
matrix — a natural low-rank structure.

### Step 2 — Variational Update (Eq 13)

Conjugate Bayes' rule in natural parameter space:

```
λ_t = λ̄_t + λ̃_t
```

Simply adds the pseudo-observation natural parameters to the predictive
natural parameters. Exact because pseudo-observations are conjugate
Gaussian potentials.

## Smoothing as Filtering (Key Insight)

Rather than running a separate backward pass, XFADS defines each
pseudo-observation to depend on **current and future** data (Eq 15):

```
p(ỹ_t | z_t) ∝ exp(λ̃_ϕ(y_{t:T})^T T(z_t))
```

Then filtered marginals approximate smoothed marginals:
`π(z_t) ≈ p(z_t | y_{1:T})`.

This yields the approximate ELBO (Eq 17):

```
L̂(π) = Σ_t [ E_{π_t}[log p(y_t | z_t)] - KL(π(z_t) || π̄(z_t)) ]
```

The KL is between the posterior and one-step predictive — both Gaussian
— so it's analytic.

## Encoder Architecture (Eqs 18-21)

The pseudo-observation parameters decompose into two additive components:

```
λ̃_t = α_t + β_{t+1}
```

Code indexing convention:
- The implementation uses a beta tensor `b_t` at step `t` that corresponds to
  paper `β_{t+1}`. This is an indexing convention, not by itself an algorithmic
  mismatch.

- **Alpha (local encoder)**: `α_t = NN(y_t)` — feedforward, captures
  instantaneous information from current observation
- **Beta (backward encoder)**: `β_t = S2S(β_{t+1}, α_t)` — GRU running
  backward in time over alpha outputs, captures future information

**Low-rank structure** (Eq 19): Both encoders output low-rank natural
parameter updates:

```
α_t = [a_t; A_t A_tᵀ]     where A_t ∈ ℝ^{L×r_α}
β_t = [b_t; B_t B_tᵀ]     where B_t ∈ ℝ^{L×r_β}
```

Combined: `K_t = [A_t, B_t] ∈ ℝ^{L×r}` with `r = r_α + r_β`.

**Missing data**: Setting `α_t = 0` when `y_t` is missing means the prior
is not updated — a principled handling without special logic.

## Efficient Implementation

The low-rank structure enables `O(L(Sr + S² + r²))` per time step instead
of `O(L³)`:

- **Prediction**: `P̄_t = M̄_c M̄_cᵀ + Q` — never materialized; Woodbury
  identity for `P̄⁻¹` operations (Eq 50)
- **Update**: `P⁻¹_t = P̄⁻¹_t + K_t K_tᵀ` — structured via Cholesky of
  `r×r` matrix `Υ_t` (Eq 52)
- **Sampling**: Uses Cong et al. trick (Eq 49) to sample without
  materializing `P_t`
- **KL**: Log-determinant and trace computed via low-rank factors (App C.1)

## Causal Inference Network (Eq 29)

Alternative recursion that produces filtering distributions as a byproduct:

```
λ_t = F_θ(λ_{t-1} - β_t) + α_t + β_{t+1}
```

where `λ̆_t = λ_t - β_{t+1}` gives causal (filtering) estimates
`π̆(z_t) ≈ p(z_t | y_{1:t})`.

With the code indexing convention (`b_t ↔ β_{t+1}`), this is tracked as:
`λ_t = λ̆_t + b_t` and `λ̆_t = λ_t - b_t`.

## End-to-End Learning (Algorithm 1)

```
while not converged:
    # Backward pass: encode observations
    for t = T to 1:
        α_t = NN(y_t)
        β_t = S2S(β_{t+1}, α_t)
        k_t = a_t + b_t;  K_t = [A_t, B_t]

    # Forward pass: variational filtering (Algorithm 2)
    z_{1:T}, m_{1:T}, m̄_{1:T}, Υ_{1:T} = filter(k_{1:T}, K_{1:T})

    # ELBO
    L̂ = Σ_t [ (1/S) Σ_s log p(y_t | z^s_t) - KL(π_t || π̄_t) ]

    # Gradient step on all parameters
    (ϕ, θ, ψ) ← (ϕ, θ, ψ) - ∇L̂
```

---

# Implementation Review

## Mapping: Paper → Codebase

| Paper Concept | Code Location | Notes |
|---------------|---------------|-------|
| Generative model `p(z_t \| z_{t-1})` | `Dynamics.forward()` in `base.py` | Deterministic transition `m_θ(z, u, c)`; process noise owned by `XFADS` |
| State noise `Q_θ` | `XFADS.noise_free` in `smoother.py` | Stored as free-form; constrained via `approx.free_to_canon → canon_to_moment` |
| Observation model `p(y_t \| z_t)` | `Observation.eloglik()` in `observations.py` | Poisson and Gaussian implementations |
| Pseudo-observation `λ̃_t` | `alpha + beta` computed in `XFADS.__call__()` | Added in natural parameter space; code `b_t` corresponds to paper `β_{t+1}` |
| Local encoder `α_t = NN(y_t)` | `AlphaEncoder` in `encoders.py` | MLP mapping observations → natural params |
| Backward encoder `β_t = S2S(β_{t+1}, α_t)` | `BetaEncoder` in `encoders.py` | GRU running backward over alpha outputs |
| Variational predict (Eq 12) | `expected_predictive_moment()` in `core.py` | MC approximation of `E[μ_θ(z_{t-1})]` |
| Variational update (Eq 13) | `nature_t = nature_p_t + a_t` in `core.filter()` | Additive natural parameter update |
| Approximate ELBO (Eq 17) | `elbo()` in `vi.py` | `E[log p(y\|z)] - β·KL(π \|\| π̄)` |
| KL(π ∥ π̄) | `approx.kl()` on `MVN` | Via TFP `MultivariateNormalFullCovariance` |
| Exp-family natural params | `Approx.moment_to_natural()` | Flat array; layout defined by MVN |
| Exp-family moment params | `Approx.natural_to_moment()` | Flat array `[E[z], E[-½ zzᵀ]_flat]` |
| MVN implementation | `MVN(dim, structure=...)` | Supports `structure="full"` and `structure="diag"` |
| Causal/streaming inference (Eq 29) | Implemented via `mode="causal"` | Uses alpha-only filtering for `λ̆_t` and reconstructs `λ_t = λ̆_t + b_t` (code indexing) |
| Efficient structured ops (App B.5) | Not implemented | Current impl materializes full covariance |

## Faithful Implementations

1. **Variational filtering recursion** (`core.filter`): Correctly implements
   the predict-update loop. Predict via MC sampling of dynamics
   (`expected_predictive_moment` → Eq 12), update via additive natural params
   (Eq 13).

2. **Encoder architecture** (`encoders.py`): Alpha encoder is feedforward
   MLP mapping `y_t → α_t` (Eq 21). Beta encoder is GRU running backward
   over alpha outputs `α_{T:1} → β_{1:T}` (Eq 21). Additive decomposition
   `λ̃_t = α_t + β_{t+1}` in smoother's `__call__`.

3. **Missing data handling** (`smoother.py`): Sets `α_t = 0` for non-finite
   observations (Eq 18 discussion). Additional dropout masking via
   `DataMasker` for pseudo-missing data.

4. **ELBO** (`vi.py`): `E[log p(y|z)] - β·KL(π || π̄)` matches Eq 17.
   KL warm-up via `beta` parameter.

5. **Exponential family design** (`Approx` ABC, `MVN`): Natural/mean
   parameterization, conversions, sampling — matches the paper's
   exponential family framework (Eqs 2-4).

## Simplifications / Deviations

1. **No structured linear algebra**: The paper's key efficiency contribution
   (App B.5, O(L(Sr + S² + r²)) per step) is not implemented. The current
   code materializes full covariance matrices via TFP, giving O(L³) cost.
   This limits scalability to large `L`.

2. **Causal indexing caveat**: For Eq 29, paper notation uses `β_{t+1}` while
   code uses `b_t` at the same implementation step. Parity should be checked by
   recurrence equivalence, not by raw index labels alone.

3. **Noise ownership**: Paper has `Q_θ` as part of the dynamics. Codebase
   separates: `Dynamics` is purely deterministic `m_θ(z, u, c)`; process noise
   is owned by `XFADS` as `noise_free` (constrained via the `Approx`).

4. **Low-rank encoder output is optional**: The paper specifies low-rank
   precision updates in pseudo-observations. The codebase now supports this via
   `LoRaMVN` compact encoder outputs, but the default `MVN` path remains dense.
   See `docs/paper_parity_2403_01371.md` for a full parity matrix.

5. **`predictive_moment` averaging**: The paper averages in mean (moment) parameter
   space (Eq 22). The code does this correctly via `predictive_moment` on `MVN`,
   which computes moment parameters (expected sufficient statistics) `E[T(z)]`
   per sample, averages, then converts back.

6. **KL computation**: Paper derives efficient O(LSr + LS² + Lr²) KL
   evaluation (App C.1) using structured factors. Code uses TFP's
   generic `MultivariateNormalFullCovariance` KL, which is O(L³).

## Summary

The codebase is a faithful implementation of the XFADS algorithm's
mathematical framework: variational filtering with pseudo-observations,
exponential family parameterization, additive natural parameter updates,
and the encoder architecture. The main gap is the paper's structured
linear algebra optimizations for large `L` — the current implementation
materializes full covariance matrices, making it correct but O(L³) rather
than the paper's O(L(Sr + S² + r²)).
