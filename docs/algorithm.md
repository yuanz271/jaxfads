# XFADS Algorithm

Reference: Dowling, Zhao, Park (2024). *eXponential FAmily Dynamical Systems
(XFADS): Large-scale nonlinear Gaussian state-space modeling.* NeurIPS 2024.
[arXiv:2403.01371](https://arxiv.org/abs/2403.01371)

See also: [Quickstart](quickstart.md), [Dynamics](dynamics.md),
[Training](training.md), and [Reproducibility](reproducibility.md).

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

### Transition-point propagation

The prediction expectation is delegated to the approximation family through:

```python
points, weights = approx.transition_points(key, moment, mc_size)
```

`points` has shape `(n_points, state_dim)` and `weights` has shape
`(n_points,)`. Core inference propagates the points through the latent
transition, evaluates the predictive moment at each point, and computes the
weighted average. The process-noise moment is part of the predictive-moment
operation, not the point-selection policy.

#### Monte Carlo default

The default policy draws independent samples from the distribution represented
by `moment` and assigns uniform weights:

$$
z_i \sim q(z),
\qquad
w_i = \frac{1}{S},
\qquad
S=\texttt{mc\_size}.
$$

When `mc_size <= state_dim`, the between-point covariance has rank at most
`mc_size - 1`; the resulting approximation can be poorly conditioned for a
full-dimensional predictive spread.

#### MVN unscented sigma points

`MVN` uses deterministic unscented-transform sigma points by default. For a
Gaussian posterior with mean $m$, covariance $P$, and dimension $D$:

$$
\lambda = \alpha^2(D+\kappa)-D,
\qquad
c=D+\lambda.
$$

With $L L^\mathsf{T}=P$, the points are:

$$
X_0=m,
\qquad
X_i=m+\sqrt{c}\,L_{:,i},
\qquad
X_{D+i}=m-\sqrt{c}\,L_{:,i},
$$

for $i=1,\ldots,D$, with weights:

$$
w_0=\frac{\lambda}{c},
\qquad
w_i=\frac{1}{2c}
\quad (i=1,\ldots,2D).
$$

The default parameters are `ut_alpha=1.0` and `ut_kappa=0.0`, giving
`2D+1` points, `w_0=0`, and noncentral weights `1/(2D)`. `mc_size` is
ignored when sigma points are enabled; set `use_sigma_points=False` to use
Monte Carlo. Full MVN layouts use a Cholesky covariance factor, while the
`rank=0` layout uses its diagonal covariance factor.

Non-finite per-point predictive moments are excluded and the remaining valid
weights are renormalized. If no positive valid weight remains, the predictive
moment is non-finite. Very small scaled-UT `alpha` values can cause severe
weight cancellation in float32; users overriding `ut_alpha` or `ut_kappa` are
responsible for the resulting conditioning.

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

**Low-rank structure** (Eq 19, optional in this codebase): With
`MVN(dim, rank=r)` where `r < dim`, encoders emit compact low-rank updates:

```
α_t = [a_t; A_t A_tᵀ]     where A_t ∈ ℝ^{L×r_α}
β_t = [b_t; B_t B_tᵀ]     where B_t ∈ ℝ^{L×r_β}
```

Combined: `K_t = [A_t, B_t] ∈ ℝ^{L×r}` with `r = r_α + r_β`.

With default `MVN`, encoder outputs are dense full-parameter updates.

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

In this repository, those structured linear algebra optimizations are not yet
the default compute path; see parity notes below.

## Causal Inference Network (Eq 29)

Alternative recursion that produces filtering distributions as a byproduct:

```
λ_t = F_θ(λ_{t-1} - β_t) + α_t + β_{t+1}
```

where `λ̆_t = λ_t - β_{t+1}` gives causal (filtering) estimates
`π̆(z_t) ≈ p(z_t | y_{1:t})`.

With the code indexing convention (`b_t ↔ β_{t+1}`), this is tracked as:
`λ_t = λ̆_t + b_t` and `λ̆_t = λ_t - b_t`.

In this codebase, inference-mode semantics are:

- `mode="filter"`: alpha-only filtering (`λ̆_t`)
- `mode="smooth"`: additive-site filtering (`alpha + beta`)
- `mode="causal"`: alpha-only filtering plus reconstructed smoothing natural
  parameters via `λ_t = λ̆_t + b_t` (code indexing)

### Practical Mode API

All modes return the same tuple from `XFADS.__call__`:

`(natural_params, moment_params, predictions)`

with shapes `(N, T, param_dim)`, `(N, T, param_dim)`, `(N, T, param_dim)`.

| Mode | Encoder sites used | `natural_params` meaning |
|------|--------------------|--------------------------|
| `filter` | `alpha` | filtering natural parameters (`λ̆_t`) |
| `smooth` | `alpha + beta` | smoothing-side natural parameters (`λ_t`) |
| `causal` | `alpha`, `beta` | reconstructed smoothing natural parameters (`λ_t = λ̆_t + b_t`) |

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

## Implementation Parity

For implementation parity, consistency notes, and known deviations from
arXiv:2403.01371, see:

- [`paper_parity_2403_01371.md`](paper_parity_2403_01371.md)
