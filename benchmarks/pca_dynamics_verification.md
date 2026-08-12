# PCA dynamics verification

This document describes the PCA-specific verification benchmark implemented in
[`pca_dynamics_verification.py`](pca_dynamics_verification.py). It combines two
checks:

1. the limiting equivalence between the XFADS dynamics objective and MSE on PCA
   coordinates; and
2. end-to-end dynamics fitting in `NOFILT` mode with fixed PCA coordinates.

The general `NOFILT` inference contract is documented in
[`docs/algorithm.md`](../docs/algorithm.md#nofilt-inference).

## PCA setup

Let the frozen PCA readout be

$$
y_t = C z_t + b + \epsilon_t,
\qquad
\epsilon_t \sim \mathcal{N}(0, \sigma^2 I),
$$

where $C$ has full column rank. The PCA coordinate of an observation is

$$
z_t^{\mathrm{PCA}} = C^\dagger(y_t-b).
$$

The benchmark freezes the PCA readout and compares dynamics learning against
coordinate-space regression.

## ELBO/MSE equivalence

Consider fixed process noise $Q$ and let the observation noise tend to zero.
The posterior concentrates at the PCA coordinates:

$$
q(z_t) \to \delta(z_t-z_t^{\mathrm{PCA}}).
$$

The observation term is then independent of the dynamics parameters. The
dynamics-dependent part of the ELBO is the transition cross-entropy:

$$
\frac{1}{2}
\left(z_t^{\mathrm{PCA}}-f_\theta(z_{t-1}^{\mathrm{PCA}})\right)^\mathsf{T}
Q^{-1}
\left(z_t^{\mathrm{PCA}}-f_\theta(z_{t-1}^{\mathrm{PCA}})\right).
$$

Therefore:

- for $Q=I$, the objective is ordinary squared-error regression on PCA
  coordinates;
- for fixed $Q\ne I$, it is a $Q^{-1}$-weighted squared-error objective; and
- for finite observation noise, the posterior depends on the dynamics, so the
  method is not equivalent to raw PCA-coordinate MSE.

The benchmark's gradient comparison checks the limiting direction numerically
for decreasing observation-noise scales.

## NOFILT benchmark

The end-to-end benchmark uses a fixed encoder and fixed observation readout.
The encoder maps observations to PCA coordinates, while the no-filter
inference path wraps those point estimates in a small-covariance approximation.
The dynamics are then trained through the predictive objective without a
filtering recursion.

The benchmark compares:

- persistence/identity prediction;
- ordinary least-squares dynamics fitted to PCA coordinates; and
- dynamics learned through the XFADS objective.

It reports prediction RMSE and the distance between the learned linear map and
the OLS map.

## Running

From the repository root:

```bash
JAX_PLATFORMS=cpu uv run python benchmarks/pca_dynamics_verification.py
```

The benchmark uses synthetic data and fixed random seeds. Its output is
validation evidence for the PCA workflow, not a general performance guarantee
for arbitrary datasets, encoders, observation noise, or dynamics models.
