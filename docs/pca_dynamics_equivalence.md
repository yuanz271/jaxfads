# PCA Dynamics Estimation: ELBO vs MSE Equivalence

This note proves that learning dynamics via the XFADS ELBO with PCA as the
observation model recovers MSE on principal components in the appropriate limit.

## Setup

Generative model with frozen PCA readout:

$$z_t \sim \mathcal{N}(f_\theta(z_{t-1}), Q), \quad y_t \sim \mathcal{N}(Cz_t + b,\; \sigma^2 I)$$

where $C$ (loadings) and $b$ (mean) are frozen PCA parameters.

PCA projection: $z_t^{\mathrm{PCA}} = C^\dagger(y_t - b)$.

**Assumption.** $C$ has full column rank (state_dim $\le$ obs_dim), so $C^\dagger = (C^\top C)^{-1} C^\top$ is well-defined. This holds for standard PCA.

**Two approaches to learning $f_\theta$:**

1. **XFADS ELBO** with MVN posterior and Bayesian filtering.
2. **MSE on PCA coordinates**: $\mathcal{L}^{\mathrm{MSE}} = \frac{1}{2}\sum_t \|z_t^{\mathrm{PCA}} - f_\theta(z_{t-1}^{\mathrm{PCA}})\|^2$.

## Claim

When $Q$ is fixed and $\sigma^2 \to 0$, the dynamics estimate from the XFADS
ELBO exactly agrees with MSE on PCA coordinates.

## Proof

### ELBO in the $\sigma^2 \to 0$ limit

As $\sigma^2 \to 0$ the observation evidence dominates and the posterior
collapses to a point mass at the PCA projection:

$$q(z_t) \;\to\; \delta(z_t - z_t^{\mathrm{PCA}}).$$

The posterior mean no longer depends on $f_\theta$.

The per-timestep ELBO decomposes as:

$$\mathcal{L}_t
= \underbrace{\mathbb{E}_{q}[\log p(y_t \mid z_t)]}_{\text{does not depend on } f_\theta}
\;-\; \underbrace{\mathrm{KL}\!\bigl(\delta_{z_t^{\mathrm{PCA}}} \,\|\, \mathcal{N}(f_\theta(z_{t-1}^{\mathrm{PCA}}),\, Q)\bigr)}_{\text{dynamics term}}.$$

The only $f_\theta$-dependent contribution from the KL term is the cross-entropy $-\mathbb{E}_q[\log p_{\mathrm{pred}}]$ (the Dirac entropy diverges but is constant w.r.t. $f_\theta$):

$$\mathrm{KL\ term}
= -\log \mathcal{N}\!\bigl(z_t^{\mathrm{PCA}};\; f_\theta(z_{t-1}^{\mathrm{PCA}}),\; Q\bigr) + \mathrm{const}$$

$$= \tfrac{1}{2}\bigl(z_t^{\mathrm{PCA}} - f_\theta(z_{t-1}^{\mathrm{PCA}})\bigr)^\top Q^{-1} \bigl(z_t^{\mathrm{PCA}} - f_\theta(z_{t-1}^{\mathrm{PCA}})\bigr)
+ \tfrac{1}{2}\log|Q| + \mathrm{const}.$$

### Specialisation to $Q = I$

$$\mathrm{KL\ term} = \tfrac{1}{2}\|z_t^{\mathrm{PCA}} - f_\theta(z_{t-1}^{\mathrm{PCA}})\|^2 + \mathrm{const}$$

which is exactly the MSE objective.

### Gradient identity

Since $\mathcal{L}$ denotes the ELBO (maximized) and $\mathcal{L}^{\mathrm{MSE}}$ the MSE loss (minimized), their gradients are opposite:

$$\frac{\partial \mathcal{L}}{\partial \theta}
= +\sum_t \bigl(z_t^{\mathrm{PCA}} - f_\theta(z_{t-1}^{\mathrm{PCA}})\bigr)
\cdot \frac{\partial f_\theta}{\partial \theta}(z_{t-1}^{\mathrm{PCA}})
\;=\;
-\frac{\partial \mathcal{L}^{\mathrm{MSE}}}{\partial \theta}$$

so maximising the ELBO and minimising MSE yield the same critical points and the same dynamics estimate. $\square$

The gradients are identical because in the $\sigma^2 \to 0$ limit:

1. The posterior is fixed at PCA coordinates regardless of $f_\theta$.
2. No gradient flows from $f_\theta$ through the posterior (no backprop through the filtering chain).
3. Each timestep's contribution is independent.
4. The only $f_\theta$-dependent term is the transition penalty.

## Summary table

| Regime | ELBO dynamics estimate = MSE on PCs? |
|--------|--------------------------------------|
| $\sigma^2 \to 0$, $Q = I$ | **Yes, exactly** |
| $\sigma^2 \to 0$, $Q$ fixed $\neq I$ | **Yes** — weighted MSE with weight $Q^{-1}$ |
| Finite $\sigma^2$, any $Q$ | **No** — XFADS is strictly richer |

## Finite $\sigma^2$: why XFADS is richer

For finite $\sigma^2$ the posterior mean $m_t$ depends on $f_\theta(m_{t-1})$:

$$m_t = \Sigma_t\bigl(Q^{-1} f_\theta(m_{t-1}) + \sigma^{-2} C^\top(y_t - b)\bigr),
\qquad
\Sigma_t = (Q^{-1} + \sigma^{-2} C^\top C)^{-1}.$$

Dynamics receive gradient through **two paths**:

1. The KL penalty (transition fit).
2. The observation likelihood (via the posterior's dependence on $f_\theta$).

The latent trajectory is **dynamics-smoothed** rather than raw PCA, yielding
a more robust dynamics estimate.

## NOFILT mode

The `nofilt` inference mode implements the $\sigma^2 \to 0$ limiting case
directly: the posterior is set by a user-provided encoder (bypassing
filtering), and dynamics are trained via the KL term alone. See the
[workflow guide](pca_dynamics_workflow.md#alternative-nofilt-mode) for
configuration details.
