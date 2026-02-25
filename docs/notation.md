# Naming and Notation

This document describes the *mathematical* naming/notation used in JAXFADS.
(Concrete API surface and method tables live in `docs/design.md`.)

## Parameter forms (concepts)

JAXFADS uses multiple parameter representations for exponential-family
approximations.

| Term | Meaning | Typical type |
|------|---------|--------------|
| **free** | Unconstrained parameters optimized by SGD | pytree |
| **canon** | Constrained, valid parameters | pytree |
| **natural** (`η`) | Natural parameters of the exponential family | flat array |
| **moment** (`μ`) | Flat *storage* moment-parameter vector used throughout the algorithms | flat array |

Important: the word **moment** is used in two related senses:

1. **Storage moment layout**: the flat vector representation passed around the
   codebase (e.g. MVN stores `[loc, cov]` encoded as diagonal + low-rank factor).
2. **Sufficient-statistic moments**: expected sufficient statistics `E[T(z)]`.
   Some operations (notably the variational predict Eq. (12)) are most naturally
   performed in this space, then converted back to the storage layout.

## Symbols

| Symbol | Meaning |
|--------|---------|
| `z_t` | Latent state at time `t` |
| `y_t` | Observation at time `t` |
| `u_t` | Input/control at time `t` |
| `c_t` | Covariates/context at time `t` |
| `T(z)` | Sufficient statistics of the exponential family |
| `η` | Natural parameters |
| `μ` | Moment parameters (context-dependent; see above) |

## Predictive moments (Eq. 12)

For the transition model, the predictive moment parameters at time `t` are the
integrated moments:

`E_{π(z_{t-1})}[ E_{p(z_t|z_{t-1})}[T(z_t)] ]`.

This is an outer expectation over `z_{t-1}` and (in general) an inner
expectation under the transition distribution `p(z_t|z_{t-1})`.

## Method naming pattern (informal)

When reading code, conversions typically follow an explicit `X_to_Y` naming
pattern (e.g. natural→moment, free→canon). This is a convention for readability,
not a mathematical requirement.
