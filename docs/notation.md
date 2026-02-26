# Naming and Notation

This document describes the *mathematical* naming/notation used in JAXFADS.
(Concrete API surface and method tables live in `docs/design.md`.)

## Parameter forms

JAXFADS uses multiple parameter representations for exponential-family
approximations.

| Term | Meaning | Typical type |
|------|---------|--------------|
| **free** | Unconstrained parameters optimized by SGD | flat array |
| **canon** | Constrained, valid parameters | pytree |
| **natural** (`η`) | Natural parameters of the exponential family | flat array |
| **moment** (`μ`) | Moment parameters of the exponential family | flat array |

**Moment parameters** are the expected sufficient statistics:

`μ = E[T(z)]`.

This quantity is commonly called the *mean parameter* in the exponential-family
literature. To avoid confusion with the expectation of a random variable (e.g.
`E[z]`), this repo consistently uses the term **moment**.

## Symbols

| Symbol | Meaning |
|--------|---------|
| `z_t` | Latent state at time `t` |
| `y_t` | Observation at time `t` |
| `u_t` | Input/control at time `t` |
| `c_t` | Covariates/context at time `t` |
| `T(z)` | Sufficient statistics of the exponential family |
| `η` | Natural parameters |
| `μ` | Moment parameters (`E[T(z)]`) |

## Predictive moments (Eq. 4)

For the transition model, the predictive moment parameters at time `t` are the
conditional moments:

`μ(z_{t-1}) = E_{p(z_t | z_{t-1})}[T(z_t)]`.

This is an inner expectation over `z_t` under the transition distribution
`p(z_t | z_{t-1})`.

## Expected predictive moments (Eq. 12)

For the transition model, the expected predictive moment parameters at time `t`
are the integrated moments:

`E_{π(z_{t-1})}[ μ(z_{t-1}) ] = E_{π(z_{t-1})}[ E_{p(z_t | z_{t-1})}[T(z_t)] ]`.

This is an outer expectation over `z_{t-1}` and an inner expectation under the
transition distribution.

## Method naming pattern (informal)

When reading code, conversions typically follow an explicit `X_to_Y` naming
pattern (e.g. natural→moment, free→canon). This is a convention for
readability, not a mathematical requirement.
