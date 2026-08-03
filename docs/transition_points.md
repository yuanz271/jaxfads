# Transition-point propagation

This document specifies how XFADS approximates the prediction-step expectation
through nonlinear latent dynamics.

## Contract

Before propagating a posterior distribution through a transition, core
inference asks the approximation family for representative latent points and
weights:

```python
points, weights = approx.transition_points(key, moment, mc_size)
```

The result represents the distribution encoded by `moment`:

- `points`: shape `(n_points, state_dim)`;
- `weights`: shape `(n_points,)`;
- weights define the weighted approximation used to average predictive
  sufficient statistics.

Core inference is agnostic to how points are selected. It propagates each
point through the latent transition, evaluates the family-specific predictive
moment, and takes a weighted average. The process-noise moment is added by the
Approx predictive-moment operation, not by the transition-point policy.

## Default Approx behavior: Monte Carlo

`Approx.transition_points(...)` defaults to independent samples from
`sample_by_moment` with uniform weights:

$$
z_i \sim q(z),
\qquad
w_i = \frac{1}{S},
\qquad
S=\texttt{mc\_size}.
$$

This is the appropriate fallback for approximation families without a
deterministic propagation rule. For Monte Carlo points, use enough samples for
the desired accuracy. In particular, when `mc_size <= state_dim`, the
between-point covariance has rank at most `mc_size - 1`; XFADS logs a warning
at trace time because this is often a poorly conditioned approximation of a
full-dimensional predictive spread.

## MVN behavior: unscented sigma points

`MVN` uses deterministic unscented-transform sigma points by default:

```python
MVN(
    dim=state_dim,
    rank=rank,
    use_sigma_points=True,
    ut_alpha=1.0,
    ut_kappa=0.0,
)
```

For a Gaussian posterior with mean $m$, covariance $P$, and dimension $D$,
define

$$
\lambda = \alpha^2(D+\kappa)-D,
\qquad
c=D+\lambda.
$$

With $L L^\mathsf{T}=P$, the point set is

$$
X_0=m,
\qquad
X_i=m+\sqrt{c}\,L_{:,i},
\qquad
X_{D+i}=m-\sqrt{c}\,L_{:,i},
$$

for $i=1,\ldots,D$, with weights

$$
w_0=\frac{\lambda}{c},
\qquad
w_i=\frac{1}{2c}
\quad (i=1,\ldots,2D).
$$

XFADS averages predictive sufficient statistics directly, so it uses this
single weight vector rather than separate UKF mean and covariance weights.
For the default $\alpha=1$ and $\kappa=0$, there are $2D+1$ points,
$w_0=0$, and all noncentral weights are $1/(2D)$.

`mc_size` remains in the method signature for interface compatibility but is
ignored when MVN sigma points are enabled. Set `use_sigma_points=False` to use
the inherited Monte Carlo policy instead.

The implementation uses a Cholesky factor for full MVN layouts and the
existing diagonal covariance factor for `rank=0` layouts.

## Numerical behavior

Non-finite per-point predictive moments are excluded from the weighted average
and the remaining valid weights are renormalized. If no positive total valid
weight remains, the predictive moment is non-finite. Transition dynamics should
therefore remain finite over representative posterior support.

The default MVN parameters use nonnegative weights and avoid the large
cancellation associated with very small scaled-UT `alpha` values in float32.
Users who override `ut_alpha` or `ut_kappa` are responsible for the resulting
conditioning and point geometry.

## Design boundaries

- `Approx` owns the public `transition_points` interface.
- Concrete families may override point selection when they have a justified
  deterministic rule.
- `core` owns transition propagation and weighted averaging.
- `Dynamics` owns the transition map.
- Process noise remains part of `Approx.predictive_moment`, not the point
  selection policy.

See also: [Design](design.md) and [Training](training.md).
