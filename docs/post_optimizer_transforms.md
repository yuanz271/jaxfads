# Post-optimizer transforms

The reproducibility requirements for built-in and custom transforms are defined
in [Reproducibility](reproducibility.md).

This document specifies the trainer-owned interface for non-gradient model
transformations applied after the optimizer. Gaussian observation-noise and MVN
process-noise updates are the initial implementations; the interface is not
limited to EM-style M-steps.

## Training order

`train()` accepts one Optax optimizer transformation and an ordered sequence:

```python
train(
    model,
    data,
    conf=trainer_conf,
    post_optimizer_transforms=(GaussianObservationMstep(),),
)
```

Each batch follows this order:

1. Run one model forward pass while evaluating the loss.
2. Differentiate the scalar loss and obtain optimizer gradients.
3. Apply the single Optax optimizer transformation.
4. Apply each `post_optimizer_transforms` item in sequence.

The optimizer is typically gradient-based, but this contract does not require
that assumption. The transforms are model-level operations and are not a
second optimizer.

## Loss and forward contract

The local differentiated function has the shape:

```python
def loss_and_forward(model):
    forward = model(t, y, u, c, key=forward_key)
    loss = ...
    return loss, forward

(loss, forward), grads = eqx.filter_value_and_grad(
    loss_and_forward,
    has_aux=True,
)(model)
```

`has_aux=True` means that `grads` is the gradient of scalar `loss` only.
`forward` is carried out as auxiliary output for the post-optimizer
transforms; gradients are not separately requested for it.

The model forward contract is exactly:

```python
natural, moment, predictive_moment = model(t, y, u, c, key=key)
```

Transforms consume these outputs and must not invoke `model(...)` themselves.
No fourth transition-statistic output or M-step-specific forward bundle is
part of the interface.

## Transform interface

A transform may initialize model state, update the post-optimizer model, and
declare model leaves that the optimizer must not update:

```python
class PostOptimizerTransform(Protocol):
    def initialize(self, model, *, key):
        ...

    def __call__(self, model, batch, forward, *, key):
        ...

    def frozen_paths(self, model) -> list[str]:
        ...
```

`initialize()` runs for each selected transform before freeze-mask
construction and `optimizer.init`. This ensures that optimizer state is built
from the actual initialized parameter arrays. `frozen_paths()` returns fully
qualified, root-relative paths such as `"noise"`; the trainer combines
these with user-configured `conf.freeze_paths`.

`post_optimizer_transforms=()` performs ordinary optimizer-managed training:
there are no transform initializers, transform-owned freeze paths, or
post-optimizer model updates.

## Gaussian observation-noise transform

`GaussianObservationMstep` updates only diagonal Gaussian observation
covariance `R`. For posterior mean and covariance from `moment`, readout `C`,
and observations `y`, it computes:

$$
R_d = \operatorname{mean}_{n,t}
\left[
(y_{n,t,d} - E[Cz_{n,t}]_d)^2
+ \operatorname{diag}\left(C\operatorname{Cov}(z_{n,t})C^\mathsf{T}\right)_d
\right].
$$

It replaces `observation.likelihood.unconstrained_cov` and declares that
root-relative path frozen from optimizer updates. It is a no-op for likelihood
models without a Gaussian covariance parameter.

## MVN process-noise transform

`MVNNoiseMstep(q_scale, q_prior_fraction)` owns the complete isotropic Q prior.
It is enabled declaratively with `conf.q_mstep=True`; the trainer constructs it
from serializable `conf.q_scale` and `conf.q_prior_fraction` values. The
transform's initializer sets:

$$
Q_0 = Q_{\mathrm{init}} = q_{\mathrm{scale}} I.
$$

The initializer encodes this covariance into `model.noise` before
optimizer initialization. On each transform call, it reconstructs the
noise-free predictive covariance for the current MVN representation by
subtracting the Q that produced the predictive output:

$$
P_f = P_{\mathrm{pred}} - Q.
$$

Using posterior covariance, noise-free predictive covariance, and transition
mean residuals, it forms the residual covariance estimate $\widehat Q$ and
applies:

$$
Q_{\mathrm{new}}
= \frac{\widehat Q + \alpha Q_0}{1 + \alpha},
\qquad
\alpha = \texttt{q\_prior\_fraction}.
$$

It replaces `noise` and declares `noise` frozen from optimizer
updates. `MVNNoiseMstep` requires the concrete `MVN` Approx representation;
unsupported representations fail explicitly. With `conf.q_mstep=False`, Q is
optimizer-managed. Directly supplying `MVNNoiseMstep(...)` in
`post_optimizer_transforms` remains available for custom composition, but the
serializable trainer configuration is the reproducible built-in policy.

## One-step-lagged update and ownership

Transform statistics come from the pre-optimizer forward output but are applied
to the post-optimizer model. In particular, the Gaussian transform reconstructs
readout quantities from the posterior `moment` and the current post-optimizer
readout state. This accepted one-step-lagged approximation avoids a second
inference pass and keeps the forward-output contract minimal.

`XFADS`, `Observation`, and `Approx` remain model/inference and distribution
components. They do not own transform policy, transform registries, M-step
dispatch, accumulation, or transform-specific freeze paths. `XFADS.noise` is
the free-form process-noise array; `XFADS.approx` is static distribution
configuration. Trainer policy owns initialization, update mathematics,
ordering, and optimizer freeze composition.
