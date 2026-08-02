# Trainer-owned Gaussian observation covariance update

**Status:** active design and rationale.

The active implementation uses the trainer-owned
`GaussianObservationMstep` passed explicitly through
`train(..., post_optimizer_transforms=(GaussianObservationMstep(),))`.
It performs a direct post-optimizer minibatch update from the ordinary
three-value forward output:

```python
natural, moment, predictive_moment = model(t, y, u, c, key=key)
```

The former component/model M-step APIs, standalone observation M-step
utilities, and epoch-accumulation design are removed and are not supported.

## The problem

`Gaussian.eloglik` computes the joint likelihood

```text
log N(y; E[Cz], C Cov(z) C.T + R)
```

where `C Cov(z) C.T` is low-rank and `R` is diagonal observation noise. This
factor-analysis structure admits a Heywood-type degeneracy: joint
gradient-based optimization can drive one or more components of `R` toward the
numerical floor while the corresponding reconstruction residual remains large.

## The update

`GaussianObservationMstep` estimates each observation variance using the
expected squared residual under the current posterior:

$$
R_d = \operatorname{mean}_{n,t}
\left[(y_{n,t,d}-E[Cz_{n,t}]_d)^2
+ \operatorname{diag}(C\operatorname{Cov}(z_{n,t})C^\mathsf{T})_d\right].
$$

This update is implemented by the trainer transform. It uses the pre-optimizer
posterior moments and the current post-optimizer readout state, then replaces
only the Gaussian likelihood covariance. The accepted one-step lag avoids an
additional inference pass.

## Active usage

```python
from jaxfads.msteps import GaussianObservationMstep
from jaxfads.trainer import train

trained = train(
    model,
    data,
    conf=trainer_conf,
    post_optimizer_transforms=(GaussianObservationMstep(),),
)
```

The transform owns the R update and declares
`observation.likelihood.unconstrained_cov` as a root-relative frozen path, so
optimizer gradients do not fight the closed-form replacement. Omit the
transform for ordinary optimizer-managed observation covariance.

`MVNNoiseMstep` is an independent transform for Q and may be selected in the
same ordered `post_optimizer_transforms` tuple.

## Historical record

Earlier versions of this document described standalone functions such as
`mstep_gaussian_cov` and `mstep_observation_cov`, component methods such as
`Observation.mstep` and `Likelihood.mstep`, recursive `XFADS.mstep` dispatch,
epoch-local accumulation, automatic model freeze-path composition, and the
`mstep_mode` cadence option. Those APIs and cadence rules were superseded by
the trainer-owned post-optimizer transform interface and are retained only in
git history, not as supported usage.
