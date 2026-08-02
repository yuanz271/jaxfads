# Plan: move M-step orchestration into pluggable trainer policy

**Status:** proposed architecture refactor; no code behavior is changed by this
plan file.

## Goal

Move M-step behavior out of `XFADS`, `Observation`, `Noise`, and `Approx` into
configurable trainer-owned transformations.

The active trainer order is:

$$
M_k
\xrightarrow{\text{one forward: loss + gradients + outputs}}
\xrightarrow{\text{SGD}}
M_k^{\mathrm{SGD}}
\xrightarrow{\text{trainer M-step plugins using prior outputs}}
M_{k+1}.
$$

The M-step is intentionally one-step-lagged: plugin statistics come from the
pre-SGD model's forward outputs but are applied to the post-SGD model. This
avoids a second inference/transition-propagation pass per batch.

## Target architecture

```text
XFADS
├── observation: Observation       # model/inference state
├── noise: Noise                   # model/inference state
│   ├── approx: Approx             # static, array-free distribution config
│   └── free: Array                # Q parameter leaf
└── __call__(...) -> inference outputs

Trainer
└── msteps: Sequence[MStep]        # ordered independent transformations
    ├── GaussianObservationMstep   # R update
    ├── MVNNoiseMstep              # Q update
    ├── root-relative frozen paths
    └── post-SGD update ordering
```

`Approx` remains distribution-only. `Noise` retains only static Approx
composition, Q initialization state, `free`, and generic predictive decoding.
Remove M-step registries, M-step execution, M-step policy, and freeze-path
policy from `Noise`.

## 1. Forward-output contract

The model forward pass remains inference-only and returns:

```python
natural, moment, predictive_moment = model(
    t, y, u, c, key=key
)
```

The trainer supplies these outputs to all selected plugins. Plugins must not
invoke `model(...)` themselves.

For the current MVN/additive-Gaussian process noise, `MVNNoiseMstep`
reconstructs the noise-free predictive covariance from model outputs:

```python
mean_f, cov_pred = model.approx.unpack(predictive_moment[:, 1:])
_, q = model.approx.unpack(model.noise.moment())
cov_f = 0.5 * ((cov_pred - q) + (cov_pred - q).T)
```

This is strategy-specific, not generic `Approx` or `core` behavior. Do not
add a fourth `transition_stat` output or a general auxiliary-output bundle
until a concrete future plugin requires one.

## 2. Trainer-owned plugin interface

Start with explicit plugin objects, not a config-name registry:

```python
class MStep(Protocol):
    def __call__(
        self,
        model: XFADS,
        batch,
        forward,
        *,
        key,
    ) -> XFADS:
        ...

    def frozen_paths(self, model: XFADS) -> list[str]:
        ...
```

`frozen_paths` returns fully qualified paths relative to root `XFADS`, since
the plugin is a root-level trainer policy.

```python
train(
    model,
    data,
    *,
    conf,
    msteps=(),
    ...,
)
```

Semantics:

- `msteps=()`: ordinary SGD; no plugin-owned freeze paths;
- each supplied plugin receives the same forward outputs and transforms the
  post-SGD model in sequence;
- R-only and Q-only experiments are explicit;
- current R/Q plugins commute because they read the same immutable outputs and
  modify disjoint model leaves; sequence order remains explicit for future
  plugins.

```python
msteps=(
    GaussianObservationMstep(),
    MVNNoiseMstep(q_prior_fraction=0.1),
)
```

## 3. Independent R and Q plugins

### GaussianObservationMstep

```python
class GaussianObservationMstep:
    def __call__(self, model, batch, forward, *, key):
        # Use t, y, posterior moment, observation/readout state.
        # Replace model.observation.
        ...

    def frozen_paths(self, model):
        return ["observation.likelihood.unconstrained_cov"]
```

It owns only the R residual/covariance update.

### MVNNoiseMstep

```python
class MVNNoiseMstep:
    q_prior_fraction: float

    def __call__(self, model, batch, forward, *, key):
        # Use moment, predictive_moment, model.noise.moment(), and MVN
        # conversion helpers. Replace model.noise.free.
        ...

    def frozen_paths(self, model):
        return ["noise.free"]
```

It owns only the Q update:

$$
Q_{\mathrm{new}}
=
\frac{\widehat Q+\alpha q_{\mathrm{scale}}I}
{1+\alpha},
\qquad
\alpha=\texttt{q\_prior\_fraction}.
$$

`MVNNoiseMstep` validates its required concrete Approx/noise representation at
construction or first use and raises a clear error when unsupported. A future
plugin may intentionally implement a no-op or alternate Q policy.

Plugins may use narrowly scoped distribution/model helpers but must not call
component M-step methods, recursively trigger another M-step, or invoke
`model(...)`.

## 4. Exact trainer ordering

Inside the jitted step:

```python
(loss, forward), grads = value_and_grad_with_aux(loss_and_forward)(
    model, batch
)
params = eqx.filter(model, eqx.is_inexact_array)
updates, opt_state = optimizer.update(grads, opt_state, params)
model = eqx.apply_updates(model, updates)

for mstep in msteps:
    model = mstep(model, batch, forward, key=stat_key)
```

`params` is the pre-SGD filtered parameter pytree corresponding to `grads`.
Plugin-owned frozen leaves are not optimizer-owned, so the plugin update is
their final update for the batch.

There is no epoch accumulator, delayed epoch M-step, extra full-data pass, or
extra post-SGD forward pass in normal training.

## 5. Configuration ownership

Keep `q_scale` on model/Noise because it defines initial Q and the shrinkage
center:

```yaml
q_scale: 1.0
```

Keep M-step-specific policy on the selected plugin:

```python
MVNNoiseMstep(q_prior_fraction=0.1)
```

Do not duplicate `q_scale` in the plugin. Q-only behavior is selected by
including or omitting `MVNNoiseMstep`; R-only behavior uses only
`GaussianObservationMstep`.

## 6. Simplify model components

`Approx` retains only distribution operations:

- natural/moment/canonical/free conversions;
- sampling and KL;
- transition-point generation and transition-stat reduction;
- predictive moment computation given noise moments.

`Noise` retains only:

```python
class Noise(eqx.Module):
    approx: Approx = eqx.field(static=True)
    free: Array
```

Remove from active model code:

- exact Approx M-step registry;
- `batch_stat`, `accumulate_stat`, `mstep`;
- `mstep_active`, `supports_mstep`;
- `q_prior_fraction`, `mstep_enabled`;
- component-level M-step freeze policy.

Remove from XFADS:

- `_batch_stat`;
- `_apply_mstep_stat`;
- `mstep_from_data`;
- component M-step dispatch;
- Q-specific M-step policy;
- M-step-specific freeze-path composition.

XFADS retains model state, `approx` access through the composed Noise Approx,
inference `__call__`, and save/load/initialization.

If a manual full-data update is needed, use trainer-side utility code:

```python
def apply_mstep_from_data(model, data, *, msteps, key):
    forward = model(*data, key=key)
    for mstep in msteps:
        model = mstep(model, data, forward, key=key)
    return model
```

This utility is not an XFADS/component method and is not used by normal
training.

## 7. Freeze-path ownership

The root-level trainer plugins own fully qualified root-relative paths:

```python
GaussianObservationMstep.frozen_paths(model)
# -> ["observation.likelihood.unconstrained_cov"]

MVNNoiseMstep.frozen_paths(model)
# -> ["noise.free"]
```

The trainer concatenates plugin paths with user `conf.freeze_paths`. Components
do not expose M-step freeze paths and do not know parent member names.

## 8. Tests to remove and redirect

Remove tests for superseded model-side M-step lifecycle:

- `XFADS._batch_stat`;
- `XFADS._apply_mstep_stat`;
- `Noise.batch_stat`/`Noise.mstep` strategy behavior;
- component recursive M-step delegation;
- epoch-stat accumulation/reset.

Add/retain focused tests for:

1. `Approx` remains array-free/static and `Noise` preserves composed Approx.
2. `GaussianObservationMstep` updates only R from one supplied forward output.
3. `MVNNoiseMstep` updates only Q from the same supplied forward output.
4. `msteps=()` performs ordinary SGD without retained plugin outputs.
5. Unsupported Approx use fails clearly for `MVNNoiseMstep`.
6. No additional post-SGD forward call occurs when one or more plugins run.
7. Fractional Q formula matches an independent reference.
8. Omitting `MVNNoiseMstep` leaves `noise.free` SGD-managed while R can update.
9. Parameter-aware Optax receives pre-SGD parameters while plugin-frozen
   leaves retain M-step values and Q decomposition uses the frozen pre-SGD Q.
10. Checkpoint roundtrip preserves static Approx and Noise state.
11. VDP MLP regression reports RMSE, aligned covariance, aligned Q, and flow
    metrics.

## 9. Migration and validation order

1. Add the trainer `MStep` protocol and `msteps=()` argument.
2. Implement `GaussianObservationMstep` and `MVNNoiseMstep` using existing
   R/Q formulas.
3. Wire loss-and-gradient forward outputs into each plugin call.
4. Remove XFADS/Noise/Observation M-step orchestration and obsolete APIs.
5. Migrate freeze paths and configuration/docs.
6. Remove superseded tests and add plugin/ordering tests.
7. Run focused tests, then the full CPU suite.
8. Run the VDP example and small Lorenz smoke benchmark.
9. Review diff and commit. Do not push without explicit instruction.
