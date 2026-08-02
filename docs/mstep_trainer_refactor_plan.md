# Plan: move M-step orchestration into pluggable trainer policy

**Status:** proposed architecture refactor; no code behavior is changed by this
plan file.

## Goal

Make M-step behavior a configurable trainer transformation rather than part of
`XFADS`, `Observation`, `Noise`, or `Approx` model architecture.

The trainer owns the update ordering and selected M-step policy:

$$
\text{model}
\xrightarrow{\text{one forward pass}}
(\text{loss},\text{gradients},\text{inference outputs})
\xrightarrow{\text{M-step transformation}}
\xrightarrow{\text{SGD update}}
\text{next model}.
$$

The model remains responsible for representing the generative/inference model
and producing three forward outputs. The M-step plugin interprets those
outputs and returns an updated model.

## Target architecture

```text
XFADS
├── observation: Observation       # model state/inference behavior
├── noise: Noise                   # model state/inference behavior
│   ├── approx: Approx             # static, array-free distribution config
│   └── free: Array                # Q parameter leaf
└── __call__(...) -> ForwardOutput

Trainer
└── mstep: MStep policy/plugin
    ├── R update
    ├── Q update
    ├── optimizer ownership/freeze paths
    └── post-SGD update ordering
```

Remove M-step strategy registries and M-step execution from `Noise`. `Noise`
may retain generic inference functionality such as decoding `free` into noise
moment parameters and delegating predictive moments to its composed `Approx`.
`Approx` remains distribution-only.

## 1. Define the forward-output contract

The active forward output is the existing three-value inference result:

```python
natural, moment, predictive_moment = model(
    t, y, u, c, key=key
)
```

The model forward pass only performs inference and returns these outputs. The
trainer/plugin computes M-step statistics outside `XFADS.__call__`; the model
forward pass must not collect, accumulate, or apply M-step statistics.

For the current MVN/additive-Gaussian process-noise model, Q's plugin
reconstructs the noise-free predictive moments from the predictive output and
current Noise covariance:

```python
mean_f, cov_pred = approx.unpack(predictive_moment[:, 1:])
_, q = approx.unpack(model.noise.moment())
cov_f = cov_pred - q
```

The subtraction must be symmetrized and numerically stabilized. This is an
MVN/Noise-strategy operation, not a generic Approx or core operation. Do not
introduce a fourth `transition_stat` output or a general `aux` bundle yet; add
another forward hook only when a concrete future Approx strategy requires it.

## 2. Define the trainer-owned MStep interface

Start with an explicit callable/plugin argument rather than a registry:

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

`frozen_paths` returns fully qualified paths relative to the root `XFADS`
model because the trainer plugin is the root-level policy owner. Components no
longer declare M-step freeze paths.

The minimal trainer API becomes:

```python
train(
    model,
    data,
    *,
    conf,
    mstep=None,
    ...,
)
```

Semantics:

- `mstep=None`: ordinary SGD; no post-SGD inference pass and no M-step-owned
  freeze paths;
- `mstep=JointGaussianMAP(...)`: R/Q transformation on every batch before
  the precomputed SGD updates, with plugin-declared frozen paths;
- future plugins can implement other policies without changing XFADS.

`mstep=None` therefore means neither R nor Q is M-step-owned. This is a
substantial but intentional policy boundary: closed-form R/Q updates are
provided by the selected trainer plugin, not unconditionally by the model.

Do not add a config-name registry until a second M-step policy exists. A direct
plugin object is easier to inspect and avoids import-order configuration magic.

## 3. Implement the joint R/Q plugin

Create a trainer-side implementation, for example:

```python
class JointGaussianMAP:
    def __call__(self, model, batch, forward, *, key):
        ...

    def frozen_paths(self, model):
        ...
```

Its `__call__` should:

1. unpack `batch` as `t, y, u, c`;
2. unpack `forward` as `natural, moment, predictive_moment`;
3. compute the Gaussian R statistic/update itself, using `t`, `y`, `moment`,
   and the model's observation/readout state;
4. compute Q transition residual covariance from `moment`,
   `predictive_moment`, and `model.noise.moment()`: for MVN, subtract current
   Q from the predictive covariance to recover the noise-free covariance,
   then delegate MVN representation conversion to `model.noise.approx`;
5. apply the Q fractional shrinkage:

   $$
   Q_{\mathrm{new}}
   =
   \frac{\widehat Q+\alpha q_{\mathrm{scale}}I}
   {1+\alpha},
   \qquad
   \alpha=\texttt{q\_prior\_fraction};
   $$

6. structurally replace `model.observation` and `model.noise.free` and return
   the updated model.

The plugin is the sole owner of joint R/Q policy and update mathematics. It
may use narrowly scoped distribution/model helpers, but it must not call
component M-step methods or recursively trigger another M-step. Its model
outputs are computed by the trainer's one forward pass; the plugin never calls
`model(...)` itself.

## 4. Preserve one-forward-pass pre-SGD ordering

The trainer uses one forward pass for the loss, gradients, and inference
outputs. It computes M-step statistics outside `XFADS.__call__`, applies the
M-step first, then applies the precomputed SGD updates:

```python
(loss, forward), grads = value_and_grad_with_aux(loss_and_forward)(
    model, batch
)
params = eqx.filter(model, eqx.is_inexact_array)
model = mstep(model, batch, forward, key=stat_key)
updates, opt_state = optimizer.update(grads, opt_state, params)
model = eqx.apply_updates(model, updates)
```

The optimizer receives the original pre-M-step parameter tree corresponding to
the gradients. Plugin-frozen leaves are not optimizer-owned, so the
subsequent SGD application cannot overwrite their M-step values. There is no
epoch accumulator, delayed epoch M-step, or extra full-data pass in normal
training.

## 5. Configuration ownership

Keep `q_scale` as model/Noise initialization configuration because it defines
initial Q:

```yaml
q_scale: 1.0
```

Move M-step-specific policy into the plugin:

```python
JointGaussianMAP(
    q_prior_fraction=0.1,
    q_enabled=True,
)
```

`q_scale` is read from the model's Noise state/config as the shrinkage center;
`q_prior_fraction` and `q_enabled` belong to the selected M-step policy. Do
not duplicate `q_scale` in the plugin configuration.

A plugin with `q_enabled=False` updates R only. `JointGaussianMAP` should
validate its required Approx/noise representation at construction or first use
and raise a clear error for an unsupported Approx; this is explicit plugin
policy, not an `Approx` responsibility. A future plugin may intentionally use a
no-op or alternate Q policy.

## 6. Simplify Noise and Approx

`Approx` retains only distribution functionality:

- natural/moment/canonical/free conversions;
- sampling and KL;
- transition-point generation and transition-stat reduction;
- predictive moment computation given noise moments.

`Noise` retains only model/inference state:

```python
class Noise(eqx.Module):
    approx: Approx = eqx.field(static=True)
    free: Array
```

Remove from `Noise`:

- exact Approx M-step registry;
- `batch_stat`;
- `mstep`;
- `mstep_active` / `supports_mstep`;
- `q_prior_fraction` and `mstep_enabled`; these are trainer-plugin policy,
  not Noise state.

The selected trainer plugin owns Q M-step strategy, shrinkage fraction, and
freeze paths. Noise retains only static Approx composition, Q initialization
state, free storage, and generic predictive decoding.

## 7. Simplify XFADS

Remove from XFADS:

- `_batch_stat`;
- `_apply_mstep_stat`;
- `mstep_from_data`;
- component M-step dispatch;
- Q-specific policy predicates;
- M-step-specific frozen-path composition.

XFADS should retain:

- `observation` and `noise` model state;
- `approx` access through the canonical composed Noise Approx;
- inference `__call__` and its three named outputs (`natural`, `moment`,
  `predictive_moment`);
- structural model initialization/save/load.

If manual full-data M-step use is needed, provide a trainer-side utility:

```python
def apply_mstep_from_data(model, data, *, mstep, key):
    forward = model(*data, key=key)
    return mstep(model, data, forward, key=key)
```

This utility is not an XFADS/component method and is not used by normal
`train()`.

## 8. Freeze-path ownership

This trainer-plugin architecture replaces component-relative freeze-path
composition. The selected root-level M-step plugin owns fully qualified
XFADS-root-relative freeze paths:

```python
class JointGaussianMAP:
    def frozen_paths(self, model):
        paths = ["observation.likelihood.unconstrained_cov"]
        if self.q_enabled:
            paths.append("noise.free")
        return paths
```

The trainer combines these with user `conf.freeze_paths`. No component needs to
know the parent member name, and the trainer does not query component-level
`frozen_paths()` methods.

## 9. Tests to reduce and redirect

Remove tests for the superseded model-side M-step lifecycle:

- `XFADS._batch_stat`;
- `XFADS._apply_mstep_stat`;
- `Noise.batch_stat`/`Noise.mstep` strategy behavior;
- component recursive M-step delegation;
- epoch-stat accumulation/reset.

Retain or add focused tests for:

1. `Approx` remains array-free/static and `Noise` preserves the composed Approx.
2. `JointGaussianMAP` updates R and Q from one supplied forward output.
3. `mstep=None` performs ordinary SGD without a second inference pass.
4. Unsupported Approx construction/use fails clearly for `JointGaussianMAP`.
5. One post-SGD M-step forward call occurs per batch when a plugin is supplied.
5. Fractional Q formula matches an independent reference.
6. `q_enabled=False` leaves `noise.free` SGD-managed.
7. Parameter-aware Optax receives pre-SGD parameters while the plugin’s frozen
   leaves retain M-step values.
8. Plugin rejection/no-op behavior for unsupported Approx families.
9. Checkpoint roundtrip preserves the static Approx and Noise state.
10. VDP MLP regression reports RMSE, aligned covariance, aligned Q, and flow
    metrics.

## 10. Migration and validation order

1. Add the trainer `MStep` protocol and `mstep=` argument.
2. Implement `JointGaussianMAP` using existing R/Q formulas.
3. Wire post-SGD forward output into the plugin call.
4. Remove XFADS/Noise/Observation M-step orchestration and obsolete APIs.
5. Migrate freeze paths and configuration/docs.
6. Remove superseded tests and add plugin/ordering tests.
7. Run focused tests, then the full CPU suite.
8. Run the VDP example and small Lorenz smoke benchmark.
9. Review diff and commit. Do not push without explicit instruction.
