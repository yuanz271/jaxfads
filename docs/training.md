# Training Configuration

This guide explains how to configure and run XFADS training.

See also: [Quickstart](quickstart.md), [Dynamics](dynamics.md),
[Algorithm](algorithm.md).

## Overview

Training maximises the Evidence Lower Bound (ELBO) over mini-batches using
Optax optimizers, with multi-device data parallelism and gradient clipping.
The trainer is a pure mechanism: it runs the full `max_epoch` and returns the
final-epoch model. Validation, checkpointing, best-model tracking, and early
stopping are epoch-level policy supplied via an `on_epoch_end` callback (see
[`EpochHandler`](#epoch-callbacks-and-the-epochhandler)); the caller owns the
train/validation split.

```python
from jaxfads.trainer import train

trained_model = train(model, train_data, conf=trainer_conf)
```

## Data Format

`train_data` (and any `valid_data`) is a 4-tuple of JAX arrays:

| Element | Shape | Description |
|---------|-------|-------------|
| `times` | `(N, T)` | Integer time indices |
| `observations` | `(N, T, D_obs)` | Observed data |
| `controls` | `(N, T, D_u)` | Control inputs (use `(N, T, 0)` if none) |
| `covariates` | `(N, T, D_c)` | Covariates (use `(N, T, 0)` if none) |

where `N` is the number of trials and `T` is the sequence length.

## Configuration Reference

Pass a `DictConfig` (or plain dict) as `conf`. Missing keys are filled from
`DEFAULT_TRAINER_CONFIG`.

| Key | Default | Description |
|-----|---------|-------------|
| `learning_rate` | `1e-3` | Adam learning rate |
| `max_epoch` | `50` | Number of training epochs (always run in full unless a callback stops early) |
| `batch_size` | `1` | Mini-batch size (must be divisible by device count) |
| `seed` | `0` | Random seed for shuffling |
| `freeze_paths` | `[]` | Optional list of dot-separated model attribute paths to freeze (e.g. `["noise"]`) |

The default optimizer is **vanilla Adam** and `learning_rate` is its only
knob. It applies **no gradient clipping, no gradient noise, and no weight
decay** -- in a plugin framework the trainer cannot know which leaves are
weight matrices vs variances/biases, and clipping/noise can destabilize
sensitive objectives (e.g. chaotic dynamical-systems reconstruction), so the
default imposes no such policy. To add clipping, weight decay, gradient
noise, or a custom schedule, pass your own `optax` optimizer via
`train(..., optimizer=...)` (see below); `learning_rate` is then ignored, but
`freeze_paths` is still applied on top.

A model regularizer is **not** a config field; it is passed directly to
`train(..., regularizer=...)` (see below), because it is a Python callable and
the config is meant to stay serializable.

Validation, checkpointing, and early stopping are not training config; they are
handler concerns (see [`EpochHandler`](#epoch-callbacks-and-the-epochhandler)).

### Example

```python
from omegaconf import OmegaConf

trainer_conf = OmegaConf.create(dict(
    seed=42,
    learning_rate=5e-4,
    max_epoch=200,
    batch_size=64,
))
```

### Custom optimizer (e.g. masked weight decay)

The default optimizer applies no weight decay. To decay only the weight
matrices of your model (and not biases / variance-like free parameters), build
an `optax` optimizer with a model-derived mask and pass it to `train`:

```python
import jax, equinox as eqx, optax

# Decay only 2-D leaves (weight matrices); skip biases, scales, and the
# free-form covariance parameters (noise, unconstrained_prior_natural).
wd_mask = jax.tree.map(lambda p: eqx.is_inexact_array(p) and p.ndim >= 2, model)

optimizer = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.scale_by_adam(),
    optax.add_decayed_weights(1e-3, mask=wd_mask),
    optax.scale_by_learning_rate(1e-3),
)

train(model, data, conf=trainer_conf, optimizer=optimizer)
```

Any `optax` optimizer works, including **params-aware** ones that read the
current parameters at update time -- decoupled weight decay (`adamw`,
`add_decayed_weights`), trust-ratio methods (`lamb`, `lars`), and learning-rate-
free methods (`optax.contrib.prodigy`, `dadapt_adamw`). For example, a
learning-rate-free run (no LR to tune; recommended with a `1.0 -> 0` schedule):

```python
import optax

steps = max_epoch * steps_per_epoch
optimizer = optax.contrib.prodigy(
    learning_rate=optax.linear_schedule(1.0, 0.0, steps),  # 1.0 = base/max
    safeguard_warmup=True,
)
train(model, data, conf=trainer_conf, optimizer=optimizer)
```

### Freezing Parameters

Use `freeze_paths` to freeze arbitrary leaves declaratively in serialized
configs:

```python
trainer_conf = {
    "freeze_paths": ["noise"],  # freeze process noise
}
```

## Default Optimizer

The default optimizer is **vanilla Adam** -- `optax.adam(conf.learning_rate)`,
nothing more:

- **no gradient clipping**
- **no gradient noise**
- **no weight decay**

This is a deliberate "no policy" default. In a plugin framework the trainer
cannot know which leaves are weight matrices vs variances/biases (so it
imposes no weight decay), and gradient noise/clipping can destabilize
sensitive objectives -- e.g. chaotic dynamical-systems reconstruction, where
injected gradient noise prevented the model from settling onto the true
attractor. Anything beyond plain Adam is opt-in: build your own `optax`
optimizer and pass it via `optimizer=...` (e.g. add `optax.clip_by_global_norm`,
`optax.add_decayed_weights`, or `optax.add_noise` to a chain -- see the custom
optimizer example above).

## Scheduling a Model Attribute (`param_schedule`)

`on_epoch_end`, `regularizer`, and `optimizer` cover per-epoch policy,
additive loss terms, and the update rule. A fourth extension point,
`param_schedule`, drives an arbitrary **SGD-managed** model attribute through
a step-indexed schedule.

`param_schedule(model, step) -> model` is called at the **start of every
step**, before the loss/gradient computation, so the scheduled value is what
the loss is evaluated on and what persists in the returned model. It is a
general mechanism — the trainer only calls the function and does not
interpret what it changes.

Do **not** schedule `noise` while `MVNNoiseMstep` is selected. That
plugin owns Q and would overwrite a scheduled value. With `post_optimizer_transforms=()`, Q is
SGD-managed and may be scheduled explicitly if the path is also frozen from
optimizer updates.

**`freeze_paths` is required, not optional, when using `param_schedule`.**
Without it, the optimizer's own gradient-based update (plus gradient noise
and Adam momentum) will perturb the scheduled attribute away from its target
value within the same step — the schedule and the optimizer fight each other.
Since `param_schedule` is an opaque callable, the trainer cannot detect which
paths it touches and enforce this automatically.

To write a custom schedule (e.g. for a different attribute or shape), the
pattern is:

```python
import equinox as eqx

def my_schedule(model, step):
    value = my_optax_schedule(step)
    return eqx.tree_at(lambda m: m.some_attribute, model, value)
```

**If the scheduled attribute is stored in a constrained/free-form
(unconstrained) parameterization**, anneal in its constrained space, not
its free-form space, and convert only as the last step. The free-form
encoding is typically nonlinear (e.g. a sqrt or inverse-softplus transform),
so interpolating free-form values directly traces a different, distorted path
through the intended constrained space.

## Trainer-owned Observation-Noise and Transition-Noise Updates (`post_optimizer_transforms`)

For a Gaussian-likelihood model, the observation noise covariance `R` is
driven by a closed-form EM M-step instead of gradient descent, avoiding a
Heywood-case degeneracy that plain gradient-based MLE of `R` is prone to
(see [`post_optimizer_transforms`](post_optimizer_transforms.md) for the full specification).
Select the R policy explicitly with `GaussianObservationMstep()`. During
every minibatch's existing **pre-SGD** ELBO forward pass, the plugin derives
an R statistic from the same posterior moments and applies it after SGD,
without an extra inference pass.

The transition/process-noise covariance `Q` (`model.noise`) is
initialized by `MVNNoiseMstep(q_scale=...)` before freeze-mask and optimizer
state creation. That plugin owns the complete Q prior policy: its initializer
sets `Q_init = q_scale * I`, and its additive MAP statistic uses the same
`q_scale` with `q_prior_fraction`; `noise` is automatically frozen from
SGD. Omit `MVNNoiseMstep` to leave `noise` SGD-managed.

All selected transforms consume the same immutable three-value forward output,
then transform the post-optimizer model in explicit order.
`post_optimizer_transforms=()` is optimizer-only training with no
transform-owned freeze paths. The accepted one-step lag means R
reconstructs deterministic readout quantities from posterior moments and the
post-SGD readout state; no additional forward output or inference pass is
introduced.

## Multi-Device Training

When multiple devices are available, training data is automatically sharded
across them using `jax.sharding.NamedSharding`. The batch size must be a
multiple of the device count.

## Epoch Callbacks and the EpochHandler

The training loop is validation agnostic: it runs the full `max_epoch`,
computes the per-epoch training loss, and calls `on_epoch_end(model, info)`
once per finished epoch. Returning a truthy value stops training. `info` is
train-only with plain Python values:

```python
def on_epoch_end(model, info):
    # info: {"epoch", "step", "train_loss", "train_losses"}
    return info["train_loss"] != info["train_loss"]  # stop on NaN

trained = train(model, train_data, conf=trainer_conf, on_epoch_end=on_epoch_end)
```

The callback contract is just a callable. `train` always returns the
final-epoch model and has no notion of "best".

### EpochHandler

`jaxfads.trainer.EpochHandler` is a self-contained handler that bundles the common
policy. You construct it with the validation set and read `best_model`
afterwards:

```python
from jaxfads.trainer import EpochHandler, train, train_test_split
import numpy as np

train_data, valid_data = train_test_split(
    data, rng=np.random.default_rng(0), test_size=64
)

handler = EpochHandler(
    valid_data=valid_data,        # enables validation + best tracking
    checkpoint_path="runs/exp1",  # enables checkpoint/metrics/config writing
    checkpoint_every=10,          # save current model every 10 epochs
    patience=20,                  # early stopping; None disables
    config=trainer_conf,          # dumped to config.yaml
)

train(model, train_data, conf=trainer_conf, on_epoch_end=handler)
best_model = handler.best_model
```

EpochHandler is fully opt-in:

- validation/best tracking require `valid_data`; it builds its own jitted
  evaluation internally, so JAX concerns stay out of user code
- with `checkpoint_path` it writes `checkpoint_epoch{NNNN}.zip`, `best.zip`,
  `metrics.json`, and (if `config` is given) `config.yaml`
- with `patience` it stops when the validation loss fails to improve for that
  many epochs; `patience=None` never stops early

## Train/Validation Split

The trainer does not split data. Split it yourself (the helper
`jaxfads.trainer.train_test_split` is provided) and pass the validation set to
an `EpochHandler`:

```python
from jaxfads.trainer import train_test_split
import numpy as np

train_data, valid_data = train_test_split(
    data, rng=np.random.default_rng(0), test_size=64
)
```

## Loss Function

The per-batch loss is:

```
loss = mean(-ELBO) + regularizer(model)
```

where:
- **ELBO** = E_q[log p(y|z)] − KL(q(z|y) ∥ p(z))
- `regularizer(model)` is an *optional* user-provided callable passed to
  `train(model, data, conf=..., regularizer=...)`. `batch_loss` itself stays a
  pure objective; the trainer composes `loss = -ELBO + regularizer(model)`.

By default no extra regularization term is added. This keeps the core library
agnostic to the latent family (not every `Approx` has a covariance-like noise).

### Example: regularizing the process-noise covariance Q

The penalty must be written in the quantity you intend to regularize. To
regularize the process-noise covariance Q, decode `model.noise` through
`model.approx` rather than penalizing the raw free parameters (which live in a
nonlinear chart and are not a meaningful function of Q):

```python
import jax.numpy as jnp

def q_regularizer(model):
    approx = model.approx
    moment = approx.canon_to_moment(approx.free_to_canon(model.noise))
    _, Q = approx.unpack(moment)          # full (D, D) covariance
    return 1e-4 * jnp.trace(Q)            # well-defined function of Q

train(model, data, conf=trainer_conf, regularizer=q_regularizer)
```


## Logging

Training progress is logged via the `jaxfads` logger. Configure it before
calling `train`:

```python
from jaxfads import configure_logging

configure_logging("INFO")               # console output
configure_logging("DEBUG", file_path="train.log")  # + file output
```

## Checkpointing

Save and load trained models:

```python
from jaxfads import XFADS

XFADS.save(trained_model, "model.zip")
model = XFADS.load("model.zip")
```
