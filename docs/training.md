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
| `clip_norm` | `5.0` | Global gradient norm clipping threshold |
| `weight_decay` | `1e-3` | AdamW-style weight decay |
| `noise_eta` | `0.5` | Gradient noise scale (for regularisation) |
| `noise_gamma` | `0.8` | Gradient noise decay exponent |
| `seed` | `0` | Random seed for shuffling and noise |
| `freeze_paths` | `[]` | Optional list of dot-separated model attribute paths to freeze (e.g. `["noise_free"]`) |
| `noise_regularizer` | `None` | Optional `Callable[[XFADS], Array]` added to the loss |

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
    clip_norm=5.0,
    weight_decay=1e-3,
    noise_eta=0.0,         # disable gradient noise
))
```

### Freezing Parameters

Use `freeze_paths` to freeze arbitrary leaves declaratively in serialized
configs:

```python
trainer_conf = {
    "freeze_paths": ["noise_free"],  # freeze process noise
}
```

## Optimizer Chain

The default optimizer is an Optax chain:

1. **Clip by global norm** — prevents gradient explosions
2. **Add noise** — Gaussian gradient noise for regularisation
3. **Scale by Adam** — adaptive learning rates
4. **Add decayed weights** — L2 regularisation
5. **Scale by learning rate** — final LR scaling

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
a `EpochHandler`:

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
- `regularizer(model)` is an *optional* user-provided callable set on the
  **trainer configuration** as `noise_regularizer`.

By default no extra regularization term is added. This keeps the core library
agnostic to the latent family (not every `Approx` has a covariance-like noise).

### Example: L2 regularizer on process-noise free parameters

```python
import jax.numpy as jnp

# Penalize the free-form process noise parameters stored on the model.
# This is model/Approx-specific by design.
def l2_noise_regularizer(model):
    return 1e-4 * jnp.sum(model.noise_free**2)

trainer_conf.noise_regularizer = l2_noise_regularizer
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
