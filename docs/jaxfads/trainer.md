# `jaxfads.trainer`

**Source:** `src/jaxfads/trainer.py`

This page describes training configuration and the `train()` entry point.

---

# Training Configuration

This guide explains how to configure and run XFADS training.

## Overview

Training maximises the Evidence Lower Bound (ELBO) over mini-batches using
Optax optimizers, with multi-device data parallelism, gradient clipping, and
validation-based early stopping.

```python
from jaxfads.trainer import train

trained_model = train(model, data, conf=trainer_conf)
```

## Data Format

`data` is a 4-tuple of JAX arrays:

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
| `max_epoch` | `50` | Maximum training epochs |
| `min_epoch` | `0` | Minimum epochs before early stopping |
| `batch_size` | `1` | Mini-batch size (must be divisible by device count) |
| `clip_norm` | `5.0` | Global gradient norm clipping threshold |
| `weight_decay` | `1e-3` | AdamW-style weight decay |
| `noise_eta` | `0.5` | Gradient noise scale (for regularisation) |
| `noise_gamma` | `0.8` | Gradient noise decay exponent |
| `seed` | `0` | Random seed for shuffling and noise |
| `valid_ratio` | `0.2` | Fraction of data used for validation |
| `validation_size` | `80` | Fixed validation set size (overrides `valid_ratio` when > 0) |
| `patience` | auto | Early-stopping patience in epochs; auto-computed from training budget when not provided |

### Example

```python
from omegaconf import OmegaConf

trainer_conf = OmegaConf.create(
    dict(
        seed=42,
        learning_rate=5e-4,
        max_epoch=200,
        batch_size=64,
        clip_norm=5.0,
        weight_decay=1e-3,
        noise_eta=0.0,  # disable gradient noise
        validation_size=64,
    )
)
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

## Early Stopping

Patience is auto-computed from the total training budget:

```
patience_steps = total_steps × 0.1
patience_epochs = max(1, patience_steps / batches_per_epoch)
```

Training stops when the validation loss fails to improve for `patience`
consecutive epochs (after `min_epoch` is reached).

## Train/Validation Split

Data is randomly split before training. You can control the split via:

- `validation_size` — fixed number of validation samples (takes priority)
- `valid_ratio` — fraction of data for validation (used when
  `validation_size` is 0)

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

configure_logging("INFO")  # console output
configure_logging("DEBUG", file_path="train.log")  # + file output
```

## Checkpointing

Save and load trained models:

```python
from jaxfads import XFADS

XFADS.save(trained_model, "model.zip")
model = XFADS.load("model.zip")
```
