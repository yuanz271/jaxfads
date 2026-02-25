# Writing Custom Dynamics Modules

This guide explains how to create custom deterministic state-transition models
for XFADS.

## Overview

A dynamics module defines the deterministic transition:

```
z_{t+1} = f(z_t, u_t, c_t)
```

Process noise is owned by `XFADS` (configured via `dyn_conf.state_noise`) so
that `Dynamics` stays a pure transition function.

## Minimal Example

```python
from jaxfads.base import Dynamics


class MyDynamics(Dynamics):
    def __init__(self, conf, key):
        self.conf = conf

    def forward(self, z, u, c, *, key=None):
        # Your transition function here.
        # Must return an array of shape (state_dim,).
        return z  # identity dynamics as placeholder
```

That is the complete contract. Everything else is optional.

## Required Interface

### `Dynamics` base class

`Dynamics` inherits from `ConfModule` (an `eqx.Module` with a static `conf`
field) and `SubclassRegistryMixin` (automatic name-based lookup).

You must provide:

| Member | Type | Purpose |
|--------|------|---------|
| `conf` | `DictConfig` | Configuration (set in `__init__`) |
| `forward(z, u, c, *, key=None)` | method | Deterministic state transition |

The `__init__` signature must be `(self, conf, key)` where `conf` is an
OmegaConf `DictConfig` and `key` is a JAX PRNG key.

### `forward` method

```python
def forward(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
```

- **`z`** — current state, shape `(state_dim,)`
- **`u`** — control input, shape `(input_dim,)` (may be empty: `(0,)`)
- **`c`** — covariate, shape `(covariate_dim,)` (may be empty: `(0,)`)
- **`key`** — optional PRNG key for stochastic components (e.g., dropout)
- **returns** — next state mean, shape `(state_dim,)`

## Process noise

Process noise is configured on the XFADS model (not on `Dynamics`). In configs
this is typically done via:

```yaml
dyn_conf:
  state_noise: 1.0
```

The exact parameterization of the process noise is determined by the chosen
`Approx` family.

## Built-in dynamics

### `OUDynamics`

`OUDynamics` is a ready-to-use zero-mean Ornstein–Uhlenbeck drift model. It is a
common choice as a smooth tracking prior.

Drift (Euler discretization):

```
z_next = z + dt * (-theta * z)
```

Example configuration:

```yaml
forward: OUDynamics
state_dim: 2
observation_dim: 10
approx: MVN
approx_kwargs: {structure: full}

# Model-owned process noise
# (scale this with dt if you want an SDE-consistent interpretation)
dyn_conf:
  theta: 2.0
  dt: 0.04
  state_noise: 1.0
  input_dim: 0
  context_dim: 0
```

## Registration and Configuration

Subclasses of `Dynamics` are automatically registered by class name. Built-in
models under `jaxfads.dynamics` are imported by `XFADS` for side-effect
registration.

For custom dynamics, ensure the module defining the class is imported before
constructing `XFADS(conf, key)`.

## Checklist

- [ ] Subclass `Dynamics`
- [ ] Set `self.conf = conf` in `__init__`
- [ ] Implement `forward(z, u, c, *, key=None) -> Array`
- [ ] All trainable parameters are stored as `eqx.Module` fields (arrays or sub-modules)
- [ ] Class is imported before `XFADS(conf, key)` is called
