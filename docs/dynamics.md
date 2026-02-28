# Writing Custom State Maps and Steppers

This guide explains how latent dynamics are defined in XFADS after the
`StateMap + Stepper` refactor.

## Overview

XFADS composes latent transition updates as:

```
z_{t+1} = Stepper(StateMap, z_t, u_t, c_t)
```

- `StateMap` defines the system map.
- `Stepper` defines how one-step updates are produced.
- Process noise remains owned by `XFADS` (`dyn_conf.state_noise`).

## System Types

Set `dyn_conf.system_type` to choose semantics:

- `continuous`: `StateMap.eval(...)` returns `dz/dt`.
- `discrete`: `StateMap.eval(...)` returns `z_{t+1}` directly.

## Minimal StateMap Example

```python
from jaxfads.base import StateMap


class MyStateMap(StateMap):
    def __init__(self, conf, key):
        self.conf = conf

    def eval(self, z, u, c, *, key=None):
        return z
```

## Built-in Steppers

- `EulerStepper` (continuous, requires `dyn_conf.dt`)
- `RK4Stepper` (continuous, requires `dyn_conf.dt`)
- `DiscreteStepper` (discrete, no `dt`)

## Built-in StateMap

- `OUStateMap`: continuous OU drift map `dz/dt = -theta * z`

## Configuration Example (Continuous)

```yaml
state_map: OUStateMap
stepper: EulerStepper

dyn_conf:
  system_type: continuous
  theta: 2.0
  dt: 0.04
  state_noise: 1.0
  input_dim: 0
  context_dim: 0
```

## Configuration Example (Discrete)

```yaml
state_map: MyStateMap
stepper: DiscreteStepper

dyn_conf:
  system_type: discrete
  state_noise: 1.0
  input_dim: 0
  context_dim: 0
```

## Registration

Both `StateMap` and `Stepper` subclasses are registered by class name.
Ensure custom modules are imported before constructing `XFADS`.
