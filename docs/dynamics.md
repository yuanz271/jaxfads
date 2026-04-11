# Writing Custom State Maps and Steppers

This guide explains how latent dynamics are defined in XFADS after the
`StateMap + Stepper` refactor.

See also: [Quickstart](quickstart.md), [Training](training.md),
[Algorithm](algorithm.md).

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

- `IdentityStateMap`: discrete random-walk mean map `z_{t+1} = z_t`
- `OUStateMap`: continuous OU drift map `dz/dt = -theta * z`
- `FunctionStateMap`: wraps a non-trainable callable

`FunctionStateMap` only accepts:
- plain Python functions (including lambdas bound to module-level names)
- bound methods
- `functools.partial` of the above

It intentionally rejects arbitrary callable objects.
Set callable import path under `dyn_conf.fn_path` (`module:function`) and
optional kwargs under `dyn_conf.fn_kwargs`.
When running an example/script directly, use `__main__:symbol_name` for
`fn_path` (for example `__main__:vdp_state_map`).

Example:

```yaml
state_map: FunctionStateMap
stepper: DiscreteStepper

dyn_conf:
  system_type: discrete
  fn_path: my_pkg.my_maps:my_transition
  fn_kwargs: {gain: 1.0}
  state_noise: 1.0
  input_dim: 0
  context_dim: 0
```

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

## Configuration Example (Identity / random-walk baseline)

```yaml
state_map: IdentityStateMap
stepper: DiscreteStepper

dyn_conf:
  system_type: discrete
  state_noise: 1.0
  input_dim: 0
  context_dim: 0
```

Use this as the weakest built-in dynamics prior when you want
`z_{t+1} \approx z_t + noise`.

## Configuration Example (Discrete custom map)

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
