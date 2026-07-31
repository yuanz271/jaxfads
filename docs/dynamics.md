# Writing Custom Dynamics and Integrators

This guide explains how latent dynamics are defined in XFADS after the
`Dynamics + Integrator` refactor.

See also: [Quickstart](quickstart.md), [Training](training.md),
[Algorithm](algorithm.md).

## Overview

XFADS composes latent transition updates as:

```
z_{t+1} = Integrator(Dynamics, z_t, u_t, c_t)
```

- `Dynamics` defines the system map.
- `Integrator` defines how one-step updates are produced.
- Process noise remains owned by `XFADS` through top-level `q_scale`.

Choose the dynamics/integrator pair that matches your model: if your
`Dynamics.eval(...)` returns a derivative, use `Euler` or `RK4`; if it returns
the next state directly, use `Identity`.

## Minimal Dynamics Example

```python
from jaxfads.base import Dynamics


class MyDynamics(Dynamics):
    def __init__(self, conf, key):
        self.conf = conf

    def eval(self, z, u, c, *, key=None):
        return z
```

## Built-in Integrators

- `Identity` (pass-through integrator, no `dt`)
- `Euler` (for derivative-returning dynamics, requires `dyn_conf.dt`)
- `RK4` (for derivative-returning dynamics, requires `dyn_conf.dt`)

## Built-in Dynamics

- `Identity`: pass-through dynamics map `z_{t+1} = z_t`
- `OU`: mean-reverting drift map `dz/dt = -theta * z`
- `Functional`: wraps a non-trainable callable

`Functional` only accepts:
- plain Python functions (including lambdas bound to module-level names)
- bound methods
- `functools.partial` of the above

It intentionally rejects arbitrary callable objects.
Set callable import path under `dyn_conf.fn_path` (`module:function`) and
optional kwargs under `dyn_conf.fn_kwargs`.
When running an example/script directly, use `__main__:symbol_name` for
`fn_path` (for example `__main__:vdp_dynamics`).

Example:

```yaml
dynamics: Functional
integrator: Identity
q_scale: 1.0

dyn_conf:
  fn_path: my_pkg.my_maps:my_transition
  fn_kwargs: {gain: 1.0}
  input_dim: 0
  context_dim: 0
```

## Configuration Example (Continuous)

```yaml
dynamics: OU
integrator: Euler
q_scale: 1.0

dyn_conf:
  theta: 2.0
  dt: 0.04
  input_dim: 0
  context_dim: 0
```

## Configuration Example (Identity / random-walk baseline)

```yaml
dynamics: Identity
integrator: Identity
q_scale: 1.0

dyn_conf:
  input_dim: 0
  context_dim: 0
```

Use this as the weakest built-in dynamics prior when you want
`z_{t+1} \approx z_t + noise`.

## Configuration Example (Discrete custom map)

```yaml
dynamics: MyDynamics
integrator: Identity
q_scale: 1.0

dyn_conf:
  input_dim: 0
  context_dim: 0
```

## Registration

Both `Dynamics` and `Integrator` subclasses are registered by class name.
Ensure custom modules are imported before constructing `XFADS`.
