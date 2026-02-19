# Writing Custom Dynamics Modules

This guide explains how to create custom state-transition models for XFADS.

## Overview

A dynamics module defines how the latent state evolves over time:

```
z_{t+1} = f(z_t, u_t, c_t) + ε_t,   ε_t ~ N(0, Σ)
```

where `f` is the deterministic transition, `u_t` is a control input, `c_t` is a
covariate, and `Σ` is the process noise covariance.

## Minimal Example

```python
import jax.numpy as jnp
from jaxfads.base import Dynamics, Noise
from jaxfads.dynamics import DiagGaussian

class MyDynamics(Dynamics):
    noise: Noise

    def __init__(self, conf, key):
        self.conf = conf
        self.noise = DiagGaussian(jnp.array(conf.cov), conf.state_dim)

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
| `noise` | `Noise` | Process noise model (must implement `.cov() -> Array`) |
| `conf` | `DictConfig` | Configuration (inherited from `ConfModule`, set in `__init__`) |
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

The noise is handled separately by the framework; `forward` returns only the
deterministic part.

## Noise Model

Use the built-in `DiagGaussian` for diagonal process noise:

```python
from jaxfads.dynamics import DiagGaussian

self.noise = DiagGaussian(jnp.array(conf.cov), conf.state_dim)
```

The covariance is parameterized in unconstrained space via softplus, so it is
always positive and trainable by default.

To implement a custom noise model, satisfy the `Noise` protocol:

```python
class MyNoise:
    def cov(self) -> Array:
        """Return diagonal covariance vector of shape (state_dim,)."""
        ...
```

## Optional Overrides

### `loss() -> Array | float`

Return a scalar regularization term added to the training loss. Default is
`0.0`. A common pattern penalizes large noise:

```python
def loss(self):
    return jnp.mean(self.cov())
```

## Registration and Configuration

Subclasses of `Dynamics` are automatically registered by class name. To use
your dynamics in an XFADS model, set the `forward` config field to the class
name:

```python
conf = OmegaConf.create({
    ...
    "forward": "MyDynamics",
    "dyn_conf": {
        "state_dim": 2,
        "cov": 0.01,
        # any other fields your __init__ reads from conf
    },
    ...
})
```

The framework calls `Dynamics.get_subclass("MyDynamics")(conf.dyn_conf, key=key)`
internally. Your class must be importable at model-creation time (i.e., the
module defining it must have been imported before `XFADS(conf, key)` is called).

## Full Example: Neural Dynamics with Residual Connection

```python
import jax.numpy as jnp
from jax import Array
from jaxfads.base import Dynamics, Noise
from jaxfads.dynamics import DiagGaussian
from jaxfads.nn import make_mlp


class Nonlinear(Dynamics):
    noise: Noise
    f: Callable[..., Array]

    def __init__(self, conf, key):
        self.conf = conf
        self.noise = DiagGaussian(jnp.array(conf.cov), conf.state_dim)
        self.f = make_mlp(
            conf.state_dim + conf.input_dim,
            conf.state_dim,
            conf.width,
            conf.depth,
            key=key,
            final_bias=False,
            dropout=conf.dropout,
        )

    def forward(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return z + self.f(x, key=key)  # residual connection

    def loss(self):
        return jnp.mean(self.cov())
```

## Full Example: Known ODE via RK4

When the dynamics are known analytically, wrap them directly:

```python
class VanDerPol(Dynamics):
    noise: DiagGaussian
    mu: float
    dt: float

    def __init__(self, conf, key):
        self.conf = conf
        self.noise = DiagGaussian(jnp.array(conf.cov), conf.state_dim)
        self.mu = float(conf.mu)
        self.dt = float(conf.dt)

    def forward(self, z, u, c, *, key=None):
        # RK4 integration of Van der Pol oscillator
        def rhs(s):
            return jnp.stack([s[1], self.mu * (1 - s[0]**2) * s[1] - s[0]])

        k1 = rhs(z)
        k2 = rhs(z + 0.5 * self.dt * k1)
        k3 = rhs(z + 0.5 * self.dt * k2)
        k4 = rhs(z + self.dt * k3)
        return z + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def loss(self):
        return jnp.mean(self.cov())
```

## Checklist

- [ ] Subclass `Dynamics`
- [ ] Declare `noise: Noise` (or a concrete type like `DiagGaussian`)
- [ ] Set `self.conf = conf` in `__init__`
- [ ] Implement `forward(z, u, c, *, key=None) -> Array`
- [ ] All trainable parameters are stored as `eqx.Module` fields (arrays or sub-modules)
- [ ] Class is imported before `XFADS(conf, key)` is called
