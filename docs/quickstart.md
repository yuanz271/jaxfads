# XFADS Quickstart

This page gives a minimal end-to-end setup for model construction, training,
and inference.

## Minimal Configuration

```python
from omegaconf import OmegaConf
import jax.random as jr
import jax.numpy as jnp

from jaxfads import XFADS
from jaxfads.trainer import train

# Ensure built-in subclasses are imported/registered as needed.
from jaxfads.observations import GLM  # noqa: F401

conf = OmegaConf.create(
    {
        "mode": "smooth",  # "filter" | "smooth" | "causal" | "nofilt"
        "state_dim": 4,
        "observation_dim": 10,
        "n_steps": 100,
        "seed": 0,
        "mc_size": 4,
        "approx": "MVN",
        "approx_kwargs": {},  # rank defaults to state_dim (full); use {"rank": r} for low-rank
        "state_map": "OUStateMap",
        "stepper": "EulerStepper",
        "dyn_conf": {
            "system_type": "continuous",
            "theta": 2.0,
            "dt": 0.05,
            "state_noise": 1.0,
            "input_dim": 0,
            "context_dim": 0,
        },
        "enc_conf": {
            "width": 64,
            "depth": 1,
            "dropout": 0.0,
        },
        "obs_conf": {
            "model": "GLM",
            "likelihood": "Gaussian",
            "cov": [1.0] * 10,
            "norm_readout": False,
            "readout_init": "fa",
            "readout_init_conf": {"obs_noise_var": 1.0},
        },
    }
)

# Dummy data with no controls/covariates.
N, T, D_obs = 32, 100, 10
times = jnp.tile(jnp.arange(T)[None, :], (N, 1))
observations = jnp.zeros((N, T, D_obs))
controls = jnp.zeros((N, T, 0))
covariates = jnp.zeros((N, T, 0))
data = (times, observations, controls, covariates)

model = XFADS(conf, jr.key(0)).initialize(*data)

trainer_conf = OmegaConf.create(
    {
        "seed": 0,
        "learning_rate": 1e-3,
        "max_epoch": 50,
        "batch_size": 16,
        "validation_size": 16,
        "freeze_paths": [],
    }
)
trained = train(model, data, conf=trainer_conf)

natural_params, moment_params, predictions = trained(
    times, observations, controls, covariates, key=jr.key(1)
)
```

## Choosing Settings

| Topic | Option | Choose when |
|------|--------|-------------|
| `mode` | `filter` | Need alpha-only filtering natural parameters. |
| `mode` | `smooth` | Default offline smoothing-style inference (`alpha + beta`). |
| `mode` | `causal` | Need Eq. 29-style causal recursion with smoothing reconstruction. |
| `mode` | `nofilt` | Posterior set by custom encoder (e.g. pretrained DR); no filtering recursion. |
| `state_map` | `IdentityStateMap` | Need a null/random-walk dynamics prior `z_{t+1}=z_t+noise`. |
| `stepper` | `DiscreteStepper` | `dyn_conf.system_type="discrete"` and map already returns `z_{t+1}`. |
| `stepper` | `EulerStepper` | Continuous-time map, faster/rougher integration. |
| `stepper` | `RK4Stepper` | Continuous-time map, better local accuracy than Euler. |
| `approx` | `MVN` | Gaussian approximation. Use `approx_kwargs={"rank": r}` for low-rank. |

## Common Pitfalls

1. `FunctionStateMap` import path must be importable:
   use `dyn_conf.fn_path="module:function"`.
1. If running a script directly, use `__main__:symbol_name` in `fn_path`.
1. `batch_size` must be divisible by the number of JAX devices.
1. `validation_size > 0` overrides `valid_ratio`.

## Next Docs

- [Dynamics (`StateMap` + `Stepper`)](dynamics.md)
- [Training Configuration](training.md)
- [Algorithm Overview](algorithm.md)
- [Paper Parity Review](paper_parity_2403_01371.md)
- [Learning Dynamics from PCA Coordinates](pca_dynamics_workflow.md)
