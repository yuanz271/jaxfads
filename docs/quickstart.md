# XFADS Quickstart

This page gives a minimal end-to-end setup for model construction, training,
and inference.

## Minimal Configuration

```python
from omegaconf import OmegaConf
import jax.random as jr
import jax.numpy as jnp

from jaxfads import XFADS
from jaxfads.training import GaussianObservationMstep, train

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
        "dynamics": "OU",
        "integrator": "Euler",
        "dyn_conf": {
            "theta": 2.0,
            "dt": 0.05,
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
        "q_mstep": True,
        "q_scale": 1.0,
        "q_prior_fraction": 0.1,
        "freeze_paths": [],
    }
)
# Minimal run trains on all data and returns the final-epoch model.
# For validation/checkpointing/early stopping, split the data and pass a
# `EpochHandler` as `on_epoch_end` (see Training Configuration).
trained = train(
    model,
    data,
    conf=trainer_conf,
    post_optimizer_transforms=(GaussianObservationMstep(),),
)

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
| `dynamics` | `Identity` | Need a null/random-walk dynamics prior `z_{t+1}=z_t+noise`. |
| `integrator` | `Identity` | Discrete-time map already returns `z_{t+1}`. |
| `integrator` | `Euler` | Continuous-time map, faster/rougher integration. |
| `integrator` | `RK4` | Continuous-time map, better local accuracy than Euler. |
| `approx` | `MVN` | Gaussian approximation. Use `approx_kwargs={"rank": r}` for low-rank. |

## Common Pitfalls

1. `Functional` import path must be importable:
   use `dyn_conf.fn_path="module:function"`.
1. If running a script directly, use `__main__:symbol_name` in `fn_path`.
1. `batch_size` must be divisible by the number of JAX devices.
1. The trainer does not split data; build `valid_data` yourself and pass a
   `EpochHandler` for validation/best tracking.

## Next Docs

- [Dynamics (`Dynamics` + `Integrator`)](dynamics.md)
- [Training Configuration](training.md)
- [Algorithm Overview](algorithm.md)
- [Paper Parity Review](paper_parity_2403_01371.md)
- [Learning Dynamics from PCA Coordinates](pca_dynamics_workflow.md)
