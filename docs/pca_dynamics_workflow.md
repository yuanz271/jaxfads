# Learning Dynamics from PCA Coordinates

This page shows how to use XFADS with frozen PCA readout to learn latent
dynamics from principal components — using the standard MVN posterior, no
special approximation family needed.

See also: [PCA Dynamics Equivalence Proof](pca_dynamics_equivalence.md),
[Quickstart](quickstart.md), [Training Configuration](training.md).

## Motivation

Given high-dimensional observations $y_t$, a common workflow is:

1. Run PCA to get low-dimensional coordinates $z_t^{\mathrm{PCA}}$.
2. Fit a dynamical model $z_t \approx f(z_{t-1})$ to these coordinates.

Step 2 is typically done via MSE regression on PCA coordinates. XFADS can
do the same thing — and more — by treating PCA as a **Gaussian observation
model** within a proper Bayesian state-space framework.

With small observation noise $\sigma^2$, the posterior is close to PCA
coordinates and dynamics estimation is equivalent to MSE regression
([proof](pca_dynamics_equivalence.md)). With moderate $\sigma^2$, the
posterior is **dynamics-smoothed**, yielding more robust estimates.

## Configuration

```python
from omegaconf import OmegaConf

state_dim = 4       # PCA rank
obs_dim = 50        # observation dimensionality
sigma2 = 1e-4       # observation noise (small = near-PCA limit)

conf = OmegaConf.create(
    {
        "mode": "smooth",
        "state_dim": state_dim,
        "observation_dim": obs_dim,
        "seed": 0,
        "mc_size": 4,
        "approx": "MVN",
        "approx_kwargs": {},
        "state_map": "OUStateMap",       # or any dynamics model
        "stepper": "EulerStepper",
        "dropout": 0.0,
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
            "cov": [sigma2] * obs_dim,
            "norm_readout": False,
            # FA with obs_noise_var=0 gives exact PCA loadings.
            "readout_init": "fa",
            "readout_init_conf": {"obs_noise_var": 0.0},
        },
    }
)
```

## Key settings explained

### Observation noise `cov`

Controls how strongly observations pin the posterior to PCA coordinates:

| `sigma2` | Posterior behaviour | Dynamics estimate |
|----------|--------------------|--------------------|
| `1e-6` | ≈ raw PCA projection | ≈ MSE on PCs |
| `1e-4` | near-PCA, numerically stable | near-MSE, stable |
| `1e-1` | dynamics-smoothed | richer than MSE |
| `1.0` | balanced obs/dynamics | full Bayesian |

**Do not use exactly zero** — the Gaussian likelihood becomes
ill-conditioned. Use a small positive value like `1e-4`.

### Readout initialisation

Setting `readout_init_conf.obs_noise_var = 0.0` makes the FA initialiser
produce **exact PCA loadings** (no noise subtraction from the data
covariance). This is documented in the FA initialiser:
"Use `0.0` to degrade gracefully to PCA."

### Freezing readout and observation noise

To keep PCA loadings and observation noise fixed during training:

```python
trainer_conf = OmegaConf.create(
    {
        "seed": 0,
        "learning_rate": 1e-3,
        "max_epoch": 200,
        "batch_size": 32,
        "validation_size": 32,
        "freeze_paths": [
            "observation.readout",
            "observation.likelihood",
        ],
    }
)
```

This freezes:
- `observation.readout` — PCA loading matrix $C$ and bias $b$
- `observation.likelihood` — observation noise $\sigma^2$

Trainable parameters are then only:
- dynamics (`state_map`)
- state noise (`noise_free`)
- encoders (`alpha_encoder`, `beta_encoder`)
- prior (`unconstrained_prior_natural`)

## Training and inference

```python
import jax.random as jr
from jaxfads import XFADS
from jaxfads.trainer import train
from jaxfads.observations import GLM  # noqa: F401

model = XFADS(conf, jr.key(0)).initialize(*data)
trained = train(model, data, conf=trainer_conf)

# Inference
natural_params, moment_params, predictions = trained(
    times, observations, controls, covariates, key=jr.key(1)
)
```

## Interpretation

- **`moment_params`** — posterior moments at each timestep. For small
  $\sigma^2$, the posterior mean is close to PCA coordinates. For larger
  $\sigma^2$, it is dynamics-smoothed.

- **`predictions`** — predictive moments from the learned dynamics model.
  The KL term in the ELBO compares posterior to prediction, which is how
  dynamics receive gradient signal.

- The learned dynamics $f$ and state noise $Q$ describe the temporal
  structure of the latent process in PCA coordinates.

## Equivalence to MSE regression

In the limit $\sigma^2 \to 0$ with fixed $Q$, the ELBO dynamics gradient
equals the MSE gradient on PCA coordinates
([proof](pca_dynamics_equivalence.md)). With finite $\sigma^2$, the XFADS
estimate is strictly richer because dynamics also receive gradient through
the posterior's dependence on $f$.
