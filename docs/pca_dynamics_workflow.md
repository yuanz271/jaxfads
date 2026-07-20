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
        "dynamics": "OU",       # or any dynamics model
        "integrator": "Euler",
        "dropout": 0.0,
        "dyn_conf": {
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
- dynamics (`dynamics`)
- state noise (`noise_free`)
- encoders (`alpha_encoder`, `beta_encoder`)
- prior (`unconstrained_prior_natural`)

## Training and inference

```python
import numpy as np
import jax.random as jr
from jaxfads import XFADS
from jaxfads.trainer import EpochHandler, train, train_test_split
from jaxfads.observations import GLM  # noqa: F401

train_data, valid_data = train_test_split(
    data, rng=np.random.default_rng(0), test_size=32
)
model = XFADS(conf, jr.key(0)).initialize(*train_data)

handler = EpochHandler(valid_data=valid_data)
train(model, train_data, conf=trainer_conf, on_epoch_end=handler)
trained = handler.best_model

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

## Alternative: NOFILT mode

The workflow above uses `mode="smooth"` with small observation noise so the
encoder learns posteriors near PCA coordinates while filtering still runs.
An alternative is `mode="nofilt"`, which **bypasses filtering entirely**: the
posterior at each timestep is set directly by a user-defined encoder, and
dynamics are trained purely through the KL term.

### When to use NOFILT

- You have a **pretrained** dimensionality reduction (PCA, autoencoder, etc.)
  and want to fit dynamics to its output without the filtering recursion
  influencing the latent trajectory.
- The encoder is non-trainable — its output defines the latent coordinates.

### Defining a custom encoder

In NOFILT mode you provide your own `Encoder` subclass. The encoder maps a
single observation to a point estimate of the latent state:

```python
import jax.numpy as jnp
from jax import Array
from jaxfads.base import Encoder

class PCAEncoder(Encoder):
    """Project observations to PCA coordinates."""
    weight: Array   # (obs_dim, state_dim) — PCA loadings
    bias: Array     # (obs_dim,) — observation mean

    def __init__(self, conf, key=None):
        self.conf = conf
        # Load your pretrained PCA parameters here
        self.weight = ...  # C_pca
        self.bias = ...    # mean_y

    def __call__(self, y: Array, *, key=None) -> Array:
        return (y - self.bias) @ self.weight
```

### Configuration

```python
conf = OmegaConf.create(
    {
        "mode": "nofilt",
        "state_dim": state_dim,
        "observation_dim": obs_dim,
        "nofilt_eps": 1e-6,   # tight MVN variance wrapping point estimates
        "enc_conf": {
            "alpha_encoder": "PCAEncoder",
        },
        # ... dynamics and observation config as before ...
    }
)
```

Key differences from the `smooth` workflow:
- `mode: "nofilt"` — no filtering recursion.
- `enc_conf.alpha_encoder` — name of your custom `Encoder` subclass.
- `nofilt_eps` — the point estimate is wrapped as `MVN(z_hat, eps * I)` for
  framework compatibility (sampling, KL computation).
- No beta encoder is constructed.
- The encoder is typically frozen via `freeze_paths: ["alpha_encoder"]`.

### How it works

1. Your encoder maps each observation to a point estimate `z_hat`.
2. XFADS wraps it as a tight MVN: `q(z_t) = N(z_hat, eps * I)`.
3. Dynamics predictions `p(z_t | z_{t-1})` are computed in parallel (no
   sequential filtering).
4. The ELBO's KL term `KL(q_t || p_pred_t)` trains the dynamics to predict
   the next encoder output.
5. The observation log-likelihood term still contributes if the readout is
   trainable.

### Identity baseline

For PCA / NOFILT experiments, `Identity` is a good no-op
integrator for discrete-time dynamics:

```python
"dynamics": "Identity",
"integrator": "Identity",
"dyn_conf": {
    "state_noise": 1.0,
    "input_dim": 0,
    "context_dim": 0,
},
```

This corresponds to the random-walk prior `z_{t+1} = z_t + noise`. Use it to
check whether a learned dynamics model improves over simple persistence in PCA
coordinates.
