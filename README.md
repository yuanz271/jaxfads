# XFADS

**eXponential FAmily Dynamical Systems**

XFADS is a JAX-based library for Bayesian state-space modeling using variational inference with exponential family approximations. It combines expressive neural parameterizations with efficient filtering and smoothing routines for high-dimensional time series, leveraging automatic differentiation and accelerator support.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Documentation](#documentation)
- [Examples](#examples)
- [Citation & Resources](#citation--resources)

## Overview

XFADS provides a unified framework for building differentiable dynamical systems where both the latent transitions and observation models can be learned. The package focuses on scalable variational smoothing techniques, enabling practitioners to prototype custom exponential-family models without re-implementing inference algorithms.

## Features

- **Expressive exponential-family models** covering Gaussian, Poisson, and extensible custom distributions.
- **Variational smoothing** with forward–backward pseudo-filter and bi-filter routines.
- **Neural parameterizations** for dynamics, observations, and approximate posteriors using Equinox modules.
- **Accelerated JAX execution** with support for automatic differentiation on CPU, GPU, and TPU.
- **Modular design** that decouples core filtering primitives from task-specific components.

## Quick Start

```python
import jax.numpy as jnp
import jax.random as jr
from omegaconf import OmegaConf

from jaxfads import XFADS
from jaxfads.observations import GLM  # registers GLM observation model
from jaxfads.trainer import train

# Dimensions
state_dim, obs_dim, T, N = 2, 10, 100, 64

# Model configuration
conf = OmegaConf.create(dict(
    mode="pseudo", state_dim=state_dim, observation_dim=obs_dim,
    approx="DiagMVN", forward="Nonlinear", mc_size=4,
    seed=0, n_steps=T, dropout=0.0,
    fb_penalty=0.0, noise_penalty=0.01,
    enc_conf=dict(
        observation_dim=obs_dim, state_dim=state_dim,
        approx="DiagMVN", width=32, depth=2, dropout=None,
    ),
    obs_conf=dict(
        model="GLM", likelihood="Poisson",
        observation_dim=obs_dim, state_dim=state_dim,
        norm_readout=False,
    ),
    dyn_conf=dict(
        state_dim=state_dim, input_dim=0, context_dim=0,
        cov=0.1, width=32, depth=1,
    ),
))

# Create and initialise model
model = XFADS(conf, jr.key(0))
data = (
    jnp.broadcast_to(jnp.arange(T), (N, T)),       # times
    jr.poisson(jr.key(1), 5.0, (N, T, obs_dim)),    # observations
    jnp.zeros((N, T, 0)),                            # controls
    jnp.zeros((N, T, 0)),                            # covariates
)
model = model.initialize(*data)

# Train
trainer_conf = OmegaConf.create(dict(
    seed=0, learning_rate=1e-3, max_epoch=50, batch_size=32,
    validation_size=32,
))
trained = train(model, data, conf=trainer_conf)

# Inference
key = jr.key(42)
natural, moments, predictions = trained(*data, key=key)
```

## Installation

```bash
pip install git+https://github.com/yuanz271/jaxfads.git
```

`gearax` is installed from its GitHub repository at a pinned commit during installation.

## Documentation

- [Writing Custom Dynamics Modules](docs/dynamics.md)
- [Training Configuration](docs/training.md)

## Examples

- [`examples/vdp_example.py`](examples/vdp_example.py) — Van der Pol oscillator with both exact and learned dynamics, Factor Analysis readout initialization, Procrustes evaluation, and flow-field visualization.

## Citation & Resources

If you build on XFADS in academic work, please cite the accompanying paper.

```bibtex
@inproceedings{NEURIPS2024_18595bc3,
 author = {Dowling, Matthew and Zhao, Yuan and Park, Il Memming},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/079017-0430},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {13458--13488},
 publisher = {Curran Associates, Inc.},
 title = {eXponential FAmily Dynamical Systems (XFADS): Large-scale nonlinear Gaussian state-space modeling},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/18595bc3e802a3b11035927fd928eb9c-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```

Supplementary resources:
- [Paper](https://papers.neurips.cc/paper_files/paper/2024/hash/18595bc3e802a3b11035927fd928eb9c-Abstract-Conference.html)
- [PyTorch implementation](https://github.com/catniplab/xfads)
