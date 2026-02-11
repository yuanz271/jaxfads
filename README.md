# XFADS

**eXponential FAmily Dynamical Systems**

XFADS is a JAX-based library for Bayesian state-space modeling using variational inference with exponential family approximations. It combines expressive neural parameterizations with efficient filtering and smoothing routines for high-dimensional time series, leveraging automatic differentiation and accelerator support.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Layout](#project-layout)
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
import jax.random as jr
from omegaconf import DictConfig

from jaxfads import XFADS

conf = DictConfig({
    "state_dim": 10,
    "observation_dim": 50,
    "mc_size": 100,
    "approx": "DiagMVN",
    "forward": "Linear",
    "observation": "Poisson",
})

key = jr.key(0)
model = XFADS(conf, key)

# observations: array with shape (time_steps, observation_dim)
posterior = model.smooth(observations)
```

The returned posterior distribution object exposes smoothed means and covariances that can be decoded or fed into downstream tasks.

For a complete, runnable example with synthetic data (2D Van der Pol latents and 10D Gaussian observations), see [`examples/toy_example.py`](examples/toy_example.py).

## Installation

```bash
git clone --recursive https://github.com/yuanz271/jaxfads.git
cd jaxfads
pip install -e ".[dev]"
```

## Project Layout

- `src/jaxfads/core.py`: Filtering primitives and shared interfaces.
- `src/jaxfads/smoother.py`: XFADS orchestrator and smoothing logic.
- `src/jaxfads/dynamics.py`: State-transition parameterizations.
- `src/jaxfads/observations.py`: Observation model components.
- `src/jaxfads/nn.py` and `src/jaxfads/encoders.py`: Neural network building blocks.
- `tests/`: Unit tests mirroring the public API surface.

## Citation & Resources

If you build on XFADS in academic work, please cite the accompanying [paper](https://papers.neurips.cc/paper_files/paper/2024/hash/18595bc3e802a3b11035927fd928eb9c-Abstract-Conference.html). Refer to the official arXiv entry for the most up-to-date citation details. Supplementary resources:

- [Paper](https://papers.neurips.cc/paper_files/paper/2024/hash/18595bc3e802a3b11035927fd928eb9c-Abstract-Conference.html)
- [PyTorch implementation](https://github.com/catniplab/xfads)
