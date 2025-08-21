# XFADS

**eXponential FAmily Dynamical Systems**

XFADS, built on JAX and Equinox.
For more information, see the [paper](https://arxiv.org/abs/2403.01371) and [PyTorch implementation](https://github.com/catniplab/xfads).

## Overview

XFADS provides a flexible framework for modeling and inference in dynamical systems where the state evolution follows exponential family distributions.
The library implements advanced variational inference techniques with support for:

- **Exponential Family Distributions**: Gaussian and other exponential family distributions
- **Variational Smoothing**: Forward-backward algorithms with pseudo and bi-filter modes
- **Neural Network Integration**: Deep learning components for dynamics and observation models
- **JAX Acceleration**: Fast computation with automatic differentiation and GPU/TPU support
- **Modular Design**: Extensible architecture for custom dynamics and observation models

## Installation

### Requirements

- Python >= 3.11
- JAX
- Equinox
- Other dependencies listed in `pyproject.toml`

### Install from Source

```bash
git clone --recursive https://github.com/yuanz271/jaxfads.git
cd jaxfads
pip install -e ".[dev]"
```
