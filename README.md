# eXponential FAmily Dynamical Systems (XFADS)

A JAX-based implementation of [XFADS](https://github.com/catniplab/xfads) for variational Bayesian state-space modeling with exponential family approximations. It pairs expressive neural parameterizations with efficient filtering and smoothing routines, enabling scalable inference on high-dimensional time series across CPU, GPU, and TPU.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Examples](#examples)
- [Citation](#citation)

## Installation

```bash
pip install git+https://github.com/yuanz271/jaxfads.git
```

All dependencies—including `gearax` (pinned to a specific commit)—are resolved automatically.

> **GPU/TPU users:** install the appropriate `jax[cuda]` or `jax[tpu]` variant *before* installing jaxfads. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Documentation

- [Writing Custom Dynamics Modules](docs/dynamics.md)
- [Training Configuration](docs/training.md)

## Examples

- [`examples/vdp_example.py`](examples/vdp_example.py) — End-to-end Van der Pol oscillator demo covering model configuration, training, and inference with both exact and learned dynamics, Factor Analysis readout initialisation, Procrustes evaluation, and flow-field visualisation. **Start here** to see the full workflow.

## Citation

If you use XFADS in academic work, please cite:

```bibtex
@inproceedings{NEURIPS2024_18595bc3,
  author    = {Dowling, Matthew and Zhao, Yuan and Park, Il Memming},
  booktitle = {Advances in Neural Information Processing Systems},
  title     = {eXponential FAmily Dynamical Systems ({XFADS}):
               Large-scale nonlinear {G}aussian state-space modeling},
  volume    = {37},
  pages     = {13458--13488},
  year      = {2024},
  publisher = {Curran Associates, Inc.},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2024/file/18595bc3e802a3b11035927fd928eb9c-Paper-Conference.pdf},
}
```

**Resources:**
[Paper](https://papers.neurips.cc/paper_files/paper/2024/hash/18595bc3e802a3b11035927fd928eb9c-Abstract-Conference.html)
