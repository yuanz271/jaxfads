"""Run one deterministic XFADS device-parity case in a fresh process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from omegaconf import OmegaConf

from jaxfads import XFADS
from jaxfads.observations import GLM  # noqa: F401 - register observation
from jaxfads.training import batch_loss, train


def _model_conf():
    return OmegaConf.create({
        "mode": "smooth",
        "observation_dim": 4,
        "state_dim": 2,
        "dynamics": "Identity",
        "integrator": "Identity",
        "approx": "MVN",
        "approx_kwargs": {},
        "mc_size": 3,
        "seed": 17,
        "n_steps": 6,
        "dropout": 0.0,
        "dyn_conf": {"input_dim": 0, "context_dim": 0},
        "enc_conf": {"width": 8, "depth": 1, "dropout": 0.0},
        "obs_conf": {
            "model": "GLM",
            "likelihood": "Gaussian",
            "cov": [0.2] * 4,
            "norm_readout": False,
            "readout_init": None,
        },
    })


def _trainer_conf():
    return OmegaConf.create({
        "max_epoch": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "seed": 23,
        "model_transformations": [
            {"name": "gaussian_observation", "update_rate": 0.2},
            {
                "name": "mvn_noise",
                "q_scale": 0.25,
                "q_prior_fraction": 0.1,
                "update_rate": 0.2,
            },
        ],
    })


def _data():
    key = jr.key(31)
    n_trials, n_steps, observation_dim = 8, 6, 4
    times = jnp.broadcast_to(jnp.arange(n_steps), (n_trials, n_steps))
    observations = jr.normal(key, (n_trials, n_steps, observation_dim))
    controls = jnp.zeros((n_trials, n_steps, 0))
    covariates = jnp.zeros((n_trials, n_steps, 0))
    return times, observations, controls, covariates


def _leaves(model):
    return tuple(jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))


def _decoded_covariances(model):
    approx = model.approx
    moment = approx.canon_to_moment(approx.free_to_canon(model.noise))
    _, q = approx.unpack(moment)
    r = model.observation.likelihood.cov()
    return q, r


def _save_arrays(path: Path, model, inference, loss, q, r):
    arrays = {
        "loss": np.asarray(loss),
        "q": np.asarray(q),
        "r": np.asarray(r),
    }
    arrays.update({
        f"leaf_{i:04d}": np.asarray(leaf) for i, leaf in enumerate(_leaves(model))
    })
    arrays.update({
        f"inference_{i}": np.asarray(value) for i, value in enumerate(inference)
    })
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    devices = jax.devices()
    platform = jax.default_backend()
    model_conf = _model_conf()
    trainer_conf = _trainer_conf()
    data = _data()

    model = XFADS(model_conf, jr.key(41)).initialize(*data)
    trained = train(model, data, conf=trainer_conf)
    test_batch = tuple(value[:2] for value in data)
    inference = trained(*test_batch, key=jr.key(43))
    loss = batch_loss(trained, test_batch, jr.key(47))
    q, r = _decoded_covariances(trained)

    save_path = args.output / "model.zip"
    XFADS.save(trained, save_path)
    loaded = XFADS.load(save_path)
    loaded_inference = loaded(*test_batch, key=jr.key(43))
    np.testing.assert_allclose(
        np.asarray(loss),
        np.asarray(batch_loss(loaded, test_batch, jr.key(47))),
        rtol=1e-6,
        atol=1e-6,
    )
    for original, restored in zip(inference, loaded_inference, strict=True):
        np.testing.assert_allclose(original, restored, rtol=1e-6, atol=1e-6)

    _save_arrays(args.output / "result.npz", trained, inference, loss, q, r)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "backend": platform,
                "device_count": len(devices),
                "devices": [str(device) for device in devices],
                "model_conf": OmegaConf.to_container(model_conf, resolve=True),
                "trainer_conf": OmegaConf.to_container(trainer_conf, resolve=True),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
