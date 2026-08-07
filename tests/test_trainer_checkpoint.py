import json

import numpy as np
from conftest import MockDynamics  # noqa: F401 - class registration side-effect
from jax import numpy as jnp
from jax import random as jrnd
from omegaconf import OmegaConf

import jaxfads.observations  # noqa: F401 — register GLM subclass
from jaxfads.training import EpochHandler, train, train_test_split


def _sample_data():
    key = jrnd.key(0)
    n_trials, n_timesteps, obs_dim = 16, 8, 6
    times = jnp.broadcast_to(jnp.arange(n_timesteps), (n_trials, n_timesteps))
    observations = jrnd.poisson(key, jnp.ones((n_trials, n_timesteps, obs_dim)))
    controls = jnp.zeros((n_trials, n_timesteps, 0))
    contexts = jnp.zeros((n_trials, n_timesteps, 0))
    return times, observations, controls, contexts


def _split(data, valid_size=4):
    rng = np.random.default_rng(0)
    return train_test_split(data, rng=rng, test_size=valid_size)


def _model_conf(obs_dim=6, state_dim=2):
    return OmegaConf.create({
        "mode": "smooth",
        "observation_dim": obs_dim,
        "state_dim": state_dim,
        "dynamics": "MockDynamics",
        "integrator": "Identity",
        "approx": "MVN",
        "approx_kwargs": {},
        "mc_size": 1,
        "seed": 0,
        "n_steps": 8,
        "fb_penalty": 0,
        "noise_penalty": 0.0,
        "dropout": 0.0,
        "dyn_conf": OmegaConf.create({"input_dim": 0, "context_dim": 0}),
        "enc_conf": OmegaConf.create({"width": 8, "depth": 1, "dropout": 0.0}),
        "obs_conf": OmegaConf.create({
            "model": "GLM",
            "emission_noise": 1.0,
            "norm_readout": False,
            "dropout": 0.0,
            "likelihood": "Poisson",
        }),
    })


def _trainer_conf(**overrides):
    base = {
        "max_epoch": 3,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "seed": 0,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _new_model():
    from jaxfads.smoother import XFADS

    return XFADS(_model_conf(), jrnd.key(0))


def _assert_loadable_checkpoint(path, batch):
    from jaxfads.smoother import XFADS

    loaded = XFADS.load(path)
    assert loaded.noise is not None
    free_energy, post_moments, prior_moments = loaded(*batch, key=jrnd.key(123))
    assert jnp.isfinite(free_energy).all()
    assert jnp.isfinite(post_moments).all()
    assert jnp.isfinite(prior_moments).all()


def test_monitor_writes_artifacts(tmp_path):
    train_data, valid_data = _split(_sample_data())
    model = _new_model()
    conf = _trainer_conf()

    handler = EpochHandler(
        valid_data=valid_data,
        checkpoint_path=str(tmp_path),
        checkpoint_every=1,
        config=conf,
    )
    trained = train(model, train_data, conf=conf, on_epoch_end=handler)

    assert trained is not None
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "best.zip").exists()
    checkpoints = sorted(tmp_path.glob("checkpoint_epoch*.zip"))
    assert len(checkpoints) == 3  # checkpoint_every=1, max_epoch=3

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics["train_losses"]) == 3
    assert len(metrics["valid_losses"]) == 3
    assert handler.best_model is not None

    batch = tuple(x[:2] for x in train_data)
    _assert_loadable_checkpoint(tmp_path / "best.zip", batch)
    for checkpoint in checkpoints:
        _assert_loadable_checkpoint(checkpoint, batch)


def test_no_handler_writes_nothing(tmp_path):
    train_data, _ = _split(_sample_data())
    model = _new_model()

    train(model, train_data, conf=_trainer_conf())

    assert list(tmp_path.iterdir()) == []


def test_monitor_without_validation(tmp_path):
    train_data, _ = _split(_sample_data())
    model = _new_model()

    handler = EpochHandler(
        valid_data=None, checkpoint_path=str(tmp_path), checkpoint_every=1
    )
    train(model, train_data, conf=_trainer_conf(), on_epoch_end=handler)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics["train_losses"]) == 3
    assert metrics["valid_losses"] == []
    assert handler.best_model is None


def test_callback_early_stop():
    train_data, _ = _split(_sample_data())
    model = _new_model()

    calls = []

    def on_epoch_end(m, info):
        calls.append(info["epoch"])
        return True  # stop after the first finalized epoch

    trained = train(
        model, train_data, conf=_trainer_conf(max_epoch=10), on_epoch_end=on_epoch_end
    )

    assert trained is not None
    assert calls == [0]


def test_callback_receives_train_only_info():
    train_data, _ = _split(_sample_data())
    model = _new_model()

    seen = []

    def on_epoch_end(m, info):
        seen.append(info)
        return False

    train(model, train_data, conf=_trainer_conf(max_epoch=2), on_epoch_end=on_epoch_end)

    assert [d["epoch"] for d in seen] == [0, 1]
    for d in seen:
        assert isinstance(d["train_loss"], float)
        assert isinstance(d["grad_norm"], float)
        assert set(d.keys()) == {
            "epoch", "step", "train_loss", "train_losses", "grad_norm", "grad_norms",
        }
