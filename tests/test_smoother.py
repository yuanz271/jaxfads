from typing import override
from pathlib import Path
from tempfile import TemporaryDirectory

from jax import Array, random as jr
from omegaconf import OmegaConf, DictConfig
import equinox as eqx
from jaxfads.base import Dynamics
from jaxfads.smoother import XFADS


class Mock(Dynamics):
    """Mock dynamics — pure deterministic transition."""

    layer: eqx.Module | None

    def __init__(self, conf: DictConfig, key: Array):
        self.conf = conf
        self.layer = None

    @override
    def forward(
        self, z: Array, u: Array, c: Array, *, key: Array | None = None
    ) -> Array:
        return z


def test_constructor():
    T = 100
    y_size = 10
    z_size = 2
    u_size = 1
    state_noise = 1.0
    mc_size = 10
    seed = 0
    likelihood = "Poisson"
    dropout = 0.5
    width = 16
    depth = 2
    emission_noise = 1.0
    normed_readout = True

    model_conf = OmegaConf.create(
        dict(
            mode="pseudo",
            observation_dim=y_size,
            state_dim=z_size,
            forward="Mock",
            approx="MVN",
            approx_kwargs={"rank": 0},
            mc_size=mc_size,
            seed=seed,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=dropout,
            dyn_conf=OmegaConf.create(
                dict(
                    width=width,
                    depth=depth,
                    linear_input_size=1,
                    dropout=dropout,
                    observation_dim=y_size,
                    state_dim=z_size,
                    input_dim=u_size,
                    context_dim=0,
                    state_noise=state_noise,
                )
            ),
            enc_conf=OmegaConf.create(
                dict(
                    width=width,
                    depth=depth,
                    dropout=dropout,
                    observation_dim=y_size,
                    state_dim=z_size,
                    approx="MVN",
                    approx_kwargs={"rank": 0},
                )
            ),
            obs_conf=OmegaConf.create(
                dict(
                    model="GLM",
                    observation_dim=y_size,
                    state_dim=z_size,
                    emission_noise=emission_noise,
                    norm_readout=normed_readout,
                    dropout=dropout,
                    likelihood=likelihood,
                )
            ),
        )
    )

    model = XFADS(model_conf, jr.key(seed))

    # Verify noise is on the model, not on forward dynamics
    assert model.unconstrained_noise_moment is not None
    assert not hasattr(model.forward, "unconstrained_noise_moment")

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.zip"
        XFADS.save(model, path)
        loaded_model = XFADS.load(path)

        eqx.tree_equal(model, loaded_model)
