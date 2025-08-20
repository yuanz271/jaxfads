from typing import override
from pathlib import Path
from tempfile import TemporaryDirectory

from jax import Array, random as jrnd
from omegaconf import OmegaConf
import equinox as eqx
from jaxfads.dynamics import Dynamics, Noise
from jaxfads.smoother import XFADS


class Mock(Dynamics):
    noise: Noise | None
    layer: eqx.Module | None

    def __init__(self, conf, key):
        self.conf = conf
        self.noise = None  # type: ignore
        self.layer = None

    @override
    def forward(self, z, u, c, *, key=None) -> Array:
        return z

    def loss(self):
        return 0.0


def test_constructor():
    T = 100
    y_size = 10
    z_size = 2
    u_size = 1
    state_noise = 1.0
    mc_size = 10
    seed = 0
    observation = "Poisson"
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
            approx="DiagMVN",
            mc_size=mc_size,
            seed=seed,
            observation=observation,
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
                    approx="DiagMVN",
                )
            ),
            obs_conf=OmegaConf.create(
                dict(
                    observation_dim=y_size,
                    state_dim=z_size,
                    emission_noise=emission_noise,
                    norm_readout=normed_readout,
                    dropout=dropout,
                )
            ),
        )
    )

    model = XFADS(model_conf, jrnd.key(seed))

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.zip"
        XFADS.save(model, path)
        loaded_model = XFADS.load(path)

        eqx.tree_equal(model, loaded_model)
