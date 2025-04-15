from pathlib import Path
from tempfile import TemporaryDirectory
from omegaconf import OmegaConf
import equinox as eqx
from xfads.dynamics import AbstractDynamics
from xfads.smoother import XFADS


class Mock(AbstractDynamics):
    noise: eqx.Module
    layer: eqx.Module

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key,
        dropout: float | None = None,
        **kwargs,
    ):
        self.noise = None
        self.layer = None

    def __call__(self, z, u, *, key=None):
        pass

    def loss(self):
        return 0.


def test_constructor():
    T = 100
    y_size = 10
    z_size = 2
    u_size = 1
    state_noise = 1.
    mc_size = 10
    seed = 0
    observation = "Poisson"
    dropout = 0.5
    width = 16
    depth = 2
    emission_noise = 1.
    normed_readout = True

    model_conf = OmegaConf.create(
        dict(
            mode="pseudo",
            observation_dim=y_size,
            state_dim=z_size,
            input_dim=u_size,
            forward="Mock",
            backward=None,
            approx="DiagMVN",
            state_noise=state_noise,
            mc_size=mc_size,
            seed=seed,
            observation=observation,
            n_steps=T,
            fb_penalty=0,
            noise_penalty=0,
            dropout=dropout,
            dyn_kwargs=dict(
                width=width,
                depth=depth,
                linear_input_size=1,
                dropout=dropout,
            ),
            enc_kwargs=dict(
                width=width, depth=depth, dropout=dropout
            ),
            obs_kwargs=dict(
                emission_noise=emission_noise,
                norm_readout=normed_readout,
                dropout=dropout,
            ),
        )
    )
    
    model = XFADS(model_conf)

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.zip"
        XFADS.save(model, path)
        loaded_model = XFADS.load(path)

        eqx.tree_equal(model, loaded_model)
