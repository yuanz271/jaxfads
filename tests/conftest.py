import pytest


@pytest.fixture
def spec():
    spec = dict(
        observation_dim=10,
        state_dim=2,
        input_dim=0,
        width=32,
        depth=2,
        emission_noise=1.0,
        state_noise=1.0,
        dynamics="Nonlinear",
        approx="DiagMVN",
        norm_readout=False,
        mc_size=1,
        random_state=0,
        min_iter=0,
        max_em_iter=1,
        max_inner_iter=1,
        batch_size=1,
        static_params="",
        covariate_dim=0,
    )
    return spec
