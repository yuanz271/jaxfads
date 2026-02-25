import pytest

from jaxfads.base import Approx
from jaxfads.distributions import MVN

_STATE_DIM = 2


@pytest.fixture
def spec():
    """Shared test parameters for common model dimensions."""
    return dict(
        observation_dim=10,
        state_dim=_STATE_DIM,
        input_dim=0,
        width=32,
        depth=2,
        approx="MVN",
        approx_kwargs={"rank": 0},
    )


@pytest.fixture
def diag():
    """Diagonal MVN instance (rank=0)."""
    return MVN(dim=_STATE_DIM, rank=0)


def make_approx(dim: int, approx_name: str = "MVN", **kwargs) -> Approx:
    """Instantiate an Approx from name, dim, and kwargs."""
    cls = Approx.get_subclass(approx_name)
    return cls(dim=dim, **kwargs)


def make_noise(approx: Approx, state_dim: int, cov: float = 1.0):
    """Create noise transition params (flat) for test use."""
    return approx.canon_to_mean(approx.free_to_canon(approx.param_from_conf(scale=cov)))
