import pytest

from jaxfads.base import Approx
from jaxfads.distributions import MVN


@pytest.fixture
def spec():
    """Shared test parameters for common model dimensions."""
    return dict(
        observation_dim=10,
        state_dim=2,
        input_dim=0,
        width=32,
        depth=2,
        approx="MVN",
        approx_kwargs={"rank": 0},
    )


@pytest.fixture
def diag():
    """Diagonal MVN instance (rank=0)."""
    return MVN(rank=0)


@pytest.fixture
def full_mvn():
    """Full-covariance MVN instance (rank=-1)."""
    return MVN(rank=-1)


def make_approx(approx_name: str = "MVN", **kwargs) -> Approx:
    """Instantiate an Approx from name and kwargs."""
    cls = Approx.get_subclass(approx_name)
    return cls(**kwargs)


def make_noise_moment(approx: Approx, state_dim: int, cov: float = 1.0):
    """Create a constrained noise moment array for test use."""
    return approx.constrain_moment(approx.init_noise(cov, state_dim))
