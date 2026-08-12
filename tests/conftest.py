import pytest
import equinox as eqx
from jax import Array

from jaxfads.base import Approx, Dynamics
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
        approx_kwargs={},
    )


@pytest.fixture
def diag():
    """MVN instance (full rank)."""
    return MVN(dim=_STATE_DIM, rank=_STATE_DIM)


def make_approx(dim: int, approx_name: str = "MVN", **kwargs) -> Approx:
    """Instantiate an Approx from name, dim, and kwargs."""
    cls = Approx.get_subclass(approx_name)
    kwargs.setdefault("rank", dim)
    return cls(dim=dim, **kwargs)


class MockDynamics(Dynamics):
    """Mock dynamics — identity transition. Shared across test modules."""

    layer: eqx.Module | None

    def __init__(self, conf, key: Array = None):
        self.conf = conf
        self.layer = None

    def eval(self, z: Array, u: Array, c: Array, *, key: Array | None = None) -> Array:
        return z
