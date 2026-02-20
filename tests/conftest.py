import pytest


@pytest.fixture
def spec():
    """Shared test parameters for common model dimensions."""
    return dict(
        observation_dim=10,
        state_dim=2,
        input_dim=0,
        width=32,
        depth=2,
        approx="DiagMVN",
    )
