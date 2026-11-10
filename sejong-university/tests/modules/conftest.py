import pytest

SMALL = {"dataset": "synthetic", "max-samples": 1200, "local-epochs": 1.0}


@pytest.fixture
def small():
    return dict(SMALL)
