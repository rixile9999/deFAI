import numpy as np
import pytest

from defai.kalman import KalmanConfig


@pytest.mark.parametrize("key,value", [
    ("rho", -0.1), ("rho", 1.1), ("rho", True),
    ("process_noise", -1), ("observation_noise", 0),
    ("initial_variance", -1), ("variance_floor", 0),
    ("process_noise", np.nan), ("observation_noise", np.inf),
])
def test_invalid_uncertainty_config(key, value):
    with pytest.raises(ValueError, match=key):
        KalmanConfig(**{key: value})


def test_valid_zero_prior_and_process_noise():
    assert KalmanConfig(process_noise=0, initial_variance=0).rho == 1
