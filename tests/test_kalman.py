import numpy as np
import pytest

from defai.kalman import KalmanConfig, LayerState, layer_step


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


def test_hand_calculated_prediction_and_correction():
    cfg = KalmanConfig(rho=.5, process_noise=.03, observation_noise=.06)
    previous = LayerState(np.array([.2, -.2]), .04, .03, .06)
    d = np.array([.5, .3])
    result = layer_step(d, previous, cfg)
    # predicted mean=[.1,-.1], P-=.04, K=.4, P+=.024
    np.testing.assert_allclose(result.mean, [.26, .06])
    assert result.gain == pytest.approx(.4)
    assert result.variance == pytest.approx(.024)
    np.testing.assert_array_equal(previous.mean, [.2, -.2])
    np.testing.assert_array_equal(d, [.5, .3])


def test_small_observation_noise_trusts_new_direction():
    d = np.array([2., -3.])
    low = layer_step(d, None, KalmanConfig(process_noise=1, observation_noise=1e-8))
    high = layer_step(d, None, KalmanConfig(process_noise=1, observation_noise=1e8))
    assert low.gain > .999 and high.gain < 1e-7


def test_zero_prior_floor_and_scalar_observations():
    state = layer_step(np.array(3.), None, KalmanConfig(process_noise=0))
    assert state.mean.shape == () and state.mean == 0
    assert state.variance == 1e-12


@pytest.mark.parametrize("observation", [np.array([np.nan]), np.array([np.inf]),
                                         np.array([], dtype=float), np.array([1j]), np.array(["1"])])
def test_invalid_observations(observation):
    with pytest.raises(ValueError):
        layer_step(observation, None, KalmanConfig())


def test_shape_change_and_overflow_are_rejected():
    state = layer_step(np.ones(2), None, KalmanConfig())
    with pytest.raises(ValueError, match="shape"):
        layer_step(np.ones(3), state, KalmanConfig())
    with pytest.raises(ValueError, match="arithmetic"):
        layer_step(np.array([1e308]), None, KalmanConfig())
