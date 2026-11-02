import numpy as np
import pytest

from defai.kalman import KalmanConfig, LayerState, layer_step, stable_l2


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
    high = layer_step(d, None, KalmanConfig(process_noise=1, observation_noise=1e8, noise_ceiling=1e8))
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


@pytest.mark.parametrize("gain", [0., .1, .8, 1.])
def test_forced_constant_gain_matches_ema_and_joseph_covariance(gain):
    cfg = KalmanConfig(gain_min=gain, gain_max=gain, process_noise=.1, observation_noise=.2)
    mean = np.zeros(2)
    state = None
    for i in range(1, 6):
        d = np.array([i, -i], dtype=float)
        prior = (state.variance if state else 0) + .1
        state = layer_step(d, state, cfg)
        mean = (1 - gain) * mean + gain * d
        np.testing.assert_allclose(state.mean, mean)
        assert state.variance == pytest.approx(max((1-gain)**2 * prior + gain**2 * .2, cfg.variance_floor))


@pytest.mark.parametrize("kwargs", [{"gain_min": -.1}, {"gain_max": 1.1}, {"gain_min": .8, "gain_max": .2}])
def test_invalid_gain_bounds(kwargs):
    with pytest.raises(ValueError, match="gain"):
        KalmanConfig(**kwargs)


def test_adaptation_is_next_round_and_bounded():
    cfg = KalmanConfig(process_adaptation=1, observation_adaptation=.5, noise_ceiling=.1)
    fixed = layer_step(np.array([100.]), None, KalmanConfig())
    adapted = layer_step(np.array([100.]), None, cfg)
    assert adapted.gain == fixed.gain
    assert adapted.process_noise == .1
    assert adapted.observation_noise == pytest.approx(.055)
    assert np.isfinite(layer_step(np.array([0.]), adapted, cfg).mean).all()


def test_process_adaptation_responds_faster_after_sustained_direction_change():
    fixed_cfg = KalmanConfig()
    adaptive_cfg = KalmanConfig(process_adaptation=.5)
    fixed = adaptive = None
    for _ in range(20):
        fixed = layer_step(np.array([0.]), fixed, fixed_cfg)
        adaptive = layer_step(np.array([0.]), adaptive, adaptive_cfg)
    for _ in range(4):
        fixed = layer_step(np.array([1.]), fixed, fixed_cfg)
        adaptive = layer_step(np.array([1.]), adaptive, adaptive_cfg)
    assert abs(1 - adaptive.mean[0]) < abs(1 - fixed.mean[0])


@pytest.mark.parametrize("kwargs", [{"process_adaptation": -1}, {"observation_adaptation": 1.1},
                                    {"noise_ceiling": 1e-15}, {"noise_ceiling": .001}])
def test_invalid_adaptation(kwargs):
    with pytest.raises(ValueError):
        KalmanConfig(**kwargs)


@pytest.mark.parametrize("variance,q,r", [(-1, .01, .01), (np.nan, .01, .01), (0, -1, .01), (0, .01, 0)])
def test_invalid_prior_uncertainty_is_rejected(variance, q, r):
    prior = LayerState(np.zeros(2), variance, q, r)
    with pytest.raises(ValueError, match="previous"):
        layer_step(np.ones(2), prior, KalmanConfig())


def test_large_finite_covariances_do_not_overflow_the_gain_denominator():
    cfg = KalmanConfig(process_noise=0, initial_variance=1e308,
                       observation_noise=1e308, noise_ceiling=1e308)
    result = layer_step(np.array([1.]), None, cfg)
    assert result.gain == pytest.approx(.5)
    assert result.mean[0] == pytest.approx(.5)
    assert result.variance == pytest.approx(5e307)


def test_representable_innovation_mean_square_does_not_overflow_its_sum():
    result = layer_step(np.full(2, 1e154), None, KalmanConfig())
    assert result.innovation_mse == pytest.approx(1e308)


def test_sparse_large_residual_can_have_a_representable_mean_square():
    observation = np.zeros(1000)
    observation[0] = 1e155
    result = layer_step(observation, None, KalmanConfig())
    assert result.innovation_mse == pytest.approx(1e307)


def test_stable_norm_handles_large_values_and_multiple_layers():
    assert stable_l2([np.array([1e200, 1e200]), np.array(1e200)]) == pytest.approx(np.sqrt(3)*1e200)
    assert stable_l2([np.array([0.]), np.zeros(3)]) == 0
    assert stable_l2([np.array([3., 4.]), np.array(12.)]) == 13


@pytest.mark.parametrize("beta", [0., .5, .9, .99, 1-1e-15])
def test_ema_calibration_matches_the_existing_first_moment(beta):
    cfg = KalmanConfig.from_ema(beta)
    state, mean = None, np.zeros(3)
    for d in np.random.default_rng(8).normal(size=(30, 3)):
        state = layer_step(d, state, cfg)
        mean = beta*mean+(1-beta)*d
        np.testing.assert_allclose(state.mean, mean, atol=1e-14, rtol=1e-12)


def test_adaptive_calibration_gain_cannot_fall_below_original_ema_weight():
    cfg = KalmanConfig.from_ema(.9, process_adaptation=.2, gain_floor=True)
    state = None
    for _ in range(100):
        state = layer_step(np.zeros(3), state, cfg)
        assert state.gain >= 1-.9
