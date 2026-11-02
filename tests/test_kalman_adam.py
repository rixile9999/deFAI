import numpy as np
import pytest

from defai.kalman import KalmanConfig
from defai.strategies.fedopt import FedAdam
from defai.strategies.kalman_adam import FedAdamKalman


def initialized(cls=FedAdamKalman, **kwargs):
    strategy = cls(server_learning_rate=.03, **kwargs)
    strategy._global = {"w": np.zeros(3, np.float32), "bias": np.zeros((), np.float64)}
    return strategy


def test_zero_blend_is_bitwise_identical_to_fedadam_over_multiple_rounds():
    plain, extension = initialized(FedAdam), initialized(kalman_blend=0)
    rng = np.random.default_rng(3)
    for rnd in range(1, 21):
        delta = {k: rng.normal(size=v.shape) for k, v in plain._global.items()}
        a = plain.update_global(rnd, delta)
        b = extension.update_global(rnd, delta)
        for key in delta:
            np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(plain._m[key], extension._m[key])
            np.testing.assert_array_equal(plain._v[key], extension._v[key])
        plain._global, extension._global = a, b
    assert extension._layers == {}


def test_first_moment_is_kalman_not_double_smoothed_and_second_moment_is_raw():
    strategy = initialized(kalman=KalmanConfig(process_noise=.02, observation_noise=.03), beta_1=.999)
    delta = {"w": np.array([1., -2., 3.]), "bias": np.array(.5)}
    strategy.update_global(1, delta)
    for k, d in delta.items():
        np.testing.assert_allclose(strategy._m[k], .4 * d)
        np.testing.assert_allclose(strategy._v[k], .99 * 1e-6 + .01 * d**2)
    assert strategy._ema is None


def test_blend_combines_independent_ema_and_filter_means():
    strategy = initialized(kalman_blend=.25, kalman=KalmanConfig(process_noise=.02, observation_noise=.03))
    strategy.update_global(1, {"w": np.ones(3), "bias": np.array(1.)})
    np.testing.assert_allclose(strategy._m["w"], .75 * .1 + .25 * .4)


def test_global_dtype_is_preserved_and_inputs_are_not_mutated():
    s = initialized()
    delta = {k: np.ones_like(v, dtype=np.float64) for k, v in s._global.items()}
    output = s.update_global(1, delta)
    for k in delta:
        assert output[k].dtype == s._global[k].dtype
        np.testing.assert_array_equal(delta[k], 1)
        np.testing.assert_array_equal(s._global[k], 0)
