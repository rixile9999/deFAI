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


@pytest.mark.parametrize("blend", [0., .5, 1.])
def test_failed_late_layer_does_not_advance_any_state(blend):
    from copy import deepcopy
    s = initialized(kalman_blend=blend)
    valid = {"w": np.ones(3), "bias": np.array(.1)}
    s.update_global(1, valid)
    before = deepcopy((s._m, s._v, s._ema, s._layers, s._last_round))
    with pytest.raises(ValueError):
        s.update_global(2, {**valid, "bias": np.array(1e308)})
    assert s._last_round == before[-1]
    for group, saved in zip((s._m, s._v), before[:2]):
        for k in group:
            np.testing.assert_array_equal(group[k], saved[k])
    for k in s._layers:
        np.testing.assert_array_equal(s._layers[k].mean, before[3][k].mean)
        assert s._layers[k].variance == before[3][k].variance
    # The rejected round can be retried.
    assert all(np.isfinite(x).all() for x in s.update_global(2, valid).values())


@pytest.mark.parametrize("change", ["missing", "shape", "nan", "integer"])
def test_bad_schema_or_input_is_rejected_before_state_changes(change):
    s = initialized()
    delta = {"w": np.ones(3), "bias": np.array(.1)}
    if change == "missing": delta.pop("bias")
    if change == "shape": delta["w"] = np.ones(4)
    if change == "nan": delta["w"][0] = np.nan
    if change == "integer": delta["w"] = np.ones(3, dtype=int)
    with pytest.raises(ValueError): s.update_global(1, delta)
    assert s._last_round == 0 and s._m is None and not s._layers


def test_duplicate_rounds_rejected_but_skipped_rounds_allowed():
    s = initialized()
    delta = {"w": np.ones(3), "bias": np.array(.1)}
    s.update_global(3, delta)
    with pytest.raises(ValueError, match="increase"): s.update_global(3, delta)
    s.update_global(5, delta)


def test_trust_radius_is_shared_across_all_layers():
    s = initialized(max_step_norm=.001)
    before = {k: v.copy() for k, v in s._global.items()}
    out = s.update_global(1, {"w": np.ones(3), "bias": np.array(2.)})
    displacement = np.sqrt(sum(np.sum((out[k] - before[k])**2) for k in out))
    assert displacement == pytest.approx(.001, rel=1e-6)
    assert 0 < s._round_metrics["kf-effective-lr"] < s.server_learning_rate


@pytest.mark.parametrize("kwargs", [{"kalman_blend": np.nan}, {"kalman_blend": True}, {"kalman_blend": 1.1},
                                    {"max_step_norm": -1}, {"max_step_norm": np.inf}, {"tau": np.inf}])
def test_invalid_strategy_options(kwargs):
    with pytest.raises(ValueError): initialized(**kwargs)


def test_large_finite_variance_metrics_do_not_overflow_across_layers():
    s = FedAdamKalman(kalman=KalmanConfig(process_noise=0, initial_variance=1e308,
                                        observation_noise=1e308, noise_ceiling=1e308))
    s._global = {str(i): np.zeros(1) for i in range(4)}
    s.update_global(1, {k: np.zeros(1) for k in s._global})
    assert s._round_metrics["kf-variance-mean"] == pytest.approx(5e307)
    assert all(np.isfinite(v) for v in s._round_metrics.values())


def test_unrepresentable_metric_does_not_commit_even_when_output_layers_are_finite():
    s = FedAdamKalman(server_learning_rate=1e308)
    s._global = {"w": np.zeros(16)}
    with pytest.raises(ValueError): s.update_global(1, {"w": np.ones(16)})
    assert s._last_round == 0 and s._m is None and s._round_metrics == {}
