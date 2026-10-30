import math

import numpy as np
import pytest

from defai.aggregation import l2_norm
from defai.privacy.accountant import RdpAccountant
from defai.privacy.dp import (
    KEY_ADAPTIVE,
    KEY_CLIP_NORM,
    AdaptiveClipping,
    FixedClipping,
    adaptive_noise_multiplier,
    clip_to_norm,
)


def update_with_norm(norm, dim=4):
    return {"w": np.full(dim, norm / math.sqrt(dim))}


def test_clip_scales_down_and_keeps_direction():
    u = {"a": np.array([3.0, 0.0]), "b": np.array([0.0, 4.0])}  # norm 5
    clipped, was_clipped = clip_to_norm(u, 1.0)
    assert was_clipped and l2_norm(clipped) == pytest.approx(1.0)
    np.testing.assert_allclose(clipped["a"], [0.6, 0.0])
    np.testing.assert_allclose(clipped["b"], [0.0, 0.8])


def test_clip_leaves_small_updates_untouched():
    u = update_with_norm(0.5)
    clipped, was_clipped = clip_to_norm(u, 1.0)
    assert not was_clipped and clipped is u


def test_fixed_clipping_without_noise_is_fixed_denominator_mean_of_clipped():
    dp = FixedClipping(0.0, 1.0)
    # m=4 sampled but only 2 replied: denominator stays 4 so sensitivity is C/4
    out, metrics = dp.aggregate([update_with_norm(10.0), update_with_norm(0.5)], num_sampled=4)
    expected = (update_with_norm(1.0)["w"] + update_with_norm(0.5)["w"]) / 4
    np.testing.assert_allclose(out["w"], expected)
    assert metrics["dp-noise-std"] == 0.0
    assert "dp-clipped-fraction" not in metrics  # the exact clipped count is not covered by the accounting


def test_noise_std_is_z_times_c_over_m():
    z, c, m = 1.5, 2.0, 10
    dp = FixedClipping(z, c, seed=1)
    out, metrics = dp.aggregate([{"w": np.zeros(400_000)}] * m, num_sampled=m)
    assert metrics["dp-noise-std"] == pytest.approx(z * c / m)
    assert out["w"].std() == pytest.approx(z * c / m, rel=0.01)
    assert abs(out["w"].mean()) < 0.01


def test_epsilon_is_tracked_per_round_and_matches_accountant():
    dp = FixedClipping(1.2, 1.0, delta=1e-5, seed=0)
    ref = RdpAccountant()
    eps = []
    for _ in range(3):
        _, metrics = dp.aggregate([update_with_norm(1.0)] * 5, 5)
        ref.compose(1.2)
        eps.append(metrics["dp-epsilon"])
        assert metrics["dp-epsilon"] == pytest.approx(ref.get_epsilon(1e-5))
    assert eps == sorted(eps) and eps[0] > 0


def test_more_updates_than_sampled_is_rejected():
    with pytest.raises(ValueError):
        FixedClipping(1.0, 1.0).aggregate([update_with_norm(1.0)] * 3, num_sampled=2)


def test_budget_exhaustion_is_detected_before_the_round_that_would_exceed():
    dp = FixedClipping(1.0, 1.0, max_epsilon=3.0, seed=0)
    rounds = 0
    while not dp.budget_exhausted():
        dp.aggregate([update_with_norm(1.0)] * 4, 4)
        rounds += 1
        assert rounds < 100
    assert dp.epsilon <= 3.0  # never overshoots the budget
    assert dp.accountant.would_exceed(1.0, 1e-5, 3.0)


def test_client_config_only_for_client_side():
    assert FixedClipping(1.0, 2.0, side="server").client_config() == {}
    assert FixedClipping(1.0, 2.0, side="client").client_config() == {KEY_CLIP_NORM: 2.0, KEY_ADAPTIVE: False}
    assert AdaptiveClipping(1.0, 2.0, side="client").client_config()[KEY_ADAPTIVE] is True


@pytest.mark.parametrize(
    "kwargs",
    [dict(noise_multiplier=-1, clip_norm=1), dict(noise_multiplier=1, clip_norm=0), dict(noise_multiplier=1, clip_norm=1, side="x")],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        FixedClipping(**kwargs)


# --- adaptive clipping --------------------------------------------------------
def test_adaptive_noise_multiplier_composes_back_to_target():
    for z, sigma_b in [(1.0, 5.0), (0.5, 2.0), (3.0, 10.0)]:
        z_u = adaptive_noise_multiplier(z, sigma_b)
        assert (z_u**-2 + (2 * sigma_b) ** -2) ** -0.5 == pytest.approx(z)
        assert z_u > z  # the bit query consumes part of the budget
    assert adaptive_noise_multiplier(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        adaptive_noise_multiplier(1.0, 0.5)


def run_adaptive(target, start_clip, rounds=500, m=60, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    dp = AdaptiveClipping(0.0, start_clip, target_quantile=target, clip_lr=0.2, seed=seed, **kwargs)
    history = []
    for _ in range(rounds):
        norms = rng.lognormal(0.0, 0.5, m)  # true q-quantile = exp(0.5 * Phi^-1(q))
        dp.aggregate([update_with_norm(n) for n in norms], m)
        history.append(dp.clip_norm)
    return float(np.mean(history[-100:]))


@pytest.mark.parametrize("target,true_quantile", [(0.5, 1.0), (0.8, 1.5230), (0.2, 0.6566)])
def test_adaptive_clip_converges_to_target_quantile_of_update_norms(target, true_quantile):
    for start in (0.05, 20.0):  # far too small and far too large
        assert run_adaptive(target, start) == pytest.approx(true_quantile, rel=0.12), (target, start)


def test_adaptive_noise_uses_the_clip_norm_of_that_round():
    dp = AdaptiveClipping(1.0, 2.0, clipped_count_stddev=10.0, seed=0)
    _, metrics = dp.aggregate([update_with_norm(5.0)] * 20, 20)
    z_u = adaptive_noise_multiplier(1.0, 10.0)
    assert metrics["dp-clip-norm"] == 2.0
    assert metrics["dp-noise-std"] == pytest.approx(z_u * 2.0 / 20)
    assert dp.clip_norm != 2.0  # updated only after the noise was calibrated


def test_adaptive_default_bit_noise_is_m_over_20():
    dp = AdaptiveClipping(1.0, 1.0, seed=0)
    _, metrics = dp.aggregate([update_with_norm(1.0)] * 40, 40)
    assert metrics["dp-noise-std"] == pytest.approx(adaptive_noise_multiplier(1.0, 2.0) / 40)


def test_client_side_adaptive_uses_reported_bits_and_requires_them():
    dp = AdaptiveClipping(0.0, 1.0, side="client", target_quantile=0.5, clip_lr=0.2)
    # every client reports "unclipped" -> fraction 1.0 > 0.5 -> clip norm must shrink
    dp.aggregate([update_with_norm(0.3)] * 4, 4, unclipped_bits=[1, 1, 1, 1])
    assert dp.clip_norm == pytest.approx(math.exp(-0.2 * 0.5))
    with pytest.raises(ValueError):
        dp.aggregate([update_with_norm(0.3)] * 4, 4, unclipped_bits=None)
    with pytest.raises(ValueError):
        dp.aggregate([update_with_norm(0.3)] * 4, 4, unclipped_bits=[1, None, 1, 1])


def test_default_bit_noise_never_drops_below_the_noise_multiplier():
    """m / 20 is infeasible for small federations (needs z < m / 10); the default must still work."""
    dp = AdaptiveClipping(1.0, 1.0, seed=0)  # m = 5 -> m / 20 = 0.25 < z
    out, metrics = dp.aggregate([update_with_norm(1.0)] * 5, 5)
    z_update = adaptive_noise_multiplier(1.0, 1.0)  # bit std falls back to z
    assert metrics["dp-noise-std"] == pytest.approx(z_update * 1.0 / 5)
    assert z_update == pytest.approx((1 - 0.25) ** -0.5)
    assert (z_update**-2 + 2.0**-2) ** -0.5 == pytest.approx(1.0)  # still composes to z overall


def test_begin_round_rejects_an_infeasible_explicit_bit_noise_with_an_actionable_message():
    dp = AdaptiveClipping(1.0, 1.0, clipped_count_stddev=0.25)
    with pytest.raises(ValueError, match="dp-clipped-count-stddev"):
        dp.begin_round(5)
    AdaptiveClipping(1.0, 1.0, clipped_count_stddev=0.6).begin_round(5)  # 2 * 0.6 > 1: fine


@pytest.mark.parametrize("bad", [[1, 1, 1, 2], [1, 1, 1, 10**6], [1, 1, 1, -1], [1, 1, 1, None]])
def test_client_reported_bits_must_be_zero_or_one_and_are_checked_before_accounting(bad):
    dp = AdaptiveClipping(1.0, 1.0, side="client", seed=0)
    with pytest.raises(ValueError):
        dp.aggregate([update_with_norm(0.5)] * 4, 4, unclipped_bits=bad)
    assert dp.accountant.num_releases == 0 and dp.clip_norm == 1.0  # nothing was spent or changed


def test_only_accounted_quantities_are_published():
    fixed = FixedClipping(1.0, 1.0, seed=0)
    _, metrics = fixed.aggregate([update_with_norm(5.0)] * 10, 10)
    assert set(metrics) == {"dp-clip-norm", "dp-noise-std", "dp-epsilon"}

    adaptive = AdaptiveClipping(1.0, 1.0, seed=0)
    _, metrics = adaptive.aggregate([update_with_norm(5.0)] * 10 + [update_with_norm(0.1)] * 30, 40)
    assert set(metrics) == {"dp-clip-norm", "dp-noise-std", "dp-epsilon", "dp-noisy-unclipped-fraction"}
    # the published fraction is the noisy one that drives the clip-norm update (exactly 30/40 would be 0.75)
    assert metrics["dp-noisy-unclipped-fraction"] != 0.75
    assert adaptive.clip_norm == pytest.approx(
        math.exp(-0.2 * (metrics["dp-noisy-unclipped-fraction"] - 0.5))
    )
