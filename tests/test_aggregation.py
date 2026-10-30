import numpy as np
import pytest

from defai.aggregation import (
    coordinate_median,
    krum_scores,
    multi_krum_select,
    pairwise_sq_distances,
    trimmed_mean,
    weighted_mean,
)


def U(*values, dtype=np.float32):
    return {"w": np.array(values, dtype=dtype)}


def test_weighted_mean_matches_manual_and_keeps_dtype():
    out = weighted_mean([U(1, 2), U(3, 6)], [1, 3])
    np.testing.assert_allclose(out["w"], [2.5, 5.0])
    assert out["w"].dtype == np.float32


def test_weighted_mean_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_mean([U(1), U(2)], [0, 0])
    with pytest.raises(ValueError):
        weighted_mean([U(1), U(2)], [1])


def test_median_ignores_outliers():
    updates = [U(1, 1), U(2, 2), U(3, 3), U(1000, -1000), U(2, 2)]
    np.testing.assert_allclose(coordinate_median(updates)["w"], [2, 2])


def test_trimmed_mean_drops_extremes_per_coordinate():
    updates = [U(0, 10), U(1, 11), U(2, 12), U(3, 13), U(100, -100)]
    # n=5, ratio=0.2 -> drop 1 from each side per coordinate
    out = trimmed_mean(updates, 0.2)["w"]
    np.testing.assert_allclose(out, [2.0, 11.0])


def test_trimmed_mean_zero_ratio_is_mean_and_bad_ratio_rejected():
    updates = [U(0), U(2), U(10)]
    np.testing.assert_allclose(trimmed_mean(updates, 0.0)["w"], [4.0])
    with pytest.raises(ValueError):
        trimmed_mean(updates, 0.5)


def test_pairwise_distances_symmetric_zero_diagonal():
    d = pairwise_sq_distances([U(0, 0), U(3, 4), U(6, 8)])
    np.testing.assert_allclose(d, [[0, 25, 100], [25, 0, 25], [100, 25, 0]])


def test_krum_scores_hand_computed():
    # scalars 0, 1, 10 with f=0: n-f-2 = 1 nearest neighbour each
    d = pairwise_sq_distances([U(0), U(1), U(10)])
    np.testing.assert_allclose(krum_scores(d, 0), [1, 1, 81])


def _honest_and_byzantine(n_honest=7, n_byz=2, dim=50, seed=0):
    rng = np.random.default_rng(seed)
    honest = [{"w": rng.normal(0, 0.1, dim)} for _ in range(n_honest)]
    byz = [{"w": rng.normal(50, 1, dim)} for _ in range(n_byz)]
    return honest + byz


def test_krum_and_multikrum_pick_honest_clients():
    updates = _honest_and_byzantine()  # n=9 >= 2f+3 with f=2
    assert multi_krum_select(updates, num_byzantine=2, num_select=1)[0] < 7
    chosen = multi_krum_select(updates, num_byzantine=2, num_select=5)
    assert len(chosen) == 5 and all(i < 7 for i in chosen)


def test_krum_single_client_and_oversized_select():
    assert multi_krum_select([U(1)], 0, 3) == [0]
    assert sorted(multi_krum_select([U(0), U(1), U(2)], 0, 10)) == [0, 1, 2]


# ------------------------------------------------------------------------------ Bulyan
from defai.aggregation import bulyan_select, closest_to_median_mean  # noqa: E402


def test_closest_to_median_mean_hand_computed():
    updates = [U(0), U(1), U(2), U(3), U(100)]  # median 2
    np.testing.assert_allclose(closest_to_median_mean(updates, 3)["w"], [2.0])  # {2, 1, 3}
    np.testing.assert_allclose(closest_to_median_mean(updates, 1)["w"], [2.0])
    np.testing.assert_allclose(closest_to_median_mean(updates, 5)["w"], [106 / 5])  # all values
    with pytest.raises(ValueError):
        closest_to_median_mean(updates, 0)
    with pytest.raises(ValueError):
        closest_to_median_mean(updates, 6)


def test_closest_to_median_mean_acts_per_coordinate():
    updates = [{"w": np.array([0.0, 10.0])}, {"w": np.array([1.0, 11.0])}, {"w": np.array([2.0, 12.0])}, {"w": np.array([50.0, -50.0])}]
    # coordinate medians are 1.5 and 10.5; per coordinate the two closest values are averaged:
    # {1, 2} -> 1.5 and {10, 11} -> 10.5
    np.testing.assert_allclose(closest_to_median_mean(updates, 2)["w"], [1.5, 10.5])


def test_bulyan_select_is_iterated_krum_and_drops_the_outliers():
    updates = _honest_and_byzantine(n_honest=9, n_byz=2)  # n=11 = 4f+3 with f=2
    d = pairwise_sq_distances(updates)
    chosen = bulyan_select(d, num_byzantine=2, num_select=7)  # theta = n - 2f
    assert len(chosen) == len(set(chosen)) == 7 and all(i < 9 for i in chosen)
    # the first pick is plain Krum
    assert chosen[0] == multi_krum_select(updates, 2, 1)[0]


def test_bulyan_select_handles_exhaustion():
    d = pairwise_sq_distances([U(0), U(1), U(2)])
    assert sorted(bulyan_select(d, 0, 10)) == [0, 1, 2]


def test_integer_arrays_are_rounded_not_truncated():
    from defai.aggregation import cast_like

    np.testing.assert_array_equal(cast_like(np.array([257.9999999, 258.4, -0.6]), np.int64), [258, 258, -1])
    assert cast_like(np.array([1.5]), np.float32).dtype == np.float32
    ints = [{"n": np.array([258], dtype=np.int64)}] * 3
    out = weighted_mean(ints, [0.1, 0.2, 0.7])
    assert out["n"].dtype == np.int64 and out["n"][0] == 258
