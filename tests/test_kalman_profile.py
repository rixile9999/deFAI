import pytest

from defai.kalman_profile import profile_estimator


def test_profile_reports_actual_sizes_and_separate_fixed_adaptive_results():
    result = profile_estimator(sizes=(16, 100), repeats=3)
    assert len(result["results"]) == 4
    assert {r["mode"] for r in result["results"]} == {"fixed", "adaptive"}
    for row in result["results"]:
        assert row["state_numeric_bytes"] == row["elements"]*8+40
        assert 0 <= row["min_seconds"] <= row["median_seconds"] <= row["p90_seconds"]
        assert row["traced_peak_bytes"] > 0


@pytest.mark.parametrize("kwargs", [{"sizes": ()}, {"sizes": (0,)}, {"sizes": (True,)}, {"repeats": 2}])
def test_profile_rejects_invalid_workloads(kwargs):
    with pytest.raises(ValueError): profile_estimator(**kwargs)
