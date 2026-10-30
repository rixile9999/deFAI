from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.policy_stats import AnalysisConfig, daily_counts, district_treatment
from core.synthetic_regions import RegionGeneration, export_regions, generate_regions
from run_policy_validation import load_records, reference_analysis
from run_tier2 import DistrictClient


def test_explicit_randomized_assignment_and_reproducibility():
    settings = RegionGeneration(n_provinces=8, n_treated=4, seed=72)
    records, truth = generate_regions(settings)
    second, same_truth = generate_regions(settings)
    assert len(records) == 24
    assert truth == same_truth
    assert len({r["province"] for r in records if r["treated"]}) == 4
    for a, b in zip(records, second):
        np.testing.assert_array_equal(a["counts"], b["counts"])
        np.testing.assert_array_equal(a["diagnosed_counts"], b["diagnosed_counts"])
        assert np.all(a["counts"] <= a["policy_only_counts"])
        assert np.all(a["policy_only_counts"] <= a["counterfactual_counts"])
        assert np.all(a["counts"] <= a["true_counts"])
        assert a["diagnosed_counts"].sum() == a["counts"].sum()


def test_null_and_separate_reporting_confounding():
    settings = RegionGeneration(policy_ratio=1, n_provinces=8, n_treated=4)
    _, truth = generate_regions(settings)
    assert truth["oracle_policy_effect_log1p"] == 0
    records, reported_truth = generate_regions(replace(settings, reporting_ratio=.5))
    assert reported_truth["oracle_policy_effect_log1p"] == 0
    assert sum(r["counts"].sum() for r in records) < sum(r["policy_only_counts"].sum() for r in records)


def test_panel_roundtrip_matches_the_tier2_client(tmp_path):
    config = AnalysisConfig()
    settings = RegionGeneration(n_provinces=8, n_treated=4, unequal_districts=True)
    records, truth = generate_regions(settings, config)
    folder = tmp_path / "regions"
    manifest = export_regions(folder, settings, config, records, truth)
    loaded = load_records(folder, config)
    expected = {r["district"]: r for r in records}
    for record in loaded:
        original = expected[record["district"]]
        np.testing.assert_array_equal(record["counts"], original["counts"])
        client = DistrictClient(record["path"], use_smpc=False)
        assert client.treated == original["treated"]
        np.testing.assert_array_equal(client.counts, original["counts"])
        diagnosis = daily_counts(pd.read_csv(record["path"]), replace(config, date_column="진단일"))
        np.testing.assert_array_equal(diagnosis, original["diagnosed_counts"][:len(config.calendar)])
    assert len(manifest["regions"]) == 8
    with pytest.raises(FileExistsError):
        export_regions(folder, settings, config, records, truth)
    assert np.isfinite(reference_analysis(loaded, config)["effect"])
    with pytest.raises(ValueError, match="disagrees"):
        DistrictClient(loaded[0]["path"], use_smpc=False, treated=1-loaded[0]["treated"])


@pytest.mark.parametrize("problem", ["missing_day", "duplicate_date", "negative", "fraction", "nan", "unknown_format"])
def test_bad_panel_fails_instead_of_recounting_rows(problem):
    config = AnalysisConfig("2024-01-01", "2024-01-03", "2024-01-05")
    frame = pd.DataFrame({"date": config.calendar, "record_format": "daily_counts_v1", "cases": np.ones(5)})
    if problem == "missing_day":
        frame = frame.iloc[1:]
    elif problem == "duplicate_date":
        frame.loc[0, "date"] = frame.loc[1, "date"]
    elif problem == "unknown_format":
        frame.loc[0, "record_format"] = "unknown"
    else:
        frame.loc[0, "cases"] = {"negative": -1, "fraction": .5, "nan": np.nan}[problem]
    with pytest.raises(ValueError):
        daily_counts(frame, config)


def test_treatment_not_guessed_from_region_name():
    assert district_treatment(pd.DataFrame({"treated": [1, 1]}), "Region_007") == 1
    with pytest.raises(ValueError):
        district_treatment(pd.DataFrame({"treated": [0, 1]}), "Region_007")
