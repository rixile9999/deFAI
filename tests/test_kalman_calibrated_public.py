import numpy as np

from defai.kalman_benchmark import METHODS
from defai.kalman_data import prepare_public_data
from defai.kalman_public import run_public_case


def test_calibrated_fixed_public_model_matches_ema_validation_and_test_scores():
    data = prepare_public_data("digits", seed=100)
    base = run_public_case(data, method="ema", rounds=6, local_steps=1, calibrated=True)
    fixed = run_public_case(data, method="kalman", rounds=6, local_steps=1, calibrated=True)
    for a,b in zip(base["history"], fixed["history"]):
        np.testing.assert_allclose([a["accuracy"], a["loss"]], [b["accuracy"], b["loss"]], atol=1e-12)
    assert base["heldout_test"]["accuracy"] == fixed["heldout_test"]["accuracy"]
    assert fixed["kalman_config"]["initial_variance"] > 0


def test_calibrated_methods_keep_identical_data_splits_and_schedules():
    data = prepare_public_data("breast-cancer", seed=101)
    rows = [run_public_case(data, method=m, rounds=3, local_steps=1, calibrated=True) for m in METHODS]
    for field in ("dataset_sha256", "split_sha256", "schedule_sha256"):
        assert len({r[field] for r in rows}) == 1
    assert rows[-1]["kalman_config"]["gain_min"] == 1-.9
