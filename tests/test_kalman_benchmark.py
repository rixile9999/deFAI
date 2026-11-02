from copy import deepcopy

import numpy as np
import pytest

from defai.kalman_benchmark import METHODS, make_clients, run_case, summarize


def test_synthetic_train_test_are_independent_and_reproducible():
    a, b = make_clients(.5, 2), make_clients(.5, 2)
    for (train, test), other in zip(a, b):
        np.testing.assert_array_equal(train[0], other[0][0])
        assert train[0].shape == (48, 7) and test[0].shape == (96, 7)
        assert not np.shares_memory(train[0], test[0])


def test_all_methods_have_paired_schedules_and_identical_payload_budgets():
    rows = [run_case(alpha=.5, local_steps=1, participation=.5, drift=True, seed=3, method=m, rounds=4) for m in METHODS]
    assert len({r["schedule_sha256"] for r in rows}) == 1
    assert len({sum(h["training_payload_bytes"] for h in r["history"]) for r in rows}) == 1
    assert all(np.isfinite(h["loss"]) for r in rows for h in r["history"])
    summary = summarize(rows)
    assert all(v["runs"] == 1 for v in summary.values())
    assert summary["ema"]["paired_more_rounds"] == 0
    bad = deepcopy(rows)
    bad[1]["schedule_sha256"] = "wrong"
    with pytest.raises(ValueError, match="paired"): summarize(bad)


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"local_steps": 0}, {"participation": 0},
                                    {"rounds": 1}, {"seed": -1}, {"target": 2}, {"method": "x"}])
def test_invalid_benchmark_conditions(kwargs):
    cfg = dict(alpha=.5, local_steps=1, participation=.5, drift=False, seed=3, method="ema", rounds=4)
    cfg.update(kwargs)
    with pytest.raises(ValueError): run_case(**cfg)
