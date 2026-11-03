from dataclasses import replace

import pytest

from defai.kalman_benchmark import METHODS
from defai.kalman_data import prepare_public_data
from defai.kalman_public import run_public_case, summarize_public


def test_public_methods_are_paired_and_test_is_evaluated_only_after_training(monkeypatch):
    import defai.kalman_public as module
    data = prepare_public_data("breast-cancer", seed=3, alpha=.1)
    original = module.evaluate
    evaluations = []
    def track(*args):
        evaluations.append(args[-1])
        return original(*args)
    monkeypatch.setattr(module, "evaluate", track)
    rows = [run_public_case(data, method=m, rounds=3, local_steps=1) for m in METHODS]
    assert evaluations == ["validation"]*3+["test"]+["validation"]*3+["test"]+["validation"]*3+["test"]
    assert len({r["schedule_sha256"] for r in rows}) == 1
    assert all(r["test_evaluations"] == 1 for r in rows)
    summary = summarize_public(rows[::-1])
    assert len(summary["condition_strata"]) == 3
    assert summary["datasets"]["breast-cancer"]["ema"]["runs"] == 1


def test_heldout_test_labels_cannot_change_training_or_validation_history():
    data = prepare_public_data("digits", seed=1)
    altered = data.y.copy()
    altered[data.test] = (altered[data.test]+1)%10
    baseline = run_public_case(data, method="adaptive-kalman", rounds=3, local_steps=1)
    changed = run_public_case(replace(data, y=altered), method="adaptive-kalman", rounds=3, local_steps=1)
    for a,b in zip(baseline["history"], changed["history"]):
        assert a["accuracy"] == b["accuracy"] and a["loss"] == b["loss"]
    assert baseline["schedule_sha256"] == changed["schedule_sha256"]


@pytest.mark.parametrize("kwargs", [{"rounds": 2}, {"local_steps": 0}, {"participation": True}, {"method": "x"}])
def test_invalid_public_experiment_settings(kwargs):
    data = prepare_public_data("digits", seed=1)
    options = dict(method="ema", rounds=3, local_steps=1)
    options.update(kwargs)
    with pytest.raises(ValueError): run_public_case(data, **options)
