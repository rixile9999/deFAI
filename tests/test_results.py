import json
import math

import numpy as np
from flwr.app import MetricRecord
from flwr.serverapp.strategy import Result

from defai.privacy.dp import FixedClipping
from defai.records import to_record
from defai.results import build_history, final_arrays, jsonable, privacy_summary, run_name, write_summary

BASE = {"dataset": "mnist", "strategy": "fedavg", "partitioner": "iid", "dp": "none", "dp-side": "server",
        "attack": "none", "num-malicious": 0}


def test_run_name_lists_only_the_axes_that_are_switched_on():
    assert run_name(BASE, "T") == "mnist_fedavg_iid_T"
    assert run_name({**BASE, "dp": "adaptive", "dp-side": "client"}, "T") == "mnist_fedavg_iid_dp-adaptive-client_T"
    assert run_name({**BASE, "attack": "sign-flip", "num-malicious": 2}, "T") == "mnist_fedavg_iid_sign-flipx2_T"
    assert run_name({**BASE, "attack": "sign-flip", "num-malicious": 0}, "T") == "mnist_fedavg_iid_T"  # no attacker, no tag
    assert run_name(BASE).startswith("mnist_fedavg_iid_20")  # timestamp by default


def test_build_history_merges_the_three_metric_sources_per_round():
    result = Result()
    result.evaluate_metrics_serverapp = {0: MetricRecord({"test_acc": 0.1}), 1: MetricRecord({"test_acc": 0.5})}
    result.train_metrics_clientapp = {1: MetricRecord({"train_loss": 2.0})}
    result.evaluate_metrics_clientapp = {1: MetricRecord({"eval_acc": 0.4})}
    assert build_history(result) == [
        {"round": 0, "test_acc": 0.1},
        {"round": 1, "test_acc": 0.5, "train_loss": 2.0, "eval_acc": 0.4},
    ]


def test_jsonable_replaces_only_non_finite_floats():
    data = {"a": float("inf"), "b": [1.0, float("nan"), {"c": -math.inf}], "d": 3, "e": "x", "f": (1.5,)}
    assert jsonable(data) == {"a": None, "b": [1.0, None, {"c": None}], "d": 3, "e": "x", "f": [1.5]}


def test_summary_is_strict_json_even_with_infinite_epsilon(tmp_path):
    dp = FixedClipping(0.0, 1.0)  # no noise -> epsilon is infinite
    dp.accountant.compose(0.0)
    summary = {"config": {}, "history": [{"round": 1, "dp-epsilon": dp.epsilon}], "privacy": privacy_summary(dp)}
    path = tmp_path / "history.json"
    write_summary(path, summary)
    text = path.read_text()
    assert "Infinity" not in text and "NaN" not in text
    data = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # strict parser
    assert data["privacy"]["epsilon"] is None and data["privacy"]["epsilon_unbounded"] is True
    assert data["history"][0]["dp-epsilon"] is None


def test_privacy_summary_for_a_finite_budget():
    dp = FixedClipping(1.0, 1.0, delta=1e-5)
    dp.accountant.compose(1.0, 3)
    out = privacy_summary(dp)
    assert out["epsilon"] == dp.epsilon and out["epsilon_unbounded"] is False and out["rounds_accounted"] == 3


def test_final_model_falls_back_to_the_initial_one_when_no_round_trained():
    initial = to_record({"w": np.arange(3, dtype=np.float32)})
    empty = Result()
    assert final_arrays(empty, initial) is initial  # used to save an empty state dict
    trained = Result()
    trained.arrays = to_record({"w": np.ones(3, dtype=np.float32)})
    assert final_arrays(trained, initial) is trained.arrays
