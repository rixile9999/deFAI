"""server_app.main driven in-process against the real ClientApp (no Flower runtime)."""

import json
import tomllib
from pathlib import Path

import pytest
import torch
from flwr.app import Context, RecordDict

from defai import server_app
from defai.task import build_model
from conftest import InProcessGrid

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
NUM_CLIENTS = 4


def run_config(tmp_path, **overrides):
    base = tomllib.loads(PYPROJECT.read_text())["tool"]["flwr"]["app"]["config"]
    return {**base, "dataset": "synthetic", "batch-size": 64, "num-server-rounds": 2,
            "fraction-train": 1.0, "fraction-evaluate": 1.0, "output-dir": str(tmp_path), **overrides}


def run(tmp_path, **overrides):
    cfg = run_config(tmp_path, **overrides)
    grid = InProcessGrid(NUM_CLIENTS, cfg)
    server_app.main(grid, Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config=cfg))
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return out, json.loads((out / "history.json").read_text()), grid


def test_full_stack_run_writes_a_complete_strict_json_history(tmp_path):
    out, summary, grid = run(tmp_path)
    rounds = [row["round"] for row in summary["history"]]
    assert rounds == [0, 1, 2]  # initial evaluation + two rounds
    final = summary["history"][-1]
    assert {"test_acc", "test_loss", "train_loss", "eval_acc", "eval_loss"} <= set(final)
    assert {"eval_acc_min", "eval_acc_max", "eval_acc_std"} <= set(final)  # per-client spread, not just the mean
    assert final["eval_acc_min"] <= final["eval_acc"] <= final["eval_acc_max"]
    assert final["test_acc"] > summary["history"][0]["test_acc"]  # learned something on the synthetic task
    assert summary["config"]["strategy"] == "fedavg" and "privacy" not in summary
    assert out.name.startswith("synthetic_fedavg_iid_")
    assert not (out / "final_model.pt").exists()  # save-model is off by default


def test_every_round_reaches_every_client_with_the_real_message_flow(tmp_path):
    _, _, grid = run(tmp_path)
    types = [m.metadata.message_type for m in grid.sent]
    assert types.count("train") == types.count("evaluate") == 2 * NUM_CLIENTS


def test_save_model_writes_a_loadable_state_dict(tmp_path):
    out, _, _ = run(tmp_path, **{"save-model": True})
    state = torch.load(out / "final_model.pt")
    build_model("synthetic").load_state_dict(state)  # same keys and shapes as the model


def test_scaffold_and_moon_keep_per_client_state_across_the_real_rounds(tmp_path):
    for strategy, state_key in [("scaffold", "scaffold-c"), ("moon", "moon-previous")]:
        sub = tmp_path / strategy
        sub.mkdir()
        _, _, grid = run(sub, strategy=strategy)
        assert all(state_key in ctx.state for ctx in grid.contexts.values()), strategy


def test_dp_run_reports_the_privacy_spent(tmp_path):
    _, summary, _ = run(tmp_path, dp="fixed", **{"dp-noise-multiplier": 0.05, "dp-delta": 1e-5})
    privacy = summary["privacy"]
    assert privacy["rounds_accounted"] == 2 and privacy["delta"] == 1e-5
    assert privacy["epsilon"] > 0 and privacy["epsilon_unbounded"] is False
    assert summary["history"][-1]["dp-epsilon"] == pytest.approx(privacy["epsilon"])


def test_a_budget_below_the_first_round_trains_nothing_and_saves_the_initial_model(tmp_path):
    out, summary, grid = run(tmp_path, dp="fixed", **{"dp-noise-multiplier": 1.0, "dp-max-epsilon": 0.01, "save-model": True})
    assert summary["privacy"]["rounds_accounted"] == 0
    assert [m for m in grid.sent if m.metadata.message_type == "train"] == []
    torch.manual_seed(summary["config"]["seed"])  # the initial model is seeded from the run config
    expected = build_model("synthetic").state_dict()
    saved = torch.load(out / "final_model.pt")
    assert all(torch.equal(saved[k], expected[k]) for k in expected)


def test_an_invalid_config_fails_before_any_client_is_contacted(tmp_path):
    cfg = run_config(tmp_path, attack="signflip")
    grid = InProcessGrid(NUM_CLIENTS, cfg)
    with pytest.raises(ValueError, match="attack"):
        server_app.main(grid, Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config=cfg))
    assert grid.sent == [] and list(tmp_path.iterdir()) == []


def test_training_survives_clients_that_drop_out_and_the_traffic_shows_it(tmp_path):
    _, summary, _ = run(tmp_path, **{"dropout-rate": 0.5, "num-server-rounds": 4})
    history = summary["history"]
    assert [row["round"] for row in history] == [0, 1, 2, 3, 4]
    trained = [row for row in history if "comm-mb-up" in row]  # rounds in which at least one client answered
    assert trained, "with 4 clients per round at least one round should have survivors"
    # missing replies mean less upload than download: the accounting sees the dropouts
    assert any(row["comm-mb-up"] < row["comm-mb-down"] for row in trained)
    assert all(row["comm-mb-up"] <= row["comm-mb-down"] for row in trained)


def test_compressed_uploads_are_accounted_at_their_real_size_and_training_still_learns(tmp_path):
    _, summary, _ = run(tmp_path, compression="topk", **{"compression-ratio": 0.2, "error-feedback": True, "num-server-rounds": 3})
    rounds = summary["history"][1:]
    assert all(row["comm-mb-up"] < 0.5 * row["comm-mb-down"] for row in rounds)   # ~0.4 bytes/param vs ~4 dense
    assert all(0.0 < row["compression-error"] < 1.0 for row in rounds)
    assert summary["history"][-1]["test_acc"] > summary["history"][0]["test_acc"] + 0.1


def test_personalisation_metrics_reach_the_history_next_to_the_global_ones(tmp_path):
    _, summary, _ = run(tmp_path, partitioner="dirichlet", **{"dirichlet-alpha": 0.1, "personalize-epochs": 1, "num-server-rounds": 2})
    final = summary["history"][-1]
    assert {"eval_acc", "eval_acc_personalized", "eval_acc_personalized_min", "eval_acc_personalized_std"} <= set(final)
    # on skewed clients a local epoch on top of the global model should not make them worse on average
    assert final["eval_acc_personalized"] >= final["eval_acc"] - 0.05
