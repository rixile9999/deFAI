"""Interchange preserves real numeric state; it never resumes training."""

import json

import numpy as np
import pytest
import torch

from defai.cli import main
from defai.exchange import export_result, file_hash
from defai.task import build_model


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    torch.manual_seed(42)
    model = build_model("synthetic").eval()
    torch.save(model.state_dict(), run / "final_model.pt")
    result = {
        "config": {"dataset": "synthetic", "strategy": "fedadam-kalman"},
        "history": [
            {"round": 0, "test_acc": 0.1, "test_loss": 2.3},
            {"round": 1, "test_acc": 0.7, "test_loss": 0.8},
        ],
        "backend": "in_process",
        "research_only": True,
        "network_transport_measured": False,
    }
    (run / "history.json").write_text(json.dumps(result), encoding="utf-8")
    return run, model, result


def test_numeric_roundtrip_and_executable_model(completed_run, tmp_path):
    run, model, original = completed_run
    output = tmp_path / "교환_결과"
    receipt = export_result(run, output)
    data = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert receipt["format"] == data["format"] == "federated-result-v1"
    assert data["result"] == original
    assert data["source_history_sha256"] == file_hash(run / "history.json")
    with np.load(output / "weights.npz", allow_pickle=False) as archive:
        for key, tensor in model.state_dict().items():
            np.testing.assert_array_equal(archive[key], tensor.numpy())
            assert archive[key].dtype == tensor.numpy().dtype
            assert data["model"]["tensors"][key] == {
                "shape": list(tensor.shape),
                "dtype": archive[key].dtype.str,
            }
    with (output / "model.ts").open("rb") as stream:
        restored = torch.jit.load(stream).eval()
    inputs = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    for item in data["artifacts"].values():
        assert item["sha256"] == file_hash(output / item["path"])
    assert data["metric_specs"]["test_acc"]["unit"] == "ratio"
    assert data["model"]["input_schema"]["preprocessing"] == "already_normalized_tensor"


def test_cli_export_and_existing_output(completed_run, tmp_path, capsys):
    run, _, _ = completed_run
    output = tmp_path / "bundle"
    assert main(["export-result", str(run), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["output"] == str(output.resolve())
    before = file_hash(output / "result.json")
    assert main(["export-result", str(run), "--output", str(output)]) == 2
    assert file_hash(output / "result.json") == before


@pytest.mark.parametrize(
    "change",
    [
        "missing_history",
        "empty_history",
        "oversized_history",
        "diagnostic_text",
        "diagnostic_boolean",
        "duplicate_round",
        "negative_round",
        "nan",
        "infinite",
        "accuracy",
        "boolean",
        "loss",
        "dataset",
        "strategy",
    ],
)
def test_invalid_history_never_publishes(completed_run, tmp_path, change):
    run, _, data = completed_run
    if change == "missing_history":
        del data["history"]
    elif change == "empty_history":
        data["history"] = []
    elif change == "oversized_history":
        data["history"] = [
            {"round": i, "test_acc": 0.1, "test_loss": 2.3} for i in range(10002)
        ]
    elif change == "diagnostic_text":
        data["history"][0]["diagnostic"] = "not numeric"
    elif change == "diagnostic_boolean":
        data["history"][0]["diagnostic"] = True
    elif change == "duplicate_round":
        data["history"][1]["round"] = 0
    elif change == "negative_round":
        data["history"][0]["round"] = -1
    elif change == "nan":
        data["history"][0]["test_loss"] = float("nan")
    elif change == "infinite":
        data["history"][0]["test_loss"] = float("inf")
    elif change == "accuracy":
        data["history"][0]["test_acc"] = 70
    elif change == "boolean":
        data["history"][0]["test_acc"] = True
    elif change == "loss":
        data["history"][0]["test_loss"] = -1
    elif change == "dataset":
        data["config"]["dataset"] = "../custom"
    else:
        del data["config"]["strategy"]
    (run / "history.json").write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "bundle"
    with pytest.raises(ValueError):
        export_result(run, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["keys", "shape", "dtype", "nan"])
def test_mismatched_numeric_model_is_rejected(completed_run, tmp_path, change):
    run, model, _ = completed_run
    state = model.state_dict()
    key = next(iter(state))
    if change == "keys":
        state.pop(key)
    elif change == "shape":
        state[key] = torch.zeros(1)
    elif change == "dtype":
        state[key] = state[key].double()
    else:
        state[key].flatten()[0] = float("nan")
    torch.save(state, run / "final_model.pt")
    with pytest.raises(ValueError, match="Model tensor"):
        export_result(run, tmp_path / "bundle")


def test_history_limit_never_publishes(completed_run, tmp_path, monkeypatch):
    run, _, _ = completed_run
    output = tmp_path / "bundle"
    import defai.exchange as exchange

    monkeypatch.setattr(exchange, "MAX_HISTORY_BYTES", 1)
    with pytest.raises(ValueError, match="4 MiB"):
        export_result(run, output)
    assert not output.exists()


def test_existing_empty_output_is_preserved(completed_run, tmp_path):
    run, _, _ = completed_run
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_result(run, output)
    assert list(output.iterdir()) == []
