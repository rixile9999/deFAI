"""Protocol validation, concurrent rounds, persistence and failure transactions."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json

import numpy as np
import pytest

from defai.mobile import create_app


def payload(client, client_id="one", value=1.):
    current = client.get("/get_parameters").get_json()
    for parameter in current["parameters"]:
        parameter["data"] = (np.ones(parameter["shape"]) * value).tolist()
    return {**current, "client_id": client_id}


def test_waits_for_clients_rejects_duplicates_and_stale_rounds(tmp_path):
    app = create_app(tmp_path, expected_clients=2)
    client = app.test_client()
    one, two = payload(client), payload(client, "two", 3.)
    two["num_examples"] = 3
    assert client.post("/submit_parameters", json=one).get_json()["aggregated"] is False
    assert client.post("/submit_parameters", json=one).status_code == 409
    assert client.post("/submit_parameters", json=two).get_json()["next_round"] == 1
    actual = client.get("/get_parameters").get_json()
    np.testing.assert_allclose(actual["parameters"][0]["data"], [[2.5], [2.5]])
    assert client.post("/submit_parameters", json=two).status_code == 409


@pytest.mark.parametrize("mutation", ("shape", "nan", "weight", "missing_round", "missing_client", "empty"))
def test_rejects_invalid_payloads_without_mutating_state(tmp_path, mutation):
    client = create_app(tmp_path).test_client()
    data = payload(client)
    if mutation == "shape":
        data["parameters"][0]["shape"] = [200, 1]
    elif mutation == "nan":
        data["parameters"][0]["data"] = [[float("nan")], [1.]]
    elif mutation == "weight":
        data["num_examples"] = -1
    elif mutation == "missing_round":
        data.pop("round")
    elif mutation == "missing_client":
        data.pop("client_id")
    else:
        data = []
    before = (tmp_path / "coordinator.json").read_bytes()
    response = client.post("/submit_parameters", json=data)
    assert response.status_code in (400, 409)
    assert (tmp_path / "coordinator.json").read_bytes() == before


def test_partial_round_survives_restart(tmp_path):
    client = create_app(tmp_path, expected_clients=2).test_client()
    client.post("/submit_parameters", json=payload(client))
    restarted = create_app(tmp_path, expected_clients=2).test_client()
    assert restarted.get("/status").get_json()["connected_clients"] == 1
    assert restarted.post("/submit_parameters", json=payload(restarted, "two")).get_json()["next_round"] == 1
    assert create_app(tmp_path, expected_clients=2).test_client().get("/get_parameters").get_json()["round"] == 1
    with pytest.raises(ValueError, match="mismatch"):
        create_app(tmp_path, expected_clients=3)


def test_concurrent_clients_aggregate_exactly_once(tmp_path):
    app = create_app(tmp_path, expected_clients=12)
    base = payload(app.test_client())

    def submit(index):
        data = copy.deepcopy(base)
        data["client_id"] = str(index)
        return app.test_client().post("/submit_parameters", json=data).status_code

    with ThreadPoolExecutor(max_workers=6) as executor:
        assert list(executor.map(submit, range(12))) == [200] * 12
    status = app.test_client().get("/status").get_json()
    assert status["current_round"] == 1 and status["connected_clients"] == 0


def test_disk_failure_leaves_memory_and_persisted_state_unchanged(tmp_path, monkeypatch):
    import defai.mobile as mobile
    client = create_app(tmp_path).test_client()
    data = payload(client)
    before = json.loads((tmp_path / "coordinator.json").read_text())

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mobile, "write_json", fail)
    assert client.post("/submit_parameters", json=data).status_code == 503
    assert client.get("/get_parameters").get_json()["round"] == before["round"]
    assert json.loads((tmp_path / "coordinator.json").read_text()) == before


def test_server_does_not_claim_unperformed_training_or_evaluation(tmp_path):
    client = create_app(tmp_path).test_client()
    assert client.post("/train").status_code == 405
    assert client.get("/evaluate").get_json()["evaluation_result"]["mae"] is None
