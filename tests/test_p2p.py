"""Decentralized mixing invariants and independent-process HTTP training."""

import copy
import json
import time

import numpy as np
import pytest

from defai.p2p import (
    PeerServer,
    config,
    fetch,
    local_step,
    mixing_weights,
    run_demo,
    run_peer,
    snapshot,
    validate_snapshot,
)


def graph(neighbors):
    return {
        f"p{i}": {
            "endpoint": f"http://127.0.0.1:{18000 + i}",
            "neighbors": [f"p{n}" for n in ns],
        }
        for i, ns in enumerate(neighbors)
    }


def manifest(tmp_path, peers):
    value = {
        "format": "defai-p2p-v1",
        "job_id": "test",
        "features": 2,
        "rounds": 1,
        "learning_rate": 0.1,
        "peers": peers,
    }
    path = tmp_path / "peers.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize(
    "neighbors",
    [[[1], [0]], [[1, 3], [0, 2], [1, 3], [0, 2]], [[1, 2, 3], [0], [0], [0]]],
)
def test_metropolis_preserves_mean_and_reaches_consensus(tmp_path, neighbors):
    peers = graph(neighbors)
    path, _ = manifest(tmp_path, peers)
    config(path)
    names = list(peers)
    matrix = np.array(
        [[mixing_weights(peers, n).get(m, 0) for m in names] for n in names]
    )
    np.testing.assert_allclose(matrix.sum(axis=0), 1)
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    np.testing.assert_array_equal(matrix, matrix.T)
    values = np.arange(len(peers), dtype=float)
    for _ in range(100):
        values = matrix @ values
    np.testing.assert_allclose(values, (len(peers) - 1) / 2, atol=1e-10)


@pytest.mark.parametrize(
    "change", ["round", "sender", "job", "shape", "hash", "nan", "boolean", "format"]
)
def test_wrong_peer_models_are_rejected(change):
    model = snapshot("jobhash", "p0", 1, np.zeros(3))
    if change == "round":
        model["round"] = 0
    elif change == "sender":
        model["sender"] = "p1"
    elif change == "job":
        model["config_sha256"] = "another"
    elif change == "shape":
        model["weights"].pop()
    elif change == "hash":
        model["weights"][0] = 1
    elif change == "nan":
        model["weights"][0] = float("nan")
    elif change == "boolean":
        model["weights"][0] = True
    else:
        model["format"] = "unknown"
    with pytest.raises(ValueError):
        validate_snapshot(model, "jobhash", "p0", 1, 3)


@pytest.mark.parametrize(
    "change",
    [
        "oneway",
        "isolated",
        "self",
        "duplicate",
        "unknown",
        "rate",
        "rounds",
        "endpoint",
        "features",
    ],
)
def test_invalid_graph_and_configuration(tmp_path, change):
    path, value = manifest(tmp_path, graph([[1], [0]]))
    if change == "oneway":
        value["peers"]["p1"]["neighbors"] = []
    elif change == "isolated":
        value["peers"].update(graph([[1], [0], [3], [2]]))
    elif change == "self":
        value["peers"]["p0"]["neighbors"] = ["p0"]
    elif change == "duplicate":
        value["peers"]["p0"]["neighbors"] *= 2
    elif change == "unknown":
        value["peers"]["p0"]["neighbors"] = ["p9"]
    elif change == "rate":
        value["learning_rate"] = float("nan")
    elif change == "rounds":
        value["rounds"] = True
    elif change == "endpoint":
        value["peers"]["p0"]["endpoint"] = "http://example.org:80"
    else:
        value["features"] = -1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        config(path)


def test_snapshot_immutable_and_unavailable_round_is_bounded():
    server = PeerServer("http://127.0.0.1:0")
    endpoint = f"http://127.0.0.1:{server.server.server_port}"
    model = snapshot("job", "p0", 1, np.zeros(3))
    try:
        server.publish(1, model)
        server.publish(1, copy.deepcopy(model))
        actual, size = fetch(endpoint, 1, time.monotonic() + 1)
        assert actual == model and size > 0
        with pytest.raises(ValueError):
            server.publish(1, {"different": True})
        with pytest.raises(TimeoutError):
            fetch(endpoint, 2, time.monotonic() + 0.1)
    finally:
        server.close()


def test_one_step_full_graph_matches_independent_average():
    x0, x1 = np.array([[1.0, 1.0], [-1.0, 1.0]]), np.array([[2.0, 1.0], [0.0, 1.0]])
    y0, y1 = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    w = np.zeros(2)
    actual = (local_step(w, x0, y0, 0.1) + local_step(w, x1, y1, 0.1)) / 2
    reference = -0.1 * ((x0.T @ (0.5 - y0) / 2) + (x1.T @ (0.5 - y1) / 2)) / 2
    np.testing.assert_array_equal(actual, reference)


def test_peer_failure_does_not_claim_completed_round(tmp_path):
    path, _ = manifest(tmp_path, graph([[1], [0]]))
    np.savez(
        tmp_path / "data.npz",
        x=np.array([[0.0, 1.0], [1.0, 0.0]]),
        y=np.array([0.0, 1.0]),
    )
    with pytest.raises(TimeoutError):
        run_peer(path, "p0", tmp_path / "data.npz", tmp_path / "result", timeout=0.1)
    result = json.loads((tmp_path / "result/result.json").read_text())
    assert result["status"] == "failed" and len(result["history"]) == 1


@pytest.mark.parametrize("topology", ["ring", "full", "star"])
def test_independent_peer_process_training(tmp_path, topology):
    result = run_demo(
        tmp_path / "run", peers=4, rounds=12, topology=topology, timeout=10
    )
    assert result["final_mean_local_loss"] < result["initial_mean_local_loss"] - 0.1
    assert not result["central_aggregator"] and not result["udt_connected"]
    peers = [
        json.loads(p.read_text()) for p in (tmp_path / "run").glob("peer-*/result.json")
    ]
    assert len(peers) == 4
    assert all(p["status"] == "completed" and len(p["history"]) == 13 for p in peers)
    assert all(p["application_payload_bytes_received"] > 0 for p in peers)
    if topology == "full":
        for peer in peers[1:]:
            np.testing.assert_array_equal(peer["weights"], peers[0]["weights"])
