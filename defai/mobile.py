"""Persistent, single-process round coordinator for the Android linear model.

Clients train locally. This server validates and averages two weights and one
bias; no TensorFlow runtime or fabricated server-side evaluation is required.
"""
import copy
import json
from pathlib import Path
import threading

import numpy as np

from defai.io import write_json


def _parameters(payload):
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("parameters must contain weight and bias layers")
    normalized = []
    for index, shape in enumerate(([2, 1], [1])):
        layer = payload[index]
        if not isinstance(layer, dict) or layer.get("layer") != index or layer.get("shape") != shape:
            raise ValueError("Expected layer 0 shape [2,1], then layer 1 shape [1]")
        data = np.asarray(layer.get("data"), dtype=np.float64)
        if data.size != int(np.prod(shape)) or not np.isfinite(data).all():
            raise ValueError("Layer data must have the exact size and finite values")
        normalized.append({"layer": index, "shape": shape, "size": data.size,
                           "data": data.reshape(shape).tolist()})
    return normalized


def create_app(output, *, expected_clients=1):
    """Create one coordinator per directory; use one WSGI process with threads."""
    from flask import Flask, jsonify, request

    if isinstance(expected_clients, bool) or not isinstance(expected_clients, int) or expected_clients < 1:
        raise ValueError("expected_clients must be a positive integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "coordinator.json"
    lock = threading.RLock()
    state = {"schema": 1, "round": 0, "expected_clients": expected_clients,
             "parameters": _parameters([{"layer": 0, "shape": [2, 1], "data": [[0.], [0.]]},
                                         {"layer": 1, "shape": [1], "data": [0.]}]), "pending": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema") != 1 or state.get("expected_clients") != expected_clients:
            raise ValueError("Stored coordinator schema/client-count mismatch")
        state["parameters"] = _parameters(state["parameters"])
        if type(state.get("round")) is not int or state["round"] < 0 or not isinstance(state.get("pending"), dict):
            raise ValueError("Invalid stored coordinator state")
        for pending in state["pending"].values():
            _parameters(pending["parameters"])
    else:
        write_json(state_path, state, overwrite=False)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    @app.get("/health")
    def health():
        return jsonify(status="healthy", stage="research_only")

    @app.get("/get_parameters")
    def get_parameters():
        with lock:
            return jsonify(status="success", round=state["round"], parameters=state["parameters"])

    def commit_round(candidate):
        records = list(candidate["pending"].values())
        weights = np.asarray([row["num_examples"] for row in records], dtype=np.float64)
        weights /= weights.sum()
        for index in range(2):
            average = sum(w * np.asarray(row["parameters"][index]["data"]) for w, row in zip(weights, records))
            if not np.isfinite(average).all():
                raise ValueError("Aggregate is non-finite")
            candidate["parameters"][index]["data"] = average.tolist()
        candidate["pending"] = {}
        candidate["round"] += 1

    @app.post("/submit_parameters")
    def submit():
        nonlocal state
        payload = request.get_json(silent=True)
        try:
            if not isinstance(payload, dict):
                raise ValueError("A JSON object is required")
            client = payload.get("client_id")
            if not isinstance(client, str) or not client.strip() or len(client) > 200:
                raise ValueError("A nonempty client_id of at most 200 characters is required")
            examples = payload.get("num_examples", 1)
            if type(examples) is not int or not 1 <= examples <= 10**9:
                raise ValueError("num_examples must be an integer between 1 and 1000000000")
            parameters = _parameters(payload.get("parameters"))
            with lock:
                if type(payload.get("round")) is not int or payload["round"] != state["round"]:
                    return jsonify(status="error", message="Stale or missing round; fetch parameters again", round=state["round"]), 409
                if client in state["pending"]:
                    return jsonify(status="error", message="Client already submitted this round"), 409
                candidate = copy.deepcopy(state)
                candidate["pending"][client] = {"parameters": parameters, "num_examples": examples}
                complete = len(candidate["pending"]) == expected_clients
                if complete:
                    commit_round(candidate)
                write_json(state_path, candidate)
                state = candidate
                return jsonify(status="success", aggregated=complete, next_round=state["round"], pending=len(state["pending"]))
        except (ValueError, TypeError, OverflowError) as error:
            return jsonify(status="error", message=str(error)), 400
        except OSError:
            app.logger.exception("Coordinator persistence failed; memory state was not advanced")
            return jsonify(status="error", message="Persistence failed; round unchanged"), 503

    @app.post("/aggregate")
    def aggregate():
        with lock:
            return jsonify(status="error", message="Rounds aggregate automatically only after all expected clients submit",
                           pending=len(state["pending"]), expected_clients=expected_clients), 409

    @app.get("/status")
    def status():
        with lock:
            return jsonify(current_round=state["round"], connected_clients=len(state["pending"]),
                           expected_clients=expected_clients, session_id=1)

    @app.get("/evaluate")
    def evaluate():
        return jsonify(status="not_evaluated", evaluation_result={"loss": None, "mae": None,
                       "num_examples": 0, "message": "Evaluation requires held-out client data"})

    @app.post("/train")
    def train():
        return jsonify(status="error", message="Training runs on clients; server aggregates only"), 405

    return app
