"""Decentralized local-train/Metropolis-mix research peers over HTTP.

No central aggregator; each peer fetches immutable post-local models from its
neighbors. This first version is synchronous, static-membership binary logistic
regression with equal weight per peer objective, not sample-weighted FedAvg.
UDT is deliberately not connected to this module.

References: Lian et al., NeurIPS 2017; Xiao, Boyd & Lall, 2005 Metropolis consensus.
"""

from __future__ import annotations

import hashlib
import ipaddress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import ProxyHandler, build_opener

import numpy as np

from defai.io import write_json


def config(path, *, transport="http", allow_network=False):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Peer configuration exceeds 1 MiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("format") != "defai-p2p-v1"
        or not isinstance(value.get("job_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value["job_id"])
    ):
        raise ValueError("Unsupported peer configuration or job ID")
    peers = value.get("peers", {})
    if not isinstance(peers, dict) or not 2 <= len(peers) <= 64:
        raise ValueError("Configuration needs 2..64 peers")
    if type(value.get("features")) is not int or not 1 <= value["features"] <= 1024:
        raise ValueError("Feature count must be 1..1024")
    if type(value.get("rounds")) is not int or not 1 <= value["rounds"] <= 1000:
        raise ValueError("Rounds must be 1..1000")
    lr = value.get("learning_rate")
    if type(lr) not in (int, float) or not np.isfinite(lr) or not 0 < lr <= 1:
        raise ValueError("Learning rate must be finite in (0,1]")
    endpoints = set()
    for name, peer in peers.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or not isinstance(peer, dict):
            raise ValueError("Invalid peer name")
        endpoint = peer.get("endpoint", "")
        address = urlparse(endpoint)
        try:
            valid_host = (
                address.hostname == "127.0.0.1"
                if transport == "http"
                else bool(
                    ipaddress.ip_address(address.hostname).is_loopback or allow_network
                )
            )
        except (ValueError, TypeError):
            valid_host = False
        if (
            address.scheme != transport
            or not valid_host
            or not address.port
            or address.path not in ("", "/")
            or address.query
            or address.fragment
            or address.username
        ):
            raise ValueError(
                "Invalid peer endpoint or non-loopback network not enabled"
            )
        if endpoint.rstrip("/") in endpoints:
            raise ValueError("Duplicate peer endpoint")
        endpoints.add(endpoint.rstrip("/"))
        neighbors = peer.get("neighbors")
        if (
            not isinstance(neighbors, list)
            or not neighbors
            or any(not isinstance(n, str) for n in neighbors)
            or len(set(neighbors)) != len(neighbors)
            or name in neighbors
            or any(n not in peers for n in neighbors)
        ):
            raise ValueError("Invalid peer neighbors")
    for name, peer in peers.items():
        if any(
            name not in peers[neighbor]["neighbors"] for neighbor in peer["neighbors"]
        ):
            raise ValueError("Metropolis mixing needs reciprocal edges")
    reached, pending = set(), [next(iter(peers))]
    while pending:
        current = pending.pop()
        if current not in reached:
            reached.add(current)
            pending.extend(peers[current]["neighbors"])
    if reached != set(peers):
        raise ValueError("Peer graph must be connected")
    return value


def mixing_weights(peers, name):
    neighbors = peers[name]["neighbors"]
    weights = {
        n: 1 / (1 + max(len(neighbors), len(peers[n]["neighbors"]))) for n in neighbors
    }
    weights[name] = 1 - sum(weights.values())
    return weights


def model_hash(weights):
    return hashlib.sha256(np.asarray(weights, dtype="<f8").tobytes()).hexdigest()


def snapshot(job, sender, round_number, weights):
    return {
        "format": "defai-peer-model-v1",
        "config_sha256": job,
        "sender": sender,
        "round": round_number,
        "weights": weights.tolist(),
        "sha256": model_hash(weights),
    }


def validate_snapshot(value, job, sender, round_number, dimensions):
    if (
        not isinstance(value, dict)
        or value.get("format") != "defai-peer-model-v1"
        or value.get("config_sha256") != job
        or value.get("sender") != sender
        or type(value.get("round")) is not int
        or value["round"] != round_number
    ):
        raise ValueError("Peer model job/sender/round mismatch")
    raw = value.get("weights")
    if (
        not isinstance(raw, list)
        or len(raw) != dimensions
        or any(type(x) not in (int, float) for x in raw)
    ):
        raise ValueError("Peer model schema mismatch")
    weights = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(weights).all() or model_hash(weights) != value.get("sha256"):
        raise ValueError("Peer model failed integrity check")
    return weights


def local_step(weights, x, y, lr):
    logits = x @ weights
    probability = np.exp(-np.logaddexp(0, -logits))
    return weights - lr * (x.T @ (probability - y) / len(y))


def metrics(weights, x, y):
    logits = x @ weights
    return {
        "loss": float(np.mean(np.logaddexp(0, logits) - y * logits)),
        "accuracy": float(np.mean((logits >= 0) == y)),
    }


class PeerServer:
    def __init__(self, endpoint, neighbors=()):
        self.models, self.lock, self.readers = {}, threading.Lock(), set()
        self.neighbors = set(neighbors)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = urlparse(self.path)
                try:
                    if query.path != "/snapshot":
                        raise ValueError()
                    parameters = parse_qs(query.query, strict_parsing=True)
                    number = int(parameters["round"][0])
                    reader = parameters.get("reader", [""])[0]
                    with owner.lock:
                        payload = owner.models.get(number)
                    if payload is None:
                        self.send_error(404, "Round not published yet")
                        return
                except (ValueError, KeyError):
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                    self.wfile.flush()
                    if number == -1 and reader in owner.neighbors:
                        with owner.lock:
                            owner.readers.add(reader)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        address = urlparse(endpoint)
        self.server = ThreadingHTTPServer((address.hostname, address.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def publish(self, number, model):
        encoded = json.dumps(model, allow_nan=False, separators=(",", ":")).encode()
        with self.lock:
            if number in self.models and self.models[number] != encoded:
                raise ValueError("Published peer snapshots are immutable")
            self.models[number] = encoded

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def fetch(endpoint, round_number, deadline, reader=""):
    opener = build_opener(ProxyHandler({}))
    url = endpoint.rstrip("/") + f"/snapshot?round={round_number}&reader={reader}"
    while time.monotonic() < deadline:
        try:
            with opener.open(
                url, timeout=min(0.5, max(0.01, deadline - time.monotonic()))
            ) as response:
                payload = response.read(65537)
                if len(payload) > 65536:
                    raise ValueError("Peer model exceeds 64 KiB")
                return json.loads(payload), len(payload)
        except HTTPError as exc:
            if exc.code != 404:
                raise ValueError(f"Peer HTTP failure: {exc.code}") from exc
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.02)
    raise TimeoutError("Peer unavailable or did not publish required round")


def run_peer(manifest, name, data, output, *, timeout=30):
    cfg = config(manifest)
    if (
        name not in cfg["peers"]
        or not np.isfinite(timeout)
        or not 0.1 <= timeout <= 3600
    ):
        raise ValueError("Unknown peer or invalid timeout")
    with np.load(data, allow_pickle=False) as arrays:
        x, y = (
            np.asarray(arrays["x"], dtype=np.float64),
            np.asarray(arrays["y"], dtype=np.float64),
        )
    if (
        x.ndim != 2
        or x.shape[1] != cfg["features"]
        or not 1 <= len(x) <= 100000
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Expected finite x[N,features] and binary y[N]")
    x = np.column_stack((x, np.ones(len(x))))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    weights = np.zeros(cfg["features"] + 1)
    history = [{"round": 0, **metrics(weights, x, y)}]
    received = 0
    server = PeerServer(cfg["peers"][name]["endpoint"], cfg["peers"][name]["neighbors"])
    status = "failed"
    try:
        mix = mixing_weights(cfg["peers"], name)
        for number in range(1, cfg["rounds"] + 1):
            local = local_step(weights, x, y, cfg["learning_rate"])
            server.publish(number, snapshot(identity, name, number, local))
            deadline = time.monotonic() + timeout
            values = {name: local}
            for neighbor in cfg["peers"][name]["neighbors"]:
                message, size = fetch(
                    cfg["peers"][neighbor]["endpoint"], number, deadline
                )
                values[neighbor] = validate_snapshot(
                    message, identity, neighbor, number, len(weights)
                )
                received += size
            # Each peer computes only its own row of the symmetric mixing matrix.
            weights = sum(mix[peer] * values[peer] for peer in sorted(mix))
            history.append({"round": number, **metrics(weights, x, y)})
        # Neighbors may still be reading our final immutable post-local snapshot.
        # A separate ready marker prevents fast peers closing their server too early.
        server.publish(
            -1, {"complete": True, "config_sha256": identity, "sender": name}
        )
        deadline = time.monotonic() + timeout
        for neighbor in cfg["peers"][name]["neighbors"]:
            message, _ = fetch(cfg["peers"][neighbor]["endpoint"], -1, deadline, name)
            if message != {
                "complete": True,
                "config_sha256": identity,
                "sender": neighbor,
            }:
                raise ValueError("Invalid peer completion marker")
        while True:
            with server.lock:
                seen = server.readers == server.neighbors
            if seen:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Peers did not acknowledge our completion marker")
            time.sleep(0.01)
        status = "completed"
        return {
            "peer": name,
            "status": status,
            "history": history,
            "weights": weights.tolist(),
        }
    finally:
        server.close()
        write_json(
            output / "result.json",
            {
                "format": "defai-p2p-result-v1",
                "stage": "research_loopback",
                "peer": name,
                "config_sha256": identity,
                "status": status,
                "history": history,
                "weights": weights.tolist(),
                "application_payload_bytes_received": received,
                "central_aggregator": False,
                "transport": "http",
                "udt_connected": False,
            },
        )


def run_demo(output, *, peers=4, rounds=20, topology="ring", timeout=30):
    if (
        type(peers) is not int
        or not 2 <= peers <= 16
        or topology not in {"ring", "full", "star"}
    ):
        raise ValueError("Demo needs 2..16 peers and ring/full/star topology")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = [f"peer-{i}" for i in range(peers)]
    sockets = []
    try:
        for _ in names:
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            sockets.append(listener)
        entries = {}
        for i, (name, listener) in enumerate(zip(names, sockets)):
            neighbors = (
                set(names) - {name}
                if topology == "full" or topology == "star" and i == 0
                else {names[0]}
                if topology == "star"
                else {names[(i - 1) % peers], names[(i + 1) % peers]}
            )
            entries[name] = {
                "endpoint": f"http://127.0.0.1:{listener.getsockname()[1]}",
                "neighbors": sorted(neighbors),
            }
    finally:
        for listener in sockets:
            listener.close()
    manifest = output / "peers.json"
    write_json(
        manifest,
        {
            "format": "defai-p2p-v1",
            "job_id": "offline-demo",
            "features": 3,
            "rounds": rounds,
            "learning_rate": 0.2,
            "peers": entries,
        },
    )
    config(manifest)
    rng = np.random.default_rng(42)
    processes, logs = [], []
    try:
        for i, name in enumerate(names):
            x = rng.normal(size=(96, 3)) + (i - (peers - 1) / 2) * 0.3
            y = (
                x @ np.array([1.5, -1.0, 0.7]) + rng.normal(scale=0.2, size=96) > 0
            ).astype(float)
            path = output / f"{name}.npz"
            np.savez(path, x=x, y=y)
            log = (output / f"{name}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "defai",
                        "p2p",
                        "peer",
                        str(manifest),
                        "--peer",
                        name,
                        "--data",
                        str(path),
                        "--output",
                        str(output / name),
                        "--timeout",
                        str(timeout),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + (rounds + 2) * timeout + 10
        for process in processes:
            if process.wait(timeout=max(0.1, deadline - time.monotonic())):
                raise ValueError("Peer process failed; inspect per-peer logs")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    results = [
        json.loads((output / name / "result.json").read_text()) for name in names
    ]
    summary = {
        "stage": "research_loopback",
        "central_aggregator": False,
        "udt_connected": False,
        "topology": topology,
        "peers": peers,
        "rounds": rounds,
        "initial_mean_local_loss": float(
            np.mean([x["history"][0]["loss"] for x in results])
        ),
        "final_mean_local_loss": float(
            np.mean([x["history"][-1]["loss"] for x in results])
        ),
        "peer_model_spread_l2": float(
            max(
                np.linalg.norm(np.asarray(x["weights"]) - np.asarray(y["weights"]))
                for x in results
                for y in results
            )
        ),
    }
    write_json(output / "demo.json", summary)
    return summary
