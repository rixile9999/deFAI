"""Real UDT neighbor training under loss, delay, process death and safe resume."""

import json
import os
import signal
import subprocess
import socket
import threading
import time

import numpy as np
import pytest

from defai.io import write_json
from defai.p2p import local_step, mixing_weights
from defai.p2p_udt import (
    _incoming,
    peer_command,
    prepare_demo,
    seal,
    signed,
    transfer_id,
    unseal,
    verified,
)
from defai.udt_bridge import peer_lock
from test_udt import Relay, udt_jar as udt_jar


def start(
    manifest, jar, key, root, index, *, resume=False, timeout=45, attempt_timeout=5
):
    name = f"peer-{index}"
    log = (root / (name + ("-resume.log" if resume else ".log"))).open(
        "w", encoding="utf-8"
    )
    process = subprocess.Popen(
        peer_command(
            manifest,
            name,
            root / (name + ".npz"),
            root / name,
            jar,
            key,
            resume=resume,
            round_timeout=timeout,
            attempt_timeout=attempt_timeout,
            linger=3,
        ),
        stdout=log,
        stderr=log,
    )
    log.close()
    return process


def cleanup(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def complete(processes, root, timeout=140):
    deadline = time.monotonic() + timeout
    for process in processes:
        code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if code:
            logs = "\n".join(p.read_text(encoding="utf-8") for p in root.glob("*.log"))
            pytest.fail(logs)


def reference(root, manifest):
    cfg = json.loads(manifest.read_text())
    names = sorted(cfg["peers"])
    weights = {name: np.zeros(cfg["features"] + 1) for name in names}
    data = {}
    for name in names:
        with np.load(root / (name + ".npz"), allow_pickle=False) as arrays:
            data[name] = (
                np.column_stack((arrays["x"], np.ones(len(arrays["y"])))),
                arrays["y"],
            )
    for _ in range(cfg["rounds"]):
        local = {
            name: local_step(weights[name], *data[name], cfg["learning_rate"])
            for name in names
        }
        weights = {
            name: sum(
                coefficient * local[other]
                for other, coefficient in sorted(
                    mixing_weights(cfg["peers"], name).items()
                )
            )
            for name in names
        }
    return weights


@pytest.mark.parametrize(
    "change", ["signature", "round", "sender", "recipient", "job", "hash"]
)
def test_corrupt_or_replayed_model_is_not_used(tmp_path, change):
    key = b"a" * 32
    from defai.p2p import snapshot

    message = {
        **snapshot("job", "peer-1", 1, np.zeros(3)),
        "recipient": "peer-0",
        "kind": "model",
    }
    if change == "round":
        message["round"] = 2
    elif change == "sender":
        message["sender"] = "peer-2"
    elif change == "recipient":
        message["recipient"] = "peer-2"
    elif change == "job":
        message["config_sha256"] = "different"
    elif change == "hash":
        message["weights"][0] = 1
    envelope = signed(message, key)
    if change == "signature":
        envelope["hmac_sha256"] = "0" * 64
    identifier = transfer_id("job", "peer-1", "peer-0", "model", 1)
    destination = tmp_path / "inbox" / identifier / "payload.bin"
    write_json(
        destination, seal(envelope, key, ["job", "peer-1", "peer-0", "model", 1])
    )
    with pytest.raises(ValueError):
        _incoming(tmp_path, key, "job", "peer-1", "peer-0", "model", 1, 3)


def test_workspace_lock_and_signed_checkpoint(tmp_path):
    with (
        peer_lock(tmp_path / "worker.lock"),
        pytest.raises(ValueError),
        peer_lock(tmp_path / "worker.lock"),
    ):
        pass
    with peer_lock(tmp_path / "worker.lock"):
        pass
    path = tmp_path / "checkpoint.json"
    key = b"a" * 32
    write_json(path, signed({"round": 2}, key))
    assert verified(path, key) == {"round": 2}
    with pytest.raises(ValueError):
        verified(path, b"b" * 32)


@pytest.mark.parametrize("failure", ["ciphertext", "nonce", "key", "context"])
def test_aead_rejects_changed_ciphertext_key_and_context(tmp_path, failure):
    key = b"a" * 32
    context = ["job", "a", "b", "model", 1]
    payload = {"weights": [1.0, 2.0, 3.0]}
    sealed = seal(signed(payload, key), key, context)
    again = seal(signed(payload, key), key, context)
    assert sealed["nonce"] != again["nonce"]
    path = tmp_path / "wire.json"
    write_json(path, sealed)
    assert unseal(path, key, context) == payload
    if failure == "ciphertext":
        sealed["ciphertext"] = (
            "B" if sealed["ciphertext"][0] == "A" else "A"
        ) + sealed["ciphertext"][1:]
    elif failure == "nonce":
        sealed["nonce"] = "not-base64"
    elif failure == "key":
        key = b"b" * 32
    else:
        context[-1] = 2
    write_json(path, sealed)
    with pytest.raises(ValueError):
        unseal(path, key, context)


def test_receiver_supervisor_restarts_failed_java_inbox(udt_jar, tmp_path):
    from defai.udt_bridge import Receiver, deliver

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    receiver = Receiver(udt_jar, tmp_path / "inbox", "127.0.0.1", port, 5)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and receiver.process is None:
            time.sleep(0.02)
        old = receiver.process
        assert old is not None
        old.kill()
        old.wait()
        while time.monotonic() < deadline and receiver.restarts < 2:
            time.sleep(0.02)
        assert receiver.restarts >= 2
        source = tmp_path / "model.bin"
        source.write_bytes(bytes(range(256)) * 256)
        deliver(
            udt_jar,
            source,
            "restart-test",
            "round-1",
            "127.0.0.1",
            port,
            time.monotonic() + 20,
        )
        assert (
            tmp_path / "inbox/restart-test/payload.bin"
        ).read_bytes() == source.read_bytes()
    finally:
        receiver.close()


@pytest.mark.parametrize("topology", ["ring", "star", "full"])
def test_udt_learning_matches_reference_with_concurrent_neighbors(
    udt_jar, tmp_path, topology
):
    root = tmp_path / "run"
    manifest, jar, key = prepare_demo(
        root, peers=4, rounds=2, topology=topology, jar=udt_jar
    )
    processes = [start(manifest, jar, key, root, i) for i in range(4)]
    try:
        complete(processes, root)
    finally:
        cleanup(processes)
    expected = reference(root, manifest)
    for name, weights in expected.items():
        result = json.loads((root / name / "result.json").read_text())
        assert result["status"] == "completed" and result["transport"] == "java-udt"
        assert not result["central_aggregator"]
        np.testing.assert_array_equal(result["weights"], weights)
        assert len(result["history"]) == 3


def test_peer_crash_and_receiver_restart_resume_exactly_once(udt_jar, tmp_path):
    root = tmp_path / "restart"
    manifest, jar, key = prepare_demo(root, peers=2, rounds=4, jar=udt_jar)
    processes = [start(manifest, jar, key, root, i, timeout=60) for i in range(2)]
    try:
        deadline = time.monotonic() + 45
        checkpoint = root / "peer-0/checkpoint.json"
        secret = key.read_bytes()
        while time.monotonic() < deadline:
            if checkpoint.exists():
                saved = verified(checkpoint, secret)
                if saved["round"] >= 1 and saved.get("pending") is not None:
                    break
            time.sleep(0.02)
        assert verified(checkpoint, secret)["round"] >= 1
        assert verified(checkpoint, secret)["round"] < 4
        worker = json.loads((root / "peer-0/process.json").read_text())
        os.kill(worker["pid"], signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
        processes[0].wait()
        # Java receiver uses a per-process heartbeat lease and exits on orphaning.
        processes[0] = start(manifest, jar, key, root, 0, resume=True, timeout=60)
        complete(processes, root, timeout=170)
    finally:
        cleanup(processes)
    for name, weights in reference(root, manifest).items():
        result = json.loads((root / name / "result.json").read_text())
        np.testing.assert_array_equal(result["weights"], weights)
        assert [row["round"] for row in result["history"]] == list(range(5))


def test_loss_jitter_duplicate_and_total_outage_preserve_training(udt_jar, tmp_path):
    root = tmp_path / "faults"
    manifest, jar, key = prepare_demo(root, peers=3, rounds=2, jar=udt_jar)
    cfg = json.loads(manifest.read_text())
    from urllib.parse import urlparse

    reserved = []
    relays = []
    for peer in cfg["peers"].values():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", urlparse(peer["endpoint"]).port))
        reserved.append(sock)
    try:
        for peer, sock in zip(cfg["peers"].values(), reserved):
            backend = sock.getsockname()[1]
            relay = Relay(backend, loss_every=11, delay=0.01, drop_handshake=True)
            relay.jitter = 0.01
            relay.duplicate_every = 7
            relays.append(relay)
            peer["bind_port"] = backend
            peer["endpoint"] = f"udt://127.0.0.1:{relay.port}"
        write_json(manifest, cfg)
    finally:
        for sock in reserved:
            sock.close()
    processes = []
    outage = []

    def disconnect():
        time.sleep(1)
        for relay in relays:
            relay.blackout.set()
        outage.append("started")
        time.sleep(8)
        for relay in relays:
            relay.blackout.clear()
        outage.append("restored")

    thread = threading.Thread(target=disconnect, daemon=True)
    thread.start()
    try:
        processes = [start(manifest, jar, key, root, i, timeout=75) for i in range(3)]
        complete(processes, root, timeout=210)
    finally:
        cleanup(processes)
        thread.join(timeout=5)
        for relay in relays:
            relay.close()
    assert outage == ["started", "restored"] and sum(r.dropped for r in relays) > 0
    for name, weights in reference(root, manifest).items():
        result = json.loads((root / name / "result.json").read_text())
        assert result["status"] == "completed"
        np.testing.assert_array_equal(result["weights"], weights)
    assert (
        sum(
            json.loads((root / f"peer-{i}/result.json").read_text())[
                "statistics_this_process"
            ]["send_attempts"]
            for i in range(3)
        )
        > 18
    )


def test_extended_outage_pauses_then_resumes_without_lost_round(udt_jar, tmp_path):
    root = tmp_path / "paused"
    manifest, jar, key = prepare_demo(root, peers=2, rounds=2, jar=udt_jar)
    first = start(manifest, jar, key, root, 0, timeout=8)
    try:
        assert first.wait(timeout=20) != 0
    finally:
        cleanup([first])
    saved = verified(root / "peer-0/checkpoint.json", key.read_bytes())
    assert saved["status"] == "paused" and saved["round"] == 0
    processes = [
        start(manifest, jar, key, root, 0, resume=True),
        start(manifest, jar, key, root, 1),
    ]
    try:
        complete(processes, root)
    finally:
        cleanup(processes)
    for name, weights in reference(root, manifest).items():
        result = json.loads((root / name / "result.json").read_text())
        np.testing.assert_array_equal(result["weights"], weights)
        assert [r["round"] for r in result["history"]] == [0, 1, 2]


def test_high_latency_udp_rounds_match_reference(udt_jar, tmp_path):
    from urllib.parse import urlparse

    root = tmp_path / "high-latency"
    manifest, jar, key = prepare_demo(root, peers=2, rounds=2, jar=udt_jar)
    cfg = json.loads(manifest.read_text())
    relays = []
    reserved = []
    for peer in cfg["peers"].values():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", urlparse(peer["endpoint"]).port))
        reserved.append(sock)
    try:
        for peer, sock in zip(cfg["peers"].values(), reserved):
            relay = Relay(sock.getsockname()[1], delay=0.15)
            relay.jitter = 0.05
            relays.append(relay)
            peer["bind_port"] = sock.getsockname()[1]
            peer["endpoint"] = f"udt://127.0.0.1:{relay.port}"
        write_json(manifest, cfg)
    finally:
        for sock in reserved:
            sock.close()
    processes = []
    try:
        processes = [
            start(manifest, jar, key, root, i, timeout=60, attempt_timeout=12)
            for i in range(2)
        ]
        complete(processes, root, timeout=190)
    finally:
        cleanup(processes)
        for relay in relays:
            relay.close()
    for name, weights in reference(root, manifest).items():
        result = json.loads((root / name / "result.json").read_text())
        np.testing.assert_array_equal(result["weights"], weights)


@pytest.mark.parametrize(
    "change", ["data", "key", "checkpoint", "model", "job", "format", "pending"]
)
def test_resume_rejects_changed_identity_or_state(udt_jar, tmp_path, change):
    import hashlib
    from defai.p2p_udt import canonical, run_peer
    from defai.udt_bridge import sha256

    root = tmp_path / "state"
    manifest, jar, key = prepare_demo(root, peers=2, rounds=2, jar=udt_jar)
    cfg = json.loads(manifest.read_text())
    secret = key.read_bytes()
    output = root / "peer-0"
    output.mkdir()
    state = {
        "format": "defai-udt-peer-checkpoint-v1",
        "config_sha256": hashlib.sha256(canonical(cfg)).hexdigest(),
        "peer": "peer-0",
        "dataset_sha256": sha256(root / "peer-0.npz"),
        "round": 0,
        "weights": [0.0] * 4,
        "history": [{"round": 0}],
        "status": "paused",
    }
    if change == "data":
        with (root / "peer-0.npz").open("ab") as stream:
            stream.write(b"changed-input")
    elif change == "key":
        key.write_bytes(b"b" * 32)
    elif change == "model":
        state["weights"] = [0.0]
    elif change == "job":
        cfg["learning_rate"] += 0.1
        write_json(manifest, cfg)
    elif change == "format":
        state["format"] = "unknown-checkpoint"
    elif change == "pending":
        state["pending"] = {"round": 9, "weights": []}
    document = signed(state, secret)
    if change == "checkpoint":
        document["hmac_sha256"] = "0" * 64
    write_json(output / "checkpoint.json", document)
    before = (output / "checkpoint.json").read_bytes()
    with pytest.raises(ValueError):
        run_peer(manifest, "peer-0", root / "peer-0.npz", output, jar, key, resume=True)
    assert (output / "checkpoint.json").read_bytes() == before
