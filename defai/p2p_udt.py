"""Authenticated, checkpointed neighbor learning over real Java UDT inboxes.

Static reciprocal graph, equal-peer logistic objective. Transient faults retry;
outages beyond the round budget pause with a resumable checkpoint. No HTTP path.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

import numpy as np

from defai.io import write_json
from defai.p2p import (
    config,
    local_step,
    metrics,
    mixing_weights,
    snapshot,
    validate_snapshot,
)
from defai.udt_bridge import Receiver, deliver, peer_lock, sha256


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def signed(value, key):
    return {
        "payload": value,
        "hmac_sha256": hmac.new(key, canonical(value), hashlib.sha256).hexdigest(),
    }


def verified(path, key, *, limit=2 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("Signed file exceeds size limit")
    document = json.loads(path.read_text(encoding="utf-8"))
    return _verify_envelope(document, key)


def _verify_envelope(document, key):
    if not isinstance(document, dict) or set(document) != {"payload", "hmac_sha256"}:
        raise ValueError("Invalid authenticated envelope")
    expected = hmac.new(key, canonical(document["payload"]), hashlib.sha256).hexdigest()
    if not isinstance(document["hmac_sha256"], str) or not hmac.compare_digest(
        expected, document["hmac_sha256"]
    ):
        raise ValueError("Message authentication failed")
    return document["payload"]


def _cipher(key):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Install defai[transport] for encrypted UDT peer learning"
        ) from exc
    derived = hmac.new(key, b"defai-udt-peer-aes-gcm-v1", hashlib.sha256).digest()
    return AESGCM(derived)


def seal(envelope, key, context):
    nonce = secrets.token_bytes(12)
    ciphertext = _cipher(key).encrypt(nonce, canonical(envelope), canonical(context))
    return {
        "format": "defai-peer-aes-gcm-v1",
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def unseal(path, key, context):
    from cryptography.exceptions import InvalidTag

    if Path(path).stat().st_size > 65536:
        raise ValueError("Encrypted peer message exceeds 64 KiB")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("format") != "defai-peer-aes-gcm-v1"
    ):
        raise ValueError("Unencrypted/unsupported peer message rejected")
    try:
        nonce = base64.b64decode(document["nonce"], validate=True)
        ciphertext = base64.b64decode(document["ciphertext"], validate=True)
        if len(nonce) != 12:
            raise ValueError("Invalid message nonce")
        clear = _cipher(key).decrypt(nonce, ciphertext, canonical(context))
    except (InvalidTag, ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise ValueError("Peer message decryption/authentication failed") from exc
    return _verify_envelope(json.loads(clear), key)


def transfer_id(job, sender, recipient, kind, number):
    return hashlib.sha256(canonical([job, sender, recipient, kind, number])).hexdigest()


def _checkpoint(path, state, key):
    for attempt in range(6):
        try:
            write_json(path, signed(state, key))
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * 2**attempt)


def _incoming(workspace, key, job, sender, recipient, kind, number, dimensions):
    identifier = transfer_id(job, sender, recipient, kind, number)
    path = workspace / "inbox" / identifier / "payload.bin"
    try:
        if not path.is_file():
            return None
        message = unseal(path, key, [job, sender, recipient, kind, number])
    except (PermissionError, FileNotFoundError):
        # Windows can briefly deny opening a Java-published/duplicate-checked
        # immutable file. Retry within the round budget, never skip validation.
        return None
    if (
        not isinstance(message, dict)
        or message.get("recipient") != recipient
        or message.get("kind") != kind
        or message.get("config_sha256") != job
        or message.get("sender") != sender
        or type(message.get("round")) is not int
        or message["round"] != number
    ):
        raise ValueError("Misaddressed, stale or incompatible UDT peer message")
    if kind == "model":
        return validate_snapshot(message, job, sender, number, dimensions)
    if message.get("complete") is not True:
        raise ValueError("Invalid completion message")
    return message


def _exchange(
    cfg,
    name,
    number,
    kind,
    local,
    workspace,
    key,
    job,
    jar,
    receiver,
    round_timeout,
    attempt_timeout,
    allow_network,
    statistics,
):
    neighbors = cfg["peers"][name]["neighbors"]
    deadline = time.monotonic() + round_timeout
    pending = {}
    for neighbor in neighbors:
        identifier = transfer_id(job, name, neighbor, kind, number)
        source = workspace / "outbox" / (identifier + ".json")
        message = (
            {**snapshot(job, name, number, local), "recipient": neighbor, "kind": kind}
            if kind == "model"
            else {
                "recipient": neighbor,
                "sender": name,
                "kind": kind,
                "config_sha256": job,
                "round": number,
                "complete": True,
            }
        )
        if source.exists():
            if unseal(source, key, [job, name, neighbor, kind, number]) != message:
                raise ValueError("Saved outbox differs from deterministic local state")
        else:
            write_json(
                source,
                seal(signed(message, key), key, [job, name, neighbor, kind, number]),
                overwrite=False,
            )
        receipt = workspace / "outbox" / (identifier + ".receipt")
        if receipt.exists():
            saved = verified(receipt, key)
            if saved != {"transfer_id": identifier, "payload_sha256": sha256(source)}:
                raise ValueError("Invalid saved delivery receipt")
        else:
            pending[neighbor] = (identifier, source, receipt)
    # All outgoing neighbors make progress independently; one disconnected edge
    # cannot prevent the other neighbors receiving this round's immutable model.
    with ThreadPoolExecutor(max_workers=max(1, len(pending))) as executor:
        futures = {}
        for neighbor, (identifier, source, receipt) in pending.items():
            address = urlparse(cfg["peers"][neighbor]["endpoint"])
            future = executor.submit(
                deliver,
                jar,
                source,
                identifier,
                f"{kind}.r{number}",
                address.hostname,
                address.port,
                deadline,
                attempt_timeout=attempt_timeout,
                allow_network=allow_network,
            )
            futures[future] = (identifier, source, receipt)
        while futures:
            if not receiver.thread.is_alive():
                raise RuntimeError("UDT receiver supervisor failed")
            completed, _ = wait(futures, timeout=0.1)
            for future in completed:
                identifier, source, receipt = futures.pop(future)
                result = future.result()
                statistics["send_attempts"] += result["attempts"]
                statistics["resumed_bytes"] += result["resumed_bytes"]
                write_json(
                    receipt,
                    signed(
                        {"transfer_id": identifier, "payload_sha256": sha256(source)},
                        key,
                    ),
                )
    incoming = {}
    while len(incoming) < len(neighbors):
        for neighbor in neighbors:
            if neighbor not in incoming:
                value = _incoming(
                    workspace, key, job, neighbor, name, kind, number, len(local)
                )
                if value is not None:
                    incoming[neighbor] = value
        if len(incoming) == len(neighbors):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Required UDT peer model/receipt did not arrive before round deadline"
            )
        time.sleep(0.05)
    return incoming


def run_peer(
    manifest,
    name,
    data,
    output,
    jar,
    key_file,
    *,
    resume=False,
    round_timeout=120,
    attempt_timeout=5,
    allow_network=False,
    linger=None,
):
    cfg = config(manifest, transport="udt", allow_network=allow_network)
    if cfg.get("encryption") != "aes-256-gcm-v1":
        raise ValueError("Encrypted UDT peer configuration required")
    _cipher(b"\0" * 32)
    if name not in cfg["peers"] or not shutil.which("java"):
        raise ValueError("Unknown peer or Java runtime missing on PATH")
    if (
        not np.isfinite(round_timeout)
        or not np.isfinite(attempt_timeout)
        or not 3 <= attempt_timeout <= 3600
        or not attempt_timeout + 2 <= round_timeout <= 86400
    ):
        raise ValueError(
            "Need attempt_timeout >=3 and round_timeout >= attempt_timeout+2"
        )
    linger = max(2, 2 * attempt_timeout) if linger is None else linger
    if not np.isfinite(linger) or not 0 <= linger <= 86400:
        raise ValueError("Invalid completion linger interval")
    jar, key_file = Path(jar).resolve(), Path(key_file)
    if not jar.is_file() or sha256(jar) != cfg.get("udt_jar_sha256"):
        raise ValueError("Java transport artifact differs from the configured version")
    if key_file.stat().st_size != 32:
        raise ValueError("A private, raw 32-byte shared group key is required")
    key = key_file.read_bytes()
    if hashlib.sha256(key).hexdigest() != cfg.get("key_sha256"):
        raise ValueError("Group key does not match this job")
    peer = cfg["peers"][name]
    address = urlparse(peer["endpoint"])
    bind = peer.get("bind", address.hostname)
    bind_port = peer.get("bind_port", address.port)
    if not ipaddress.ip_address(bind).is_loopback and not allow_network:
        raise ValueError(
            "Non-loopback binding requires --allow-network in a trusted lab"
        )
    if type(bind_port) is not int or not 1 <= bind_port <= 65535:
        raise ValueError("Invalid peer bind port")
    dataset_hash = sha256(data)
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
        raise ValueError("Expected finite local x[N,features] and binary y[N]")
    x = np.column_stack((x, np.ones(len(x))))
    job = hashlib.sha256(canonical(cfg)).hexdigest()
    output = Path(output).resolve()
    if resume:
        if not (output / "checkpoint.json").is_file():
            raise ValueError("No checkpoint to resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with peer_lock(output / "worker.lock"):
        write_json(output / "process.json", {"pid": os.getpid(), "peer": name})
        (output / "outbox").mkdir(exist_ok=True)
        checkpoint = output / "checkpoint.json"
        if resume:
            state = verified(checkpoint, key)
            if (
                not isinstance(state, dict)
                or state.get("format") != "defai-udt-peer-checkpoint-v1"
                or state.get("config_sha256") != job
                or state.get("peer") != name
                or state.get("dataset_sha256") != dataset_hash
                or type(state.get("round")) is not int
                or not 0 <= state["round"] <= cfg["rounds"]
            ):
                raise ValueError("Checkpoint job/peer/data/round mismatch")
            raw_weights = state.get("weights")
            history = state.get("history")
            if (
                not isinstance(raw_weights, list)
                or any(type(value) not in (int, float) for value in raw_weights)
                or not isinstance(history, list)
                or any(not isinstance(row, dict) for row in history)
            ):
                raise ValueError("Invalid checkpoint model/history schema")
            weights = np.asarray(raw_weights, dtype=np.float64)
            if (
                weights.shape != (cfg["features"] + 1,)
                or not np.isfinite(weights).all()
                or [h.get("round") for h in history] != list(range(state["round"] + 1))
            ):
                raise ValueError("Checkpoint model/history mismatch")
            if state.get("pending") is not None:
                if state["round"] >= cfg["rounds"]:
                    raise ValueError(
                        "Completed checkpoint has an unexpected pending model"
                    )
                validate_snapshot(
                    state["pending"], job, name, state["round"] + 1, len(weights)
                )
        else:
            weights = np.zeros(cfg["features"] + 1)
            history = [{"round": 0, **metrics(weights, x, y)}]
            state = {
                "format": "defai-udt-peer-checkpoint-v1",
                "config_sha256": job,
                "peer": name,
                "dataset_sha256": dataset_hash,
                "round": 0,
                "weights": weights.tolist(),
                "history": history,
            }
        statistics = {"send_attempts": 0, "resumed_bytes": 0}
        state["status"] = "running"
        _checkpoint(checkpoint, state, key)
        receiver = Receiver(
            jar,
            output / "inbox",
            bind,
            bind_port,
            attempt_timeout,
            allow_network=allow_network,
        )
        try:
            mix = mixing_weights(cfg["peers"], name)
            for number in range(state["round"] + 1, cfg["rounds"] + 1):
                if state.get("pending") is not None:
                    local = validate_snapshot(
                        state["pending"], job, name, number, len(weights)
                    )
                else:
                    local = local_step(weights, x, y, cfg["learning_rate"])
                    state["pending"] = snapshot(job, name, number, local)
                    _checkpoint(checkpoint, state, key)
                neighbors = _exchange(
                    cfg,
                    name,
                    number,
                    "model",
                    local,
                    output,
                    key,
                    job,
                    jar,
                    receiver,
                    round_timeout,
                    attempt_timeout,
                    allow_network,
                    statistics,
                )
                values = {name: local, **neighbors}
                weights = sum(mix[peer] * values[peer] for peer in sorted(mix))
                history.append({"round": number, **metrics(weights, x, y)})
                state.update(round=number, weights=weights.tolist(), history=history)
                state.pop("pending", None)
                _checkpoint(checkpoint, state, key)
            _exchange(
                cfg,
                name,
                cfg["rounds"],
                "done",
                weights,
                output,
                key,
                job,
                jar,
                receiver,
                round_timeout,
                attempt_timeout,
                allow_network,
                statistics,
            )
            state["status"] = "completed"
            _checkpoint(checkpoint, state, key)
            # Keep an inbox available for neighbors retrying a lost final commit
            # receipt. Arbitrarily long outages still need an operator restart.
            time.sleep(linger)
            return state
        except BaseException:
            state["status"] = "paused"
            _checkpoint(checkpoint, state, key)
            raise
        finally:
            receiver.close()
            write_json(
                output / "result.json",
                {
                    **state,
                    "transport": "java-udt",
                    "central_aggregator": False,
                    "statistics_this_process": {
                        **statistics,
                        "receiver_starts": receiver.restarts,
                    },
                    "authentication": "shared-group-HMAC-SHA256",
                    "encrypted": True,
                    "cipher": "AES-256-GCM",
                },
            )


def prepare_demo(output, *, peers=3, rounds=3, topology="ring", jar=None):
    if (
        type(peers) is not int
        or not 2 <= peers <= 8
        or topology not in {"ring", "full", "star"}
    ):
        raise ValueError("Demo needs 2..8 peers and ring/full/star topology")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if jar is None:
        from defai.udt import build

        jar = build(output / "transport.jar")
    jar = Path(jar).resolve()
    key = secrets.token_bytes(32)
    key_file = output / "group.key"
    with os.fdopen(
        os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as stream:
        stream.write(key)
    names = [f"peer-{i}" for i in range(peers)]
    sockets = []
    try:
        for _ in names:
            listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
                "endpoint": f"udt://127.0.0.1:{listener.getsockname()[1]}",
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
            "transport": "udt",
            "job_id": "udt-demo",
            "features": 3,
            "rounds": rounds,
            "learning_rate": 0.2,
            "peers": entries,
            "udt_jar_sha256": sha256(jar),
            "encryption": "aes-256-gcm-v1",
            "key_sha256": hashlib.sha256(key).hexdigest(),
        },
    )
    config(manifest, transport="udt")
    rng = np.random.default_rng(42)
    for i, name in enumerate(names):
        x = rng.normal(size=(96, 3)) + (i - (peers - 1) / 2) * 0.3
        y = (
            x @ np.array([1.5, -1.0, 0.7]) + rng.normal(scale=0.2, size=96) > 0
        ).astype(float)
        np.savez(output / (name + ".npz"), x=x, y=y)
    return manifest, jar, key_file


def peer_command(
    manifest,
    name,
    data,
    output,
    jar,
    key,
    *,
    resume=False,
    round_timeout=120,
    attempt_timeout=5,
    linger=None,
):
    args = [
        sys.executable,
        "-m",
        "defai",
        "p2p",
        "udt-peer",
        str(manifest),
        "--peer",
        name,
        "--data",
        str(data),
        "--output",
        str(output),
        "--jar",
        str(jar),
        "--key-file",
        str(key),
        "--round-timeout",
        str(round_timeout),
        "--attempt-timeout",
        str(attempt_timeout),
    ]
    if resume:
        args.append("--resume")
    if linger is not None:
        args.extend(["--linger", str(linger)])
    return args


def run_demo(
    output,
    *,
    peers=3,
    rounds=3,
    topology="ring",
    jar=None,
    round_timeout=120,
    attempt_timeout=5,
):
    manifest, jar, key = prepare_demo(
        output, peers=peers, rounds=rounds, topology=topology, jar=jar
    )
    output = Path(output).resolve()
    processes, logs = [], []
    try:
        for i in range(peers):
            name = f"peer-{i}"
            log = (output / (name + ".log")).open("w", encoding="utf-8")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    peer_command(
                        manifest,
                        name,
                        output / (name + ".npz"),
                        output / name,
                        jar,
                        key,
                        round_timeout=round_timeout,
                        attempt_timeout=attempt_timeout,
                    ),
                    stdout=log,
                    stderr=log,
                )
            )
        deadline = (
            time.monotonic() + (rounds + 1) * round_timeout + 2 * attempt_timeout + 20
        )
        for process in processes:
            if process.wait(timeout=max(0.1, deadline - time.monotonic())):
                raise ValueError(
                    "UDT peer paused/failed; inspect checkpoint and per-peer logs, then resume"
                )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    results = [
        json.loads((output / f"peer-{i}" / "result.json").read_text())
        for i in range(peers)
    ]
    summary = {
        "transport": "java-udt",
        "central_aggregator": False,
        "peers": peers,
        "rounds": rounds,
        "topology": topology,
        "completed": all(r["status"] == "completed" for r in results),
        "initial_mean_local_loss": float(
            np.mean([r["history"][0]["loss"] for r in results])
        ),
        "final_mean_local_loss": float(
            np.mean([r["history"][-1]["loss"] for r in results])
        ),
    }
    write_json(output / "demo.json", summary)
    return summary
