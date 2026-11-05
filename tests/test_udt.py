"""Actual Java/UDP processes, with a bidirectional datagram fault relay."""

import heapq
import importlib.util
from pathlib import Path
import selectors
import shutil
import socket
import subprocess
import threading
import time

import pytest


@pytest.fixture(scope="module")
def udt_jar(tmp_path_factory):
    if not shutil.which("javac") or not shutil.which("java"):
        pytest.skip("A JDK is required for actual Java/UDP tests")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_udt", root / "scripts/build_udt.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder.build(tmp_path_factory.mktemp("java") / "defai-udt.jar")


def launch(jar, *args):
    return subprocess.Popen(
        ["java", "-jar", jar, *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def receive(jar, root, timeout=12000):
    process = launch(jar, "receive", root, "127.0.0.1", 0, timeout)
    line = process.stdout.readline()
    assert line.startswith("READY "), (line, process.communicate(timeout=3))
    return process, int(line.split()[1])


def finish(process, success=True, timeout=20):
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        process.kill()
        process.communicate()
        raise
    assert (process.returncode == 0) == success, stdout + "\n" + stderr
    return stdout, stderr


class Relay:
    def __init__(self, server_port, *, loss_every=0, delay=0, drop_handshake=False):
        self.front, self.back = (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        )
        self.front.bind(("127.0.0.1", 0))
        self.back.bind(("127.0.0.1", 0))
        self.port, self.server = self.front.getsockname()[1], ("127.0.0.1", server_port)
        self.client, self.stop = None, threading.Event()
        self.backends = {self.back: None}
        self.blackout = threading.Event()
        self.jitter, self.duplicate_every = 0, 0
        self.loss_every, self.delay, self.drop_handshake = (
            loss_every,
            delay,
            drop_handshake,
        )
        self.dropped = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        selector = selectors.DefaultSelector()
        selector.register(self.front, selectors.EVENT_READ)
        selector.register(self.back, selectors.EVENT_READ)
        client_sockets = {}
        queue, count, order = [], 0, 0
        try:
            while not self.stop.is_set():
                for key, _ in selector.select(0.002):
                    try:
                        data, address = key.fileobj.recvfrom(65535)
                    except ConnectionResetError:
                        continue  # Windows reports ICMP from a peer that already exited.
                    if self.blackout.is_set():
                        self.dropped += 1
                        continue
                    if key.fileobj is self.front:
                        self.client = address
                        if address not in client_sockets:
                            if not client_sockets:
                                output = self.back
                            else:
                                output = socket.socket(
                                    socket.AF_INET, socket.SOCK_DGRAM
                                )
                                output.bind(("127.0.0.1", 0))
                                selector.register(output, selectors.EVENT_READ)
                            client_sockets[address] = output
                            self.backends[output] = address
                        output, destination = client_sockets[address], self.server
                    else:
                        output, destination = self.front, self.backends[key.fileobj]
                    if destination is None:
                        continue
                    if (
                        self.drop_handshake
                        and key.fileobj is self.back
                        and data[0] == 128
                        and data[1] == 0
                    ):
                        self.drop_handshake = False
                        self.dropped += 1
                        continue
                    if not data[0] & 128:
                        count += 1
                        if self.loss_every and count % self.loss_every == 0:
                            self.dropped += 1
                            continue
                    order += 1
                    heapq.heappush(
                        queue,
                        (
                            time.monotonic() + self.delay + self.jitter * (order % 3),
                            order,
                            output,
                            destination,
                            data,
                        ),
                    )
                    if self.duplicate_every and order % self.duplicate_every == 0:
                        order += 1
                        heapq.heappush(
                            queue,
                            (
                                time.monotonic() + self.delay + 0.001,
                                order,
                                output,
                                destination,
                                data,
                            ),
                        )
                while queue and queue[0][0] <= time.monotonic():
                    _, _, output, destination, data = heapq.heappop(queue)
                    if self.blackout.is_set():
                        self.dropped += 1
                    else:
                        output.sendto(data, destination)
        finally:
            selector.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.front.close()
        for backend in self.backends:
            backend.close()


@pytest.mark.parametrize(
    "loss,delay,handshake", [(0, 0, False), (11, 0.005, False), (0, 0, True)]
)
def test_actual_transfer_and_duplicate_receipt(
    udt_jar, tmp_path, loss, delay, handshake
):
    source = tmp_path / "source.bin"
    payload = bytes(range(256)) * 1024
    source.write_bytes(payload)
    server, port = receive(udt_jar, tmp_path / "store", 30000)
    relay = Relay(port, loss_every=loss, delay=delay, drop_handshake=handshake)
    try:
        client = launch(
            udt_jar,
            "send",
            source,
            "transfer-1",
            "round-2",
            "127.0.0.1",
            relay.port,
            30000,
        )
        sent, _ = finish(client, timeout=35)
        finish(server, timeout=35)
        assert "resumed_bytes=0" in sent
        assert (tmp_path / "store/transfer-1/payload.bin").read_bytes() == payload
        if loss or handshake:
            assert relay.dropped > 0
    finally:
        relay.close()
        if server.poll() is None:
            server.kill()
            server.communicate()
    server, port = receive(udt_jar, tmp_path / "store")
    client = launch(
        udt_jar, "send", source, "transfer-1", "round-2", "127.0.0.1", port, 10000
    )
    sent, _ = finish(client)
    finish(server)
    assert f"resumed_bytes={len(payload)}" in sent


def test_process_death_resumes_durable_chunks(udt_jar, tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 16384)
    server, port = receive(udt_jar, tmp_path / "store", 30000)
    relay = Relay(port, delay=0.015)
    client = launch(
        udt_jar, "send", source, "resume", "round-3", "127.0.0.1", relay.port, 30000
    )
    part = tmp_path / "store/resume/payload.part"
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and (
            not part.exists() or part.stat().st_size < 65536
        ):
            time.sleep(0.01)
        assert part.exists() and part.stat().st_size >= 65536
    finally:
        client.kill()
        client.communicate()
        server.kill()
        server.communicate()
        relay.close()
    assert not (part.parent / "payload.bin").exists()
    # A resumed connection can itself fail. Exercise the production retry/
    # receiver lifecycle instead of assuming one standalone CLI attempt succeeds.
    from defai.udt_bridge import Receiver, deliver

    reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
    reservation.close()
    receiver = Receiver(udt_jar, tmp_path / "store", "127.0.0.1", port, 5)
    try:
        receipt = deliver(
            udt_jar,
            source,
            "resume",
            "round-3",
            "127.0.0.1",
            port,
            time.monotonic() + 60,
            attempt_timeout=5,
        )
    finally:
        receiver.close()
    assert receipt["resumed_bytes"] >= 65536
    assert (part.parent / "payload.bin").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("mode", ["receive", "send"])
def test_deadline_exits_without_live_worker_leak(udt_jar, tmp_path, mode):
    if mode == "receive":
        process, _ = receive(udt_jar, tmp_path / "store", 500)
    else:
        source = tmp_path / "empty.bin"
        source.write_bytes(b"")
        process = launch(
            udt_jar, "send", source, "timeout", "round-1", "127.0.0.1", 1, 500
        )
    finish(process, success=False, timeout=5)


def test_adversarial_stream_cases(udt_jar, tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "java/udt/src/test/java/org/defai/udt/TransferChecks.java"
    )
    subprocess.run(
        [
            "javac",
            "--release",
            "8",
            "-encoding",
            "UTF-8",
            "-cp",
            udt_jar,
            "-d",
            str(tmp_path),
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    import os

    result = subprocess.run(
        [
            "java",
            "-cp",
            os.pathsep.join([udt_jar, str(tmp_path)]),
            "org.defai.udt.TransferChecks",
            str(tmp_path / "store"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "14 checks passed"
