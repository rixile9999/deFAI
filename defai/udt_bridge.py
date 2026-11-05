"""Java UDT process lifecycle and bounded retries; no HTTP transport fallback."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@contextmanager
def peer_lock(path):
    """OS-owned lock, automatically released when the Python process dies."""
    with Path(path).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("This peer workspace is already active") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Receiver:
    """Restarts a failed Java inbox and leases it to the owning Python worker."""

    def __init__(self, jar, store, bind, port, attempt_timeout, *, allow_network=False):
        self.jar, self.store = str(jar), Path(store)
        self.bind, self.port = bind, port
        self.timeout = attempt_timeout
        self.allow_network = allow_network
        self.store.mkdir(parents=True, exist_ok=True)
        self.lease = self.store.parent / ("receiver-" + uuid.uuid4().hex + ".lease")
        self.lease.touch()
        self.log = (self.store.parent / "receiver.log").open("ab", buffering=0)
        self.stop = threading.Event()
        self.process = None
        self.restarts = 0
        self.thread = threading.Thread(target=self._supervise, daemon=True)
        self.thread.start()

    def _supervise(self):
        while not self.stop.is_set():
            self.lease.touch()
            if self.process is None or self.process.poll() is not None:
                args = [
                    "java",
                    "-jar",
                    self.jar,
                    "serve",
                    str(self.store),
                    self.bind,
                    str(self.port),
                    str(int(self.timeout * 1000)),
                    "--lease",
                    str(self.lease),
                ]
                if self.allow_network:
                    args.append("--allow-insecure-network")
                self.process = subprocess.Popen(args, stdout=self.log, stderr=self.log)
                self.restarts += 1
            self.stop.wait(0.25)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.lease.unlink(missing_ok=True)


def deliver(
    jar,
    source,
    transfer_id,
    version,
    host,
    port,
    deadline,
    *,
    attempt_timeout=5,
    allow_network=False,
):
    attempts, backoff = 0, 0.1
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining < 1:
            break
        timeout = min(attempt_timeout, remaining)
        args = [
            "java",
            "-jar",
            str(jar),
            "send",
            str(source),
            transfer_id,
            version,
            host,
            str(port),
            str(int(timeout * 1000)),
        ]
        if allow_network:
            args.append("--allow-insecure-network")
        attempts += 1
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout + 2,
            )
        except subprocess.TimeoutExpired:
            result = None
        if (
            result is not None
            and result.returncode == 0
            and "COMMITTED resumed_bytes=" in result.stdout
        ):
            return {
                "attempts": attempts,
                "resumed_bytes": int(result.stdout.split("resumed_bytes=")[-1].strip()),
            }
        time.sleep(min(backoff, max(0, deadline - time.monotonic())))
        backoff = min(1, backoff * 2)
    raise TimeoutError(f"UDT delivery deadline: {transfer_id} ({attempts} attempts)")
