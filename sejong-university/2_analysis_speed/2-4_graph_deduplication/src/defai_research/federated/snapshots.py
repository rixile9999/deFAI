from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
import struct
import time
import numpy as np
from .graph import GraphDeduplicator


def _validate_arrays(client_X, client_y=None):
    arrays = GraphDeduplicator._validate(client_X)
    if any(a.dtype.hasobject for a in arrays):
        raise ValueError("Object arrays cannot provide portable record identities")
    if client_y is None:
        return arrays, None
    labels = [np.asarray(y) for y in client_y]
    if len(labels) != len(arrays) or any(y.ndim < 1 or len(y) != len(x) for x, y in zip(arrays, labels)):
        raise ValueError("Label arrays must match client rows")
    if any(y.dtype.hasobject for y in labels):
        raise ValueError("Object labels cannot provide portable record identities")
    if any(np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)) for y in labels):
        raise ValueError("Non-finite labels are unsupported")
    return arrays, labels


_fingerprints = GraphDeduplicator._fingerprints


def _version(digests: list[bytes], *, ordered: bool) -> bytes:
    digest = sha256()
    digest.update(struct.pack("<Q", len(digests)))
    for value in digests if ordered else sorted(digests):
        digest.update(value)
    return digest.digest()

def _checked_version(version):
    if not isinstance(version, bytes) or len(version) != 32:
        raise ValueError("Wire version must be a 32-byte digest")
    return version

def serialize_digest_packet(cid, round_id, version, rows):
    result = bytearray(struct.pack("<QQI", cid, round_id, len(rows)) + _checked_version(version))
    for index, digest in rows:
        if len(digest) != 32:
            raise ValueError("Wire record fingerprint must be 32 bytes")
        result.extend(struct.pack("<Q", index) + digest)
    return bytes(result)

def deserialize_digest_packet(data):
    if len(data) < 52:
        raise ValueError("Truncated digest packet")
    cid, rnd, count = struct.unpack_from("<QQI", data)
    if len(data) != 52 + 40 * count:
        raise ValueError("Digest packet count disagrees with payload")
    rows = [(struct.unpack_from("<Q", data, 52 + 40*i)[0], bytes(data[60 + 40*i:92 + 40*i]))
            for i in range(count)]
    return cid, rnd, bytes(data[20:52]), rows

def serialize_ownership_packet(cid, round_id, version, changes):
    return (struct.pack("<QQI", cid, round_id, len(changes)) + _checked_version(version)
            + b"".join(struct.pack("<QB", index, int(keep)) for index, keep in changes))

def deserialize_ownership_packet(data):
    if len(data) < 52:
        raise ValueError("Truncated ownership packet")
    cid, rnd, count = struct.unpack_from("<QQI", data)
    if len(data) != 52 + 9 * count:
        raise ValueError("Ownership packet count disagrees with payload")
    changes = [struct.unpack_from("<QB", data, 52 + 9*i) for i in range(count)]
    if any(keep not in (0, 1) for _, keep in changes):
        raise ValueError("Ownership keep flags must be Boolean")
    return cid, rnd, bytes(data[20:52]), [(i, bool(k)) for i, k in changes]

@dataclass(frozen=True, eq=False, slots=True)
class ImmutableTrainingSnapshot:
    x: tuple[np.ndarray, ...]
    y: tuple[np.ndarray, ...] | None
    fingerprints: tuple[tuple[bytes, ...], ...]
    ordered_versions: tuple[bytes, ...]
    content_versions: tuple[bytes, ...]
    unique_sizes: tuple[int, ...]
    construction_cpu_seconds: float
    construction_wall_seconds: float
    construction_hash_cpu_seconds: float
    copied_bytes: int
    _buffers: tuple[bytes, ...] = field(repr=False)

    @classmethod
    def from_arrays(cls, client_X, client_y=None):
        cpu, wall = time.process_time(), time.perf_counter()
        arrays, labels = _validate_arrays(client_X, client_y)
        if any(a.dtype.hasobject for a in arrays + ([] if labels is None else labels)):
            raise ValueError("Object arrays cannot provide portable immutable record identities")
        buffers = []

        def freeze(a):
            content = np.ascontiguousarray(a).tobytes()
            buffers.append(content)
            return np.frombuffer(content, dtype=a.dtype).reshape(a.shape)

        frozen_x = tuple(freeze(a) for a in arrays)
        frozen_y = None if labels is None else tuple(freeze(a) for a in labels)
        hash_started = time.process_time()
        fingerprints = tuple(tuple(_fingerprints(a, None if frozen_y is None else frozen_y[cid]))
                             for cid, a in enumerate(frozen_x))
        ordered = tuple(_version(list(h), ordered=True) for h in fingerprints)
        unordered = tuple(_version(list(h), ordered=False) for h in fingerprints)
        sizes = tuple(len(set(h)) for h in fingerprints)
        hash_cpu = time.process_time() - hash_started
        return cls(frozen_x, frozen_y, fingerprints, ordered, unordered, sizes,
                   time.process_time() - cpu, time.perf_counter() - wall, hash_cpu,
                   sum(map(len, buffers)), tuple(buffers))

    def validate_readonly(self):
        arrays = self.x + (() if self.y is None else self.y)
        if any(a.flags.writeable for a in arrays):
            raise ValueError("Immutable snapshot arrays unexpectedly became writable")
        if not self.x or any(a.ndim != 2 or a.shape[1] != self.x[0].shape[1] for a in self.x):
            raise ValueError("Immutable snapshot shape metadata is invalid")
