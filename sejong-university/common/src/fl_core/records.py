import hashlib

import numpy as np
from flwr.app import Array, ArrayRecord


def to_arrays(record):
    return {k: v.numpy() for k, v in record.items()}


def to_record(arrays):
    return ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in arrays.items()})


def record_bytes(record):
    return int(sum(len(a.data) for a in record.values()))


def message_bytes(message):
    if not message.has_content():
        return 0
    return int(sum(record_bytes(r) for r in message.content.array_records.values()))


def flatten(arrays, keys=None):
    keys = list(arrays) if keys is None else keys
    return np.concatenate([np.asarray(arrays[k], dtype=np.float64).ravel() for k in keys])


def unflatten(vector, like):
    out, pos = {}, 0
    for k, v in like.items():
        n = int(np.prod(v.shape)) if v.shape else 1
        out[k] = vector[pos:pos + n].reshape(v.shape).astype(v.dtype)
        pos += n
    return out


def delta(local, reference):
    return {k: local[k].astype(np.float64) - reference[k].astype(np.float64) for k in local}


def float_keys(arrays):
    return [k for k, v in arrays.items() if np.issubdtype(v.dtype, np.floating)]


def digest(arrays):
    h = hashlib.sha256()
    for k in arrays:
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()
