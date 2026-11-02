"""Atomic numeric-only model/optimizer snapshots; never loads pickled objects."""
import hashlib
from copy import copy
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def save_checkpoint(strategy, path, *, model, overwrite=False):
    """Caller supplies the latest returned model, not the last broadcast model."""
    if not isinstance(overwrite, bool): raise ValueError("overwrite must be a boolean")
    state = strategy.state_dict()
    if not isinstance(model, dict) or set(model) != set(state["schema"]) or any(not isinstance(k, str) for k in model):
        raise ValueError("checkpoint model keys must match optimizer schema")
    arrays, descriptors = {}, {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "fiu" or not value.size or not np.isfinite(value).all():
                raise ValueError("checkpoint arrays must be finite nonempty real values")
            token = f"array_{len(arrays)}"
            arrays[token] = value
            descriptors[token] = {"shape": list(value.shape), "dtype": value.dtype.str,
                                  "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            return {"array": token}
        if isinstance(value, dict): return {key: pack(v) for key,v in value.items()}
        if isinstance(value, (tuple, list)): return [pack(v) for v in value]
        if isinstance(value, np.generic): return value.item()
        return value

    for key, array in model.items():
        if not isinstance(array, np.ndarray) or (array.shape, array.dtype.str) != state["schema"][key]:
            raise ValueError("checkpoint model shape/dtype disagrees with optimizer")
    metadata = {"format": "defai-kalman", "version": 1, "model": pack(model),
                "optimizer": pack(state), "arrays": descriptors}
    payload = json.dumps(metadata, allow_nan=False, separators=(",", ":"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".kalman-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, metadata=np.array(payload), **arrays)
            stream.flush(); os.fsync(stream.fileno())
        if overwrite: os.replace(temporary, path)
        else: os.link(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return path.resolve()


def _read_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["metadata"]
        if raw.shape != () or raw.dtype.kind != "U": raise ValueError("invalid checkpoint metadata")
        metadata = json.loads(str(raw))
        if (set(metadata) != {"format", "version", "model", "optimizer", "arrays"} or
                metadata["format"] != "defai-kalman" or type(metadata["version"]) is not int or metadata["version"] != 1):
            raise ValueError("unsupported checkpoint format/version")
        descriptors = metadata["arrays"]
        if (not isinstance(descriptors, dict) or len(archive.files) != len(descriptors)+1 or
                set(archive.files) != set(descriptors) | {"metadata"}):
            raise ValueError("checkpoint payload inventory mismatch")
        loaded = {}
        for token, descriptor in descriptors.items():
            a = archive[token]
            if (a.dtype.kind not in "fiu" or not a.size or not np.isfinite(a).all() or
                    list(a.shape) != descriptor["shape"] or a.dtype.str != descriptor["dtype"] or
                    hashlib.sha256(a.tobytes()).hexdigest() != descriptor["sha256"]):
                raise ValueError("checkpoint array integrity failure")
            loaded[token] = a

        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {"array"} and isinstance(value["array"], str):
                    return loaded[value["array"]].copy()
                return {key: unpack(v) for key,v in value.items()}
            if isinstance(value, list): return [unpack(v) for v in value]
            return value

        model, state = unpack(metadata["model"]), unpack(metadata["optimizer"])
    state["schema"] = {k: (tuple(shape), dtype) for k,(shape,dtype) in state["schema"].items()}
    if (not isinstance(model, dict) or set(model) != set(state["schema"]) or
            any(not isinstance(k, str) or not isinstance(a, np.ndarray) or a.dtype.kind not in "fiu" or
                not a.size or not np.isfinite(a).all() for k,a in model.items())):
        raise ValueError("checkpoint model schema mismatch")
    return model, state


def load_checkpoint(strategy, path):
    """Verify numeric payloads and optimizer consistency before replacing state."""
    if strategy.dp is not None:
        raise ValueError("DP training resume is unsupported by optimizer snapshots")
    try:
        model, state = _read_checkpoint(path)
    except (KeyError, TypeError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid numeric Kalman checkpoint: " + str(exc)) from exc
    # Validate on an isolated strategy shell rather than temporarily swapping
    # the running strategy's model while validation/deepcopy can still fail.
    candidate = copy(strategy)
    candidate._global = model
    candidate.load_state_dict(state)
    strategy._global = model
    for field in ("_m", "_v", "_ema", "_layers", "_last_round", "_model_schema"):
        setattr(strategy, field, getattr(candidate, field))
    strategy._round_metrics = {}
    return {key: value.copy() for key,value in model.items()}
