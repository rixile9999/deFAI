from __future__ import annotations
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from flwr.client import NumPyClient
from flwr.common import ndarrays_to_parameters


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def pack(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([a.reshape(-1) for a in arrays])

def predict(arrays: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    hidden = np.maximum(x @ arrays[0] + arrays[1], 0)
    logits = hidden @ arrays[2] + arrays[3]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

def quantize(vector: np.ndarray, block_size: int = 256):
    q, scales, reconstructed = np.empty(len(vector), np.int8), [], np.empty(len(vector), np.float32)
    for start in range(0, len(vector), block_size):
        block = vector[start:start + block_size]
        scale = np.float32(max(float(np.max(np.abs(block))) / 127, 1e-12))
        codes = np.clip(np.rint(block / scale), -127, 127).astype(np.int8)
        q[start:start + len(block)] = codes
        reconstructed[start:start + len(block)] = codes.astype(np.float32) * scale
        scales.append(scale)
    return reconstructed, [q, np.asarray(scales, dtype=np.float32)]

def tensor_size(arrays: list[np.ndarray]) -> int:
    return sum(len(t) for t in ndarrays_to_parameters(arrays).tensors)

class BenchmarkClient(NumPyClient):
    def __init__(self, cid, x, y, seed):
        self.cid, self.x, self.y, self.seed = cid, x, y, seed

    def fit(self, parameters, config):
        arrays = [a.astype(np.float32).copy() for a in parameters]
        n, steps, processed = len(self.y), 0, 0
        if not n:
            raise ValueError("Empty clients must be excluded before fit")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.cid, int(config["round"]), 7727]))
        batch, epochs = int(config["batch_size"]), int(config["local_epochs"])
        target = int(config["optimizer_steps"]) if "optimizer_steps" in config else epochs * math.ceil(n / batch)
        lr, l2 = float(config["learning_rate"]), float(config["l2"])
        while steps < target:
            order = rng.permutation(n)
            for start in range(0, n, batch):
                index = order[start:start + batch]
                x, y = self.x[index], self.y[index]
                pre = x @ arrays[0] + arrays[1]
                hidden = np.maximum(pre, 0)
                logits = hidden @ arrays[2] + arrays[3]
                logits -= logits.max(axis=1, keepdims=True)
                grad_out = np.exp(logits)
                grad_out /= grad_out.sum(axis=1, keepdims=True)
                grad_out[np.arange(len(y)), y] -= 1
                grad_out /= len(y)
                grad_hidden = (grad_out @ arrays[2].T) * (pre > 0)
                grads = [x.T @ grad_hidden + l2 * arrays[0], grad_hidden.sum(0),
                         hidden.T @ grad_out + l2 * arrays[2], grad_out.sum(0)]
                arrays = [(a - lr * g).astype(np.float32) for a, g in zip(arrays, grads)]
                steps += 1
                processed += len(index)
                if steps == target:
                    break
        return arrays, n, {"optimizer_steps": steps, "examples_processed": processed}

def make_model(seed, features, classes):
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
           for v in (features, classes)):
        raise ValueError("Feature and class counts must be positive integers")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8181]))
    return [rng.normal(0, np.sqrt(2 / features), (features, 32)).astype(np.float32),
            np.zeros(32, np.float32), rng.normal(0, np.sqrt(1 / 32), (32, classes)).astype(np.float32),
            np.zeros(classes, np.float32)]

def reshape_like(vector, reference):
    arrays, offset = [], 0
    for a in reference:
        arrays.append(vector[offset:offset + a.size].reshape(a.shape).astype(np.float32).copy())
        offset += a.size
    if offset != len(vector):
        raise ValueError("Parameter vector length mismatch")
    return arrays

def source_audit(parts, masks, active):
    before = np.concatenate([parts[cid] for cid in active])
    after = np.concatenate([parts[cid][masks[cid]] for cid in active])
    before_unique, after_unique = set(before.tolist()), set(after.tolist())
    duplicates, remaining = len(before) - len(before_unique), len(after) - len(after_unique)
    removed = len(before) - len(after)
    lost = len(before_unique - after_unique)
    if lost:
        raise AssertionError("Deduplication lost unique participating records")
    return {"records_before": len(before), "records_after": len(after),
            "true_duplicate_copies": duplicates, "remaining_duplicate_copies": remaining,
            "duplicate_removed": removed, "lost_unique_records": lost,
            "dedup_recall": (duplicates - remaining) / duplicates if duplicates else None,
            "dedup_precision": 1.0 if removed else None}

@dataclass(frozen=True)
class Config:
    suite: str
    dataset: str
    seed: int
    policy: str
    candidate: str
    budget: str = "same_epochs"
    alpha: float = .3
    duplicate_rate: float = .3
    no_drift: bool = False
    missing_probability: float = 0.
    rounds: int = 20
    drift_round: int = 11
    clients: int = 12
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = .14
    l2: float = .0001

    @property
    def run_id(self):
        return (f"{self.suite}_{self.dataset}_s{self.seed}_{self.policy}_{self.candidate}_"
                f"{self.budget}_a{self.alpha:g}_d{self.duplicate_rate:g}_"
                f"n{int(self.no_drift)}_m{self.missing_probability:g}").replace(".", "p")

