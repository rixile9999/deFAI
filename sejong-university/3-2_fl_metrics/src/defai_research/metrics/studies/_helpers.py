from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from defai_research.federated.learning import (
    BenchmarkClient, make_model, reshape_like, pack, predict, json_write, csv_write,
)
from defai_research.federated.graph import GraphDeduplicator
from defai_research.metrics import classification_metrics, information_metrics

SHAPES = ((64, 32), (32,), (32, 10), (10,))
DigitsClient = BenchmarkClient


def initial_parameters(seed: int) -> list[np.ndarray]:
    return make_model(seed, 64, 10)


def unpack(vector: np.ndarray) -> list[np.ndarray]:
    return reshape_like(vector, [np.empty(shape, np.float32) for shape in SHAPES])


def metric_bundle(y, p):
    result = classification_metrics(y, p)
    pred, confidence = p.argmax(1), p.max(1)
    ece = 0.
    for lower in np.linspace(0, 1, 16)[:-1]:
        upper = lower + 1/15
        mask = (confidence >= lower) & ((confidence < upper) if upper < .999999 else (confidence <= 1))
        if mask.any():
            ece += mask.mean()*abs(float(np.mean(pred[mask] == y[mask]))-float(confidence[mask].mean()))
    info = information_metrics(y, p)
    hy, mi = info["data_entropy_bits"], info["mutual_information_bits"]
    return dict(accuracy=result["accuracy"], macro_precision=result["macro_precision"],
                macro_recall=result["macro_recall"], macro_f1=result["macro_f1"],
                log_loss=float(log_loss(y, p, labels=np.arange(p.shape[1]))), ece_15=float(ece),
                label_entropy_bits=hy, prediction_entropy_bits=info["prediction_entropy_bits"],
                joint_label_prediction_entropy_bits=info["joint_entropy_bits"],
                label_prediction_mi_bits=mi, normalized_label_prediction_mi=mi/hy if hy else 0.,
                conditional_label_entropy_bits=hy-mi)


def prepare_data(dataset, seed, alpha, duplicate_rate, clients=12):
    source = load_digits() if dataset == "digits" else load_breast_cancer()
    x, y = source.data.astype(np.float32), source.target.astype(np.int64)
    ids = np.arange(len(y))
    train, other = train_test_split(ids, train_size=.6, stratify=y, random_state=seed)
    val, test = train_test_split(other, test_size=.5, stratify=y[other], random_state=seed + 1)
    if dataset == "digits":
        x /= 16
        preprocessing = {"kind": "fixed_public_scale", "divisor": 16}
    else:
        mean, scale = x[train].mean(0), np.maximum(x[train].std(0), 1e-6)
        x = np.clip((x - mean) / scale, -5, 5).astype(np.float32)
        preprocessing = {"kind": "train_only_zscore_then_clip", "mean": mean.tolist(),
                         "scale": scale.tolist(), "clip": [-5, 5], "fit_source_ids": train.tolist()}
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise AssertionError("Source split leakage")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 211]))
    shuffled = rng.permutation(train)
    floor = 3
    base = [shuffled[cid * floor:(cid + 1) * floor].tolist() for cid in range(clients)]
    rest = shuffled[clients * floor:]
    for label in range(len(np.unique(y))):
        selected = rng.permutation(rest[y[rest] == label])
        allocation = rng.multinomial(len(selected), rng.dirichlet(np.full(clients, alpha)))
        offset = 0
        for cid, n in enumerate(allocation):
            base[cid].extend(selected[offset:offset + n].tolist())
            offset += n
    base = [np.asarray(sorted(a), np.int64) for a in base]
    migrated = [[] for _ in range(clients)]
    drift_rng = np.random.default_rng(np.random.SeedSequence([seed, 321]))
    for cid, a in enumerate(base):
        shuffled = drift_rng.permutation(a)
        n = max(1, int(.2 * len(a)))
        migrated[cid].extend(shuffled[n:].tolist())
        migrated[(cid + 4) % clients].extend(shuffled[:n].tolist())
    migrated = [np.asarray(sorted(a), np.int64) for a in migrated]

    def duplicate(parts, direction, tag):
        rows = [a.tolist() for a in parts]
        generator = np.random.default_rng(np.random.SeedSequence([seed, tag]))
        for cid, a in enumerate(parts):
            n = min(len(a) - 1, int(round(duplicate_rate * len(a))))
            if n:
                rows[(cid + direction) % clients].extend(generator.choice(a, n, replace=False).tolist())
        return [np.asarray(a, np.int64) for a in rows]

    phases = {"initial": duplicate(base, 1, 431), "drift": duplicate(migrated, 5, 432)}
    fingerprints = GraphDeduplicator._fingerprints(x[train], y[train])
    if len(set(fingerprints)) != len(fingerprints):
        raise AssertionError("Equal-content source IDs require a group-disjoint split protocol")
    counts = [np.bincount(y[a], minlength=len(np.unique(y))) / len(a) for a in base]
    entropy = [-float(np.sum(p[p > 0] * np.log2(p[p > 0]))) for p in counts]
    audit = {"dataset": dataset, "seed": seed, "alpha": alpha, "duplicate_rate": duplicate_rate,
             "train_ids": train.tolist(), "validation_ids": val.tolist(), "test_ids": test.tolist(),
             "initial_unique_source_ids": [a.tolist() for a in base],
             "drift_unique_source_ids": [a.tolist() for a in migrated],
             "phase_source_ids": {key: [a.tolist() for a in value] for key, value in phases.items()},
             "preprocessing": preprocessing, "minimum_preallocation_records_per_client": floor,
             "mean_client_label_entropy_bits": float(np.mean(entropy)),
             "min_client_unique_records": min(map(len, base)), "max_client_unique_records": max(map(len, base))}
    digest = hashlib.sha256(x.tobytes() + y.tobytes())
    digest.update(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode())
    return {"x": x, "y": y, "train": train, "val": val, "test": test, "phases": phases,
            "classes": len(np.unique(y)), "audit": audit, "scenario_sha256": digest.hexdigest()}
