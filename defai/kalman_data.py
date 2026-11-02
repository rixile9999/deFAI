"""Offline public-data loaders and disjoint, train-normalized FL partitions.

UCI datasets are loaded from scikit-learn, not redistributed by this package.
These are optimizer benchmarks, not clinical validation or institution splits.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np


SOURCES = {
    "digits": {"url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
               "authors": "E. Alpaydin and C. Kaynak", "license": "CC BY 4.0",
               "subset": "scikit-learn 1797-row copy of the UCI test subset"},
    "breast-cancer": {"url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                       "authors": "W. Wolberg, O. Mangasarian and W. Street", "license": "CC BY 4.0",
                       "subset": "scikit-learn 569-row diagnostic dataset"},
}


@dataclass(frozen=True)
class PublicData:
    name: str
    x: np.ndarray
    y: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    clients: dict
    metadata: dict


def prepare_public_data(name, *, seed=0, alpha=.5, clients=10):
    """Fit normalization on training rows only; never download a dataset."""
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.model_selection import train_test_split
    if name not in SOURCES:
        raise ValueError("dataset must be digits or breast-cancer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if isinstance(clients, bool) or not isinstance(clients, int) or clients < 2 or clients > 20:
        raise ValueError("clients must be an integer in [2, 20]")
    raw = (load_digits if name == "digits" else load_breast_cancer)()
    x, y = np.asarray(raw.data, dtype=np.float64), np.asarray(raw.target, dtype=np.int64)
    digest = hashlib.sha256(x.astype("<f8").tobytes()+y.astype("<i8").tobytes()).hexdigest()
    remain, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed, stratify=y)
    train, validation = train_test_split(remain, test_size=.2, random_state=seed+1, stratify=y[remain])
    mean, std = x[train].mean(axis=0), x[train].std(axis=0)
    std[std == 0] = 1
    normalized = (x-mean)/std
    split_hash = hashlib.sha256(b"".join(np.asarray(a, dtype="<i8").tobytes() for a in (train, validation, test))).hexdigest()
    groups, repairs = partition_splits(y, {"train": train, "validation": validation, "test": test},
                                      alpha=alpha, seed=seed+2, clients=clients)
    metadata = {"source": dict(SOURCES[name]), "dataset_sha256": digest, "split_sha256": split_hash,
                "seed": seed, "alpha": alpha, "samples": len(y), "features": x.shape[1], "classes": len(raw.target_names),
                "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
                "normalization": "mean/std fitted on training rows only", "empty_client_repairs": repairs,
                "client_sizes": {key: [len(a) for a in value] for key, value in groups.items()},
                "train_mean": mean.tolist(), "train_std": std.tolist()}
    for a in (normalized, y, train, validation, test, *(a for group in groups.values() for a in group)):
        a.setflags(write=False)
    return PublicData(name, normalized, y, train, validation, test, groups, metadata)


def partition_splits(y, splits, *, alpha, seed, clients):
    """Share class-allocation probabilities across disjoint splits.

    Empty clients receive one row from the largest client, within that split;
    repairs are counted because they change the ideal Dirichlet allocation.
    """
    rng = np.random.default_rng(seed)
    probabilities = {label: rng.dirichlet(np.full(clients, alpha)) for label in np.unique(y)}
    output, repairs = {}, {}
    for key, indices in splits.items():
        nodes = [[] for _ in range(clients)]
        for label, p in probabilities.items():
            rows = rng.permutation(indices[y[indices] == label])
            sizes = rng.multinomial(len(rows), p)
            for node, rows_for_node in zip(nodes, np.split(rows, np.cumsum(sizes)[:-1])):
                node.extend(rows_for_node.tolist())
        repaired = 0
        for node in nodes:
            if not node:
                donor = max(nodes, key=len)
                if len(donor) < 2: raise ValueError("not enough rows for nonempty clients")
                node.append(donor.pop()); repaired += 1
        output[key] = [np.asarray(node, dtype=np.int64) for node in nodes]
        repairs[key] = repaired
    return output, repairs
