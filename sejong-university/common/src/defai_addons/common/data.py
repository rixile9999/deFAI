import gzip
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
               "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
DATA_DIR = os.environ.get("DEFAI_DATA", os.path.join(os.path.expanduser("~"), ".cache", "defai_addons"))


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(root: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = os.path.abspath(os.path.join(root or DATA_DIR, "mnist"))
    os.makedirs(root, exist_ok=True)
    arr = {}
    for k, fn in MNIST_FILES.items():
        path = os.path.join(root, fn)
        if not os.path.exists(path):
            tmp = path + ".part"
            urllib.request.urlretrieve(MNIST_URL + fn, tmp)
            os.replace(tmp, path)
        arr[k] = _read_idx(path)
    f = lambda x: (x.reshape(len(x), -1).astype(np.float32) / 255.0)
    return f(arr["train_x"]), arr["train_y"].astype(np.int64), f(arr["test_x"]), arr["test_y"].astype(np.int64)


def load_split(name: str = "mnist", seed: int = 0, max_train: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "mnist":
        Xtr, ytr, Xte, yte = load_mnist()
    else:
        X, y = load_tabular(name, seed=seed)
        Xtr, ytr, Xte, yte = train_test_split(X, y, 0.2, seed=seed)
    if max_train and max_train < len(ytr):
        idx = np.random.RandomState(seed).permutation(len(ytr))[:max_train]
        Xtr, ytr = Xtr[idx], ytr[idx]
    return Xtr, ytr, Xte, yte


def load_tabular(name: str = "digits", seed: int = 0, n_samples: int = 6000,
                 n_features: int = 40, n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if name == "mnist":
        Xtr, ytr, Xte, yte = load_mnist()
        return np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte])
    if name == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        return d.data.astype(np.float32), d.target.astype(np.int64)
    if name == "synthetic":
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_features // 2,
            n_redundant=n_features // 4, n_classes=n_classes, n_clusters_per_class=2,
            class_sep=1.2, random_state=seed)
        return X.astype(np.float32), y.astype(np.int64)
    raise ValueError("unknown dataset: %s" % name)


def inject_corruption(X: np.ndarray, missing_rate: float = 0.0, outlier_rate: float = 0.0,
                      seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    X = X.astype(np.float32).copy()
    if missing_rate > 0:
        X[rng.rand(*X.shape) < missing_rate] = np.nan
    if outlier_rate > 0:
        mask = rng.rand(*X.shape) < outlier_rate
        scale = np.nanstd(X, axis=0, keepdims=True) + 1e-6
        X[mask] = (X + 15.0 * scale * rng.choice([-1, 1], size=X.shape))[mask]
    return X


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_ratio)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def label_groups(n_classes: int, classes_per_client: int) -> List[List[int]]:
    return [list(range(s, min(s + classes_per_client, n_classes))) for s in range(0, n_classes, classes_per_client)]


def partition(y: np.ndarray, n_clients: int, scheme: str = "dirichlet", alpha: float = 0.5,
              quantity_sigma: float = 0.0, seed: int = 0, min_size: int = 10,
              classes_per_client: int = 2) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(y)
    classes = np.unique(y)
    if scheme == "iid":
        props = np.ones(n_clients)
        if quantity_sigma > 0:
            props = rng.lognormal(0.0, quantity_sigma, n_clients)
        props = props / props.sum()
        idx = rng.permutation(n)
        cuts = (np.cumsum(props) * n).astype(int)[:-1]
        parts = np.split(idx, cuts)
    elif scheme == "dirichlet":
        for _ in range(100):
            parts_l = [[] for _ in range(n_clients)]
            qty = rng.lognormal(0.0, quantity_sigma, n_clients) if quantity_sigma > 0 else np.ones(n_clients)
            for c in classes:
                idx_c = rng.permutation(np.where(y == c)[0])
                p = rng.dirichlet(np.full(n_clients, alpha)) * qty
                p = p / p.sum()
                cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx_c, cuts)):
                    parts_l[k].extend(chunk.tolist())
            if min(len(p) for p in parts_l) >= min_size:
                break
        parts = [np.array(p, dtype=np.int64) for p in parts_l]
    elif scheme == "label_groups":
        groups = label_groups(int(y.max()) + 1, classes_per_client)
        owners = [[k for k in range(n_clients) if k % len(groups) == g] for g in range(len(groups))]
        parts_l = [[] for _ in range(n_clients)]
        for g, cls in enumerate(groups):
            if not owners[g]:
                continue
            idx = rng.permutation(np.where(np.isin(y, cls))[0])
            if quantity_sigma > 0:
                p = rng.lognormal(0.0, quantity_sigma, len(owners[g]))
                cuts = (np.cumsum(p / p.sum()) * len(idx)).astype(int)[:-1]
                chunks = np.split(idx, cuts)
            else:
                chunks = np.array_split(idx, len(owners[g]))
            for k, ch in zip(owners[g], chunks):
                parts_l[k].extend(ch.tolist())
        parts = [np.array(p, dtype=np.int64) for p in parts_l]
    elif scheme == "shard":
        shards_per = 2
        order = np.argsort(y, kind="stable")
        shards = np.array_split(order, n_clients * shards_per)
        perm = rng.permutation(len(shards))
        parts = [np.concatenate([shards[j] for j in perm[k * shards_per:(k + 1) * shards_per]])
                 for k in range(n_clients)]
    else:
        raise ValueError("unknown scheme: %s" % scheme)
    return [rng.permutation(p) for p in parts]


def split_clients(X: np.ndarray, y: np.ndarray, parts: List[np.ndarray], val_ratio: float = 0.2,
                  seed: int = 0) -> List[Dict[str, np.ndarray]]:
    rng = np.random.RandomState(seed)
    out = []
    for p in parts:
        p = rng.permutation(p)
        n_val = max(1, int(len(p) * val_ratio))
        out.append({"X": X[p[n_val:]], "y": y[p[n_val:]], "Xv": X[p[:n_val]], "yv": y[p[:n_val]]})
    return out


def class_distribution(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y, minlength=n_classes).astype(np.float64)


def describe_partition(parts: List[np.ndarray], y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    n_classes = n_classes or int(y.max()) + 1
    return np.stack([class_distribution(y[p], n_classes) for p in parts])
