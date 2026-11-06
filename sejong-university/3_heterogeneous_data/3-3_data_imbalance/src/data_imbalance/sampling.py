import numpy as np
import torch

METHODS = ("none", "smote", "tomek", "enn", "smote_tomek", "smote_enn", "combo")


def _flat(x):
    return x.reshape(len(x), -1).float()


def knn(features, k, query=None, exclude_self=True):
    query = features if query is None else query
    out = []
    for s in range(0, len(query), 1024):
        d = torch.cdist(query[s:s + 1024], features)
        if exclude_self and query is features:
            idx = torch.arange(s, min(s + 1024, len(query)))
            d[torch.arange(len(idx)), idx] = float("inf")
        out.append(torch.topk(d, min(k, len(features) - 1), largest=False).indices)
    return torch.cat(out) if out else torch.empty(0, k, dtype=torch.long)


def smote(x, y, k=5, target_ratio=1.0, seed=0):
    rng = np.random.default_rng(seed)
    counts = torch.bincount(y, minlength=int(y.max()) + 1 if len(y) else 1)
    present = [c for c in range(len(counts)) if counts[c] > 0]
    if len(present) < 2:
        return x, y
    goal = int(round(target_ratio * int(counts.max())))
    feats = _flat(x)
    new_x, new_y = [x], [y]
    for c in present:
        need = goal - int(counts[c])
        if need <= 0:
            continue
        idx = torch.nonzero(y == c).flatten()
        fc = feats[idx]
        if len(idx) == 1:
            base = fc.repeat(need, 1)
            synth = base + 0.05 * torch.randn(base.shape, generator=torch.Generator().manual_seed(seed + c))
        else:
            nn_idx = knn(fc, min(k, len(idx) - 1))
            a = torch.as_tensor(rng.integers(0, len(idx), need))
            b = nn_idx[a, torch.as_tensor(rng.integers(0, nn_idx.shape[1], need))]
            lam = torch.as_tensor(rng.random((need, 1)), dtype=torch.float32)
            synth = fc[a] + lam * (fc[b] - fc[a])
        new_x.append(synth.reshape((need,) + tuple(x.shape[1:])).to(x.dtype))
        new_y.append(torch.full((need,), c, dtype=y.dtype))
    return torch.cat(new_x), torch.cat(new_y)


def tomek_links(x, y):
    if len(torch.unique(y)) < 2 or len(y) < 3:
        return x, y
    feats = _flat(x)
    nn1 = knn(feats, 1)[:, 0]
    counts = torch.bincount(y)
    mutual = nn1[nn1] == torch.arange(len(y))
    link = mutual & (y != y[nn1])
    majority = counts[y] >= counts[y[nn1]]
    drop = link & majority
    keep = ~drop
    return x[keep], y[keep]


def enn(x, y, k=3):
    if len(torch.unique(y)) < 2 or len(y) <= k:
        return x, y
    feats = _flat(x)
    nbrs = knn(feats, k)
    votes = y[nbrs]
    agree = (votes == y[:, None]).sum(1)
    counts = torch.bincount(y, minlength=int(y.max()) + 1)
    protect = counts[y] <= k
    keep = (agree * 2 >= k) | protect
    return x[keep], y[keep]


def resample(x, y, method="combo", k=5, target_ratio=1.0, seed=0):
    if method not in METHODS:
        raise ValueError(f"unknown sampling method {method!r}; choose from {', '.join(METHODS)}")
    if method == "none":
        return x, y
    if method in ("tomek", "smote_tomek", "combo"):
        if method != "tomek":
            x, y = smote(x, y, k, target_ratio, seed)
        x, y = tomek_links(x, y)
    if method in ("enn", "smote_enn", "combo"):
        if method == "smote_enn":
            x, y = smote(x, y, k, target_ratio, seed)
        x, y = enn(x, y)
    if method == "smote":
        x, y = smote(x, y, k, target_ratio, seed)
    return x, y


def resampling_transform(method="combo", k=5, target_ratio=1.0, seed=0):
    cache = {}

    def transform(x, y, cfg, pid, config):
        key = (pid, len(y))
        if key not in cache:
            cache[key] = resample(x, y, method, k, target_ratio, seed + pid)
        return cache[key]

    return transform


def imbalance_ratio(y, num_classes=10):
    counts = np.bincount(np.asarray(y), minlength=num_classes)
    present = counts[counts > 0]
    return float(present.max() / present.min()) if len(present) else 0.0
