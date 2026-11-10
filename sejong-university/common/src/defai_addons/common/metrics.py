from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                           average: str = "macro") -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(0).astype(np.float64)
    true_pos = cm.sum(1).astype(np.float64)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = true_pos > 0
    if average == "weighted":
        w = true_pos / max(true_pos.sum(), 1)
        agg = lambda v: float((v * w).sum())
    else:
        agg = lambda v: float(v[present].mean()) if present.any() else 0.0
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision": agg(precision),
        "recall": agg(recall),
        "f1": agg(f1),
    }


def data_ratio(counts: Sequence[float]) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    return c / max(c.sum(), 1e-12)


def entropy(dist: Sequence[float], base: float = 2.0) -> float:
    p = np.asarray(dist, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(base))


def normalized_entropy(dist: Sequence[float]) -> float:
    k = len(dist)
    return entropy(dist) / np.log2(k) if k > 1 else 0.0


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


def fairness(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"mean": 0.0, "std": 0.0, "worst10": 0.0, "best10": 0.0}
    k = max(1, int(np.ceil(0.1 * v.size)))
    s = np.sort(v)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "worst10": float(s[:k].mean()), "best10": float(s[-k:].mean())}


def convergence_round(series: Sequence[float], target: Optional[float] = None,
                      frac: float = 0.95) -> Optional[int]:
    s = np.asarray(series, dtype=np.float64)
    if s.size == 0:
        return None
    thr = target if target is not None else frac * float(np.max(s))
    hit = np.where(s >= thr)[0]
    return int(hit[0]) + 1 if hit.size else None


def bootstrap_diff_ci(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_classes: int,
                      metric: str = "f1", n_boot: int = 300, seed: int = 0) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        a = classification_metrics(y_true[i], pred_a[i], n_classes)[metric]
        b = classification_metrics(y_true[i], pred_b[i], n_classes)[metric]
        diffs.append(a - b)
    d = np.asarray(diffs)
    return {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5))}


def summarize(list_of_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list_of_dicts[0].keys() if list_of_dicts else []
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}
