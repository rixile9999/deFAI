"""Aggregation rules over client updates, in plain numpy.

An "update" is a dict mapping parameter name -> ndarray (usually ``local - global``).
Nothing here depends on Flower, so every rule can be unit-tested in isolation.

References
----------
- Weighted mean:    McMahan et al., 2017 (FedAvg)
- Median/trimmed:   Yin et al., 2018 "Byzantine-Robust Distributed Learning"
- (Multi-)Krum:     Blanchard et al., 2017 "Machine Learning with Adversaries"
- Bulyan:           El Mhamdi et al., 2018 "The Hidden Vulnerability of Distributed Learning in Byzantium"
"""

from collections.abc import Sequence

import numpy as np

Arrays = dict[str, np.ndarray]


def l2_norm(arrays: Arrays) -> float:
    """L2 norm of all arrays viewed as one flat vector."""
    return float(np.sqrt(sum(float(np.sum(np.square(a, dtype=np.float64))) for a in arrays.values())))


def flatten(arrays: Arrays) -> np.ndarray:
    """Concatenate all arrays (in key order) into one float64 vector."""
    return np.concatenate([a.ravel().astype(np.float64) for a in arrays.values()])


def _stack(updates: Sequence[Arrays], key: str) -> np.ndarray:
    return np.stack([u[key] for u in updates]).astype(np.float64)


def cast_like(values: np.ndarray, dtype) -> np.ndarray:
    """Cast float64 results back to ``dtype``; integers are rounded, not truncated."""
    values = np.asarray(values)
    if np.issubdtype(dtype, np.integer):
        values = np.rint(values)
    return values.astype(dtype, copy=False)


def _like(result: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cast an aggregate back to the dtype of the arrays it was computed from."""
    return cast_like(result, reference.dtype)


def weighted_mean(updates: Sequence[Arrays], weights: Sequence[float]) -> Arrays:
    """sum_i (w_i / sum_j w_j) * update_i"""
    w = np.asarray(weights, dtype=np.float64)
    if not len(updates) or w.ndim != 1 or len(w) != len(updates):
        raise ValueError("one weight per update is required")
    if not np.isfinite(w).all() or not np.all(w >= 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    w = w / w.sum()
    keys = set(updates[0])
    if not keys or any(set(update) != keys for update in updates):
        raise ValueError("updates must have the same nonempty parameter keys")
    result = {}
    for key, reference in updates[0].items():
        total = np.zeros(reference.shape, dtype=np.float64)
        scratch = np.empty_like(total)
        for weight, update in zip(w, updates):
            value = np.asarray(update[key])
            if value.shape != reference.shape or not np.isfinite(value).all():
                raise ValueError("client arrays must be finite and have matching shapes")
            np.multiply(value, weight, out=scratch, dtype=np.float64)
            total += scratch
        result[key] = _like(total, reference)
    return result


def coordinate_median(updates: Sequence[Arrays]) -> Arrays:
    """Coordinate-wise median (FedMedian)."""
    return {k: _like(np.median(_stack(updates, k), axis=0), updates[0][k]) for k in updates[0]}


def trimmed_mean(updates: Sequence[Arrays], trim_ratio: float) -> Arrays:
    """Coordinate-wise trimmed mean (FedTrimmedAvg).

    For every coordinate, drop the ``floor(trim_ratio * n)`` smallest and the same
    number of largest values, then average the rest.
    """
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    n = len(updates)
    k = int(trim_ratio * n)
    out = {}
    for key in updates[0]:
        stacked = np.sort(_stack(updates, key), axis=0)
        out[key] = _like(stacked[k : n - k].mean(axis=0), updates[0][key])
    return out


def pairwise_sq_distances(updates: Sequence[Arrays]) -> np.ndarray:
    """(n, n) matrix of squared L2 distances between flattened updates."""
    flat = np.stack([flatten(u) for u in updates])
    sq = np.einsum("ij,ij->i", flat, flat)
    dist = sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T
    np.fill_diagonal(dist, 0.0)
    return np.maximum(dist, 0.0)  # clamp tiny negatives from cancellation


def krum_scores(distances: np.ndarray, num_byzantine: int) -> np.ndarray:
    """Krum score per client: sum of squared distances to its n - f - 2 nearest peers."""
    n = distances.shape[0]
    num_closest = max(1, n - num_byzantine - 2)
    masked = distances.astype(np.float64).copy()
    np.fill_diagonal(masked, np.inf)  # a client is not its own neighbour
    nearest = np.sort(masked, axis=1)[:, : min(num_closest, n - 1)]
    return nearest.sum(axis=1)


def multi_krum_select(
    updates: Sequence[Arrays], num_byzantine: int, num_select: int
) -> list[int]:
    """Indices of the ``num_select`` lowest-scoring clients (1 -> classic Krum)."""
    n = len(updates)
    if n == 1:
        return [0]
    scores = krum_scores(pairwise_sq_distances(updates), num_byzantine)
    return [int(i) for i in np.argsort(scores, kind="stable")[: min(num_select, n)]]


def bulyan_select(distances: np.ndarray, num_byzantine: int, num_select: int) -> list[int]:
    """Bulyan's selection phase: apply Krum repeatedly, each time on the clients not yet chosen."""
    remaining = list(range(distances.shape[0]))
    chosen: list[int] = []
    while remaining and len(chosen) < num_select:
        scores = krum_scores(distances[np.ix_(remaining, remaining)], num_byzantine)
        chosen.append(remaining.pop(int(np.argmin(scores))))
    return chosen


def closest_to_median_mean(updates: Sequence[Arrays], beta: int) -> Arrays:
    """Per coordinate, average the ``beta`` values closest to the coordinate-wise median."""
    n = len(updates)
    if not 1 <= beta <= n:
        raise ValueError("beta must be between 1 and the number of updates")
    out = {}
    for key in updates[0]:
        stacked = _stack(updates, key)
        distance_to_median = np.abs(stacked - np.median(stacked, axis=0))
        closest = np.argpartition(distance_to_median, beta - 1, axis=0)[:beta]
        out[key] = _like(np.take_along_axis(stacked, closest, axis=0).mean(axis=0), updates[0][key])
    return out
