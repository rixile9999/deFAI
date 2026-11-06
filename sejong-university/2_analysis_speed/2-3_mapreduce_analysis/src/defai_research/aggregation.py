from __future__ import annotations
from collections import defaultdict
import math
import numpy as np


def _validate_inputs(deltas, weights, groups, block_size):
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    arrays = [np.asarray(value) for value in deltas]
    if not arrays or len(arrays) != len(weights) or len(arrays) != len(groups):
        raise ValueError("deltas, weights and groups must have equal positive lengths")
    if any(a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.number)
           or np.issubdtype(a.dtype, np.complexfloating) or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Each update must be a finite nonempty real numeric vector")
    if len({a.size for a in arrays}) != 1:
        raise ValueError("Update dimensions must match")
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in weights):
        raise ValueError("Sample counts must be positive integers")
    try:
        set(groups)
        sorted(set(groups))
    except (TypeError, ValueError) as exc:
        raise ValueError("Group labels must be mutually sortable hashable values") from exc
    return arrays


def map_reduce(deltas: list[np.ndarray], weights: list[int], groups: list[int], block_size=256):
    deltas = _validate_inputs(deltas, weights, groups, block_size)
    mapped = {}
    for delta, n, group in zip(deltas, weights, groups):
        for start in range(0, len(delta), block_size):
            key = (group, start // block_size)
            if key not in mapped:
                mapped[key] = [np.zeros_like(delta[start:start + block_size], dtype=np.float64), 0]
            mapped[key][0] += n * delta[start:start + block_size].astype(np.float64)
            mapped[key][1] += n
    group_updates, group_counts = {}, {}
    for group in sorted(set(groups)):
        blocks = [mapped[(group, b)] for b in range(math.ceil(len(deltas[0]) / block_size))]
        group_counts[group] = blocks[0][1]
        group_updates[group] = np.concatenate([s / n for s, n in blocks])
    total = sum(int(n) for n in weights)
    result = sum(group_counts[g] * group_updates[g] for g in group_updates) / total
    flat = sum(n * d.astype(np.float64) for d, n in zip(deltas, weights)) / total
    error = float(np.max(np.abs(result - flat)))
    return result, group_updates, group_counts, error

def keyed_shuffle_reduce(deltas, weights, groups, block_size, seed):
    deltas = _validate_inputs(deltas, weights, groups, block_size)
    emitted = []
    for delta, weight, group in zip(deltas, weights, groups):
        for start in range(0, len(delta), block_size):
            emitted.append(((group, start // block_size),
                            (weight * delta[start:start + block_size].astype(np.float64), int(weight))))
    rng = np.random.default_rng(seed)
    rng.shuffle(emitted)
    buckets = defaultdict(list)
    for key, value in emitted:
        buckets[key].append(value)
    reduced = {}
    for key, values in buckets.items():
        reduced[key] = (sum(value[0] for value in values), sum(value[1] for value in values))
    regional, counts = {}, {}
    for group in sorted(set(groups)):
        items = [reduced[(group, block)] for block in range(math.ceil(len(deltas[0]) / block_size))]
        assert len({item[1] for item in items}) == 1
        counts[group] = items[0][1]
        regional[group] = np.concatenate([s / n for s, n in items])
    return sum(counts[g] * regional[g] for g in regional) / sum(counts.values()), counts


def local_sufficient_statistics(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.all(np.isfinite(x)):
        raise ValueError("values must be a finite nonempty two-dimensional array")
    return int(len(x)), x.sum(axis=0), np.square(x).sum(axis=0)


def aggregate_sufficient_statistics(client_statistics, groups):
    stats = list(client_statistics)
    if not stats or len(stats) != len(groups):
        raise ValueError("Statistics and groups must have equal positive lengths")
    counts, sums, squared = [], [], []
    for item in stats:
        if len(item) != 3:
            raise ValueError("Each statistic must contain count, sum and squared sum")
        n, s, ss = item
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Statistics counts must be positive integers")
        s, ss = np.asarray(s, dtype=np.float64), np.asarray(ss, dtype=np.float64)
        if s.ndim != 1 or not s.size or s.shape != ss.shape or not np.all(np.isfinite(s)) or not np.all(np.isfinite(ss)):
            raise ValueError("Statistic vectors must have matching finite dimensions")
        if np.any(ss < 0):
            raise ValueError("Squared sums must be nonnegative")
        counts.append(int(n)); sums.append(s); squared.append(ss)
    if len({a.size for a in sums}) != 1:
        raise ValueError("Feature dimensions must match")
    regional = {}
    for group in sorted(set(groups)):
        members = [i for i, g in enumerate(groups) if g == group]
        regional[group] = (sum(counts[i] for i in members),
                           sum(sums[i] for i in members), sum(squared[i] for i in members))
    total = sum(n for n, _, _ in regional.values())
    total_sum = sum(s for _, s, _ in regional.values())
    total_squared = sum(ss for _, _, ss in regional.values())
    mean = total_sum / total
    variance = np.maximum(total_squared / total - mean * mean, 0.)
    return {"count": total, "sum": total_sum, "sum_squares": total_squared,
            "mean": mean, "population_variance": variance, "groups": regional}
