"""Explicit alignment, with no silent dropping of duplicate or missing identities."""
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np


def checked_ids(values):
    values = tuple(values)
    if not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError("IDs must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate IDs are ambiguous")
    return values


@dataclass(frozen=True)
class AlignedData:
    features: dict
    labels: np.ndarray | None
    label_owner: str | None
    alignment: str

    @property
    def size(self):
        return len(next(iter(self.features.values())))


def align_parties(parties: Mapping, *, labels=None, label_owner=None, mode="plain"):
    """parties[name]=(local IDs, local feature array); labels follow owner's IDs.

    This function orchestrates a LOCAL simulation and can access all input arrays.
    Plain mode exposes identifiers. dh-psi only hides IDs in its protocol messages.
    """
    if not isinstance(parties, Mapping) or len(parties) < 2:
        raise ValueError("At least two named feature parties are required")
    ids, arrays = {}, {}
    for name, (keys, values) in parties.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Party names must be nonempty strings")
        ids[name] = checked_ids(keys)
        a = np.asarray(values)
        if a.ndim < 2 or len(a) != len(ids[name]) or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
            raise ValueError("Feature arrays must be finite, numeric and aligned with local IDs")
        arrays[name] = a
    if mode == "plain":
        common = sorted(set.intersection(*(set(v) for v in ids.values())))
        rows = {k: np.array([dict(zip(v, range(len(v))))[x] for x in common], dtype=int) for k, v in ids.items()}
    elif mode == "dh-psi":
        from .psi import intersection_rows
        rows = intersection_rows(ids)
    else:
        raise ValueError("Alignment must be plain or dh-psi")
    if not len(next(iter(rows.values()))):
        raise ValueError("No common samples")
    if (labels is None) != (label_owner is None):
        raise ValueError("Labels and their owner must be provided together")
    target = None
    if labels is not None:
        if label_owner not in ids:
            raise ValueError("Unknown label owner")
        target = np.asarray(labels)
        if target.ndim not in (1, 2) or len(target) != len(ids[label_owner]) or not np.isfinite(target).all():
            raise ValueError("Labels must be finite and aligned with label-owner IDs")
        target = target[rows[label_owner]].copy()
    return AlignedData({k: a[rows[k]].copy() for k, a in arrays.items()}, target, label_owner, mode)


def align_time_series(series, grid):
    """Resample local [time,features] streams on an explicit shared grid.

    No extrapolation or automatic future filling; interpolation is an offline
    preprocessing step, not safe for forecasting before both endpoints are known.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Grid must be finite and strictly increasing")
    result = {}
    for name, (times, values) in series.items():
        times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
        if times.ndim != 1 or len(times) < 2 or values.ndim != 2 or len(times) != len(values):
            raise ValueError("Expected times[t] and values[t,features]")
        if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Local timestamps must be finite and strictly increasing")
        if grid[0] < times[0] or grid[-1] > times[-1]:
            raise ValueError("Resampling would extrapolate beyond local observations")
        result[name] = np.column_stack([np.interp(grid, times, v) for v in values.T])
    if len(result) < 2:
        raise ValueError("At least two streams are required")
    return result
