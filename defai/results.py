"""Run naming and result (history.json) construction for the ServerApp."""

import json
import math
import time

from flwr.app import ArrayRecord
from flwr.serverapp.strategy import Result


def run_name(cfg, timestamp=None) -> str:
    """results/ directory name: dataset, strategy, partitioner and the non-default experiment axes."""
    name = f"{cfg['dataset']}_{cfg['strategy']}_{cfg['partitioner']}"
    if cfg["dp"] != "none":
        name += f"_dp-{cfg['dp']}-{cfg['dp-side']}"
    if cfg["attack"] != "none" and cfg["num-malicious"] > 0:
        name += f"_{cfg['attack']}x{cfg['num-malicious']}"
    return f"{name}_{timestamp or time.strftime('%Y%m%d-%H%M%S')}"


def build_history(result: Result) -> list[dict]:
    """One row per round: server-side evaluation, aggregated train and client-side evaluation metrics."""
    history = []
    for r in sorted(result.evaluate_metrics_serverapp):
        row = {"round": r, **result.evaluate_metrics_serverapp[r]}
        row.update(result.train_metrics_clientapp.get(r, {}))
        row.update(result.evaluate_metrics_clientapp.get(r, {}))
        history.append(row)
    return history


def jsonable(obj):
    """Copy of ``obj`` with non-finite floats replaced by None, so the output is strict JSON.

    Python's json writes ``Infinity``/``NaN``, which other parsers reject. A DP run with
    noise multiplier 0 has an infinite epsilon, for example.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def privacy_summary(dp) -> dict:
    """Final privacy guarantee of a run. ``epsilon`` is null (and ``epsilon_unbounded`` true) when infinite."""
    epsilon = dp.epsilon
    return {
        "mechanism": dp.describe(),
        "epsilon": epsilon,
        "epsilon_unbounded": not math.isfinite(epsilon),
        "delta": dp.delta,
        "rounds_accounted": dp.accountant.num_releases,
    }


def write_summary(path, summary: dict) -> None:
    with open(path, "w") as f:
        json.dump(jsonable(summary), f, indent=2, allow_nan=False)


def final_arrays(result: Result, initial_arrays: ArrayRecord) -> ArrayRecord:
    """The trained model, or the initial one when no round produced an aggregate.

    That happens when training stops before the first round (a privacy budget below the first
    round's epsilon, ``fraction-train=0``); ``Result.arrays`` is then empty.
    """
    return result.arrays if len(result.arrays) else initial_arrays
