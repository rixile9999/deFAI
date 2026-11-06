import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, MetricRecord

from fl_core.backends import ensure_identity
from fl_core.task import test


def quiet(level=logging.WARNING):
    for name in ("flwr", "flower"):
        logging.getLogger(name).setLevel(level)
    os.environ.setdefault("FLWR_LOG_LEVEL", "WARNING")


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (np.floating,)):
        return _clean(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def history_from_result(result):
    rounds = sorted(set(result.train_metrics_clientapp) | set(result.evaluate_metrics_serverapp)
                    | set(result.evaluate_metrics_clientapp))
    rows = []
    for r in rounds:
        row = {"round": int(r)}
        for source in (result.train_metrics_clientapp, result.evaluate_metrics_clientapp,
                       result.evaluate_metrics_serverapp):
            if r in source:
                row.update({k: _clean(v) for k, v in dict(source[r]).items()})
        rows.append(row)
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def centralized_evaluator(backend, model, callback=None):
    x, y = backend.test_set()

    def evaluate_fn(server_round, arrays):
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test(model, x, y)
        record = MetricRecord({"test_loss": float(loss), "test_acc": float(acc)})
        if callback is not None:
            callback(server_round, arrays, record)
        return record

    return evaluate_fn


def run_federated(backend, strategy, num_clients, rounds, grid=None, model=None, callback=None, timeout=3600):
    quiet()
    ensure_identity()
    if model is None:
        torch.manual_seed(int(backend.cfg["seed"]))
        model = backend.build_model()
    grid = grid if grid is not None else backend.grid(num_clients)
    start = time.perf_counter()
    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()), num_rounds=rounds,
                            timeout=timeout, train_config=backend.train_config(),
                            evaluate_fn=centralized_evaluator(backend, model, callback))
    model.load_state_dict(result.arrays.to_torch_state_dict())
    history = history_from_result(result)
    return {"history": history, "model": model, "result": result, "seconds": time.perf_counter() - start}


def final_accuracy(history, tail=1, key="test_acc"):
    vals = [row[key] for row in history if row.get(key) is not None]
    return float(np.mean(vals[-tail:])) if vals else float("nan")
