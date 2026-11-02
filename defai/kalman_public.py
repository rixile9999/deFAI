"""Paired public-data FedAdam benchmark with held-out test evaluation only at end."""
import hashlib
from itertools import product
import math
import time

import numpy as np

from defai.kalman import KalmanConfig, stable_l2
from defai.kalman_benchmark import METHODS, paired_runs, summarize
from defai.kalman_data import prepare_public_data


def log_probabilities(x, w, bias):
    logits = x @ w + bias
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def evaluate(data, arrays, split):
    indices = getattr(data, split)
    logp = log_probabilities(data.x[indices], arrays["w"], arrays["bias"])
    predictions = logp.argmax(axis=1)
    correct = predictions == data.y[indices]
    lookup = dict(zip(indices.tolist(), correct.tolist()))
    client_acc = [np.mean([lookup[int(i)] for i in node]) for node in data.clients[split]]
    return {"accuracy": float(correct.mean()), "loss": float(-logp[np.arange(len(indices)), data.y[indices]].mean()),
            "client_accuracy_p10": float(np.quantile(client_acc, .1))}


def run_public_case(data, *, method, local_steps=5, participation=.5, rounds=60):
    """No test score is computed until training and target tracking finish."""
    from defai.strategies import FedAdam, FedAdamKalman
    if method not in METHODS:
        raise ValueError("unknown public benchmark method")
    if (isinstance(local_steps, bool) or not isinstance(local_steps, int) or local_steps < 1 or
            isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 3 or
            isinstance(participation, bool) or not isinstance(participation, (int, float)) or
            not math.isfinite(participation) or not 0 < participation <= 1):
        raise ValueError("invalid public benchmark condition")
    strategy = (FedAdam(server_learning_rate=.03) if method == "ema" else
                FedAdamKalman(server_learning_rate=.03, kalman=KalmanConfig(
                    process_adaptation=.2 if method == "adaptive-kalman" else 0)))
    strategy._global = {"w": np.zeros((data.x.shape[1], data.metadata["classes"])),
                        "bias": np.zeros(data.metadata["classes"])}
    local_data = [(data.x[node], data.y[node]) for node in data.clients["train"]]
    rng = np.random.default_rng(data.metadata["seed"]+9001)
    target = .85 if data.name == "digits" else .95
    history, schedule = [], []
    first = sustained = None
    streak = 0
    previous_move = {k: np.zeros_like(v) for k, v in strategy._global.items()}
    start = time.perf_counter()
    for rnd in range(1, rounds+1):
        selected = rng.choice(len(local_data), max(1, int(participation*len(local_data))), replace=False)
        schedule.append(selected.tolist())
        global_arrays = strategy._global
        updates, counts = [], []
        for client in selected:
            x, y = local_data[client]
            w, bias = global_arrays["w"].copy(), global_arrays["bias"].copy()
            for _ in range(local_steps):
                error = np.exp(log_probabilities(x, w, bias))
                error[np.arange(len(y)), y] -= 1
                w -= .05 * (x.T @ error/len(y) + .001*w)
                bias -= .05 * error.mean(axis=0)
            updates.append({"w": w-global_arrays["w"], "bias": bias-global_arrays["bias"]})
            counts.append(len(y))
        weights = np.asarray(counts, dtype=float)/sum(counts)
        delta = {k: sum(p*u[k] for p,u in zip(weights, updates)) for k in global_arrays}
        before = time.perf_counter()
        output = strategy.update_global(rnd, delta)
        server_seconds = time.perf_counter()-before
        move = {k: output[k]-global_arrays[k] for k in output}
        strategy._global = output
        metrics = evaluate(data, output, "validation")
        elapsed = time.perf_counter()-start
        streak = streak+1 if metrics["accuracy"] >= target else 0
        if streak and first is None: first = {"round": rnd, "seconds": elapsed}
        if streak >= 3 and sustained is None: sustained = {"round": rnd, "seconds": elapsed, "consecutive_rounds": 3}
        payload = sum(a.nbytes for a in output.values())
        history.append({"round": rnd, **metrics, "elapsed_seconds": elapsed, "server_seconds": server_seconds,
                        "training_payload_bytes": int(2*len(selected)*payload), "evaluation_payload_bytes": 0,
                        "update_oscillation_l2": stable_l2((move[k]-previous_move[k] for k in move)),
                        **strategy._round_metrics})
        previous_move = move
    final_test = evaluate(data, strategy._global, "test")
    memory = (strategy.optimizer_state_bytes() if method != "ema" else
              sum(a.nbytes for group in (strategy._m, strategy._v) for a in group.values()))
    return {"condition": {"dataset": data.name, "alpha": data.metadata["alpha"], "seed": data.metadata["seed"],
                          "local_steps": local_steps, "participation": participation, "drift": False},
            "method": method, "target": target, "target_metric": "validation_accuracy",
            "target_reached": first, "sustained_target": sustained, "post_shift_recovery": None,
            "heldout_test": final_test, "test_evaluations": 1, "history": history,
            "optimizer_numeric_bytes": memory, "dataset_sha256": data.metadata["dataset_sha256"],
            "split_sha256": data.metadata["split_sha256"],
            "schedule_sha256": hashlib.sha256(np.asarray(schedule, dtype="<i8").tobytes()).hexdigest()}


def run_public_benchmark(*, dataset="public", full=False, rounds=60):
    from threadpoolctl import threadpool_limits
    names = ("digits", "breast-cancer") if dataset == "public" else (dataset,)
    if any(name not in ("digits", "breast-cancer") for name in names):
        raise ValueError("unknown public dataset")
    axes = ((.1, .5, 10.), (1, 5, 20), (.1, .5, 1.), range(5)) if full else ((.5,), (5,), (.5,), range(2))
    results, metadata = [], []
    with threadpool_limits(limits=1):
        for name, alpha, seed in product(names, axes[0], axes[3]):
            data = prepare_public_data(name, seed=seed, alpha=alpha)
            metadata.append(data.metadata | {"dataset": name})
            for index, (steps, fraction) in enumerate(product(axes[1], axes[2])):
                # Counterbalance method ordering rather than always timing EMA first.
                offset = (seed+index)%len(METHODS)
                for method in METHODS[offset:]+METHODS[:offset]:
                    results.append(run_public_case(data, method=method, local_steps=steps,
                                                   participation=fraction, rounds=rounds))
    return {"status": "research_public_data_benchmark", "network_transport_measured": False,
            "privacy_enabled": False, "blas_threads": 1, "rounds": rounds,
            "axes": {"datasets": list(names), "alpha": list(axes[0]), "local_steps": list(axes[1]),
                     "participation": list(axes[2]), "seeds": list(axes[3])},
            "method_order": "counterbalanced by seed and condition; no test-based tuning",
            "datasets": metadata, "summary": summarize_public(results), "results": results,
            "excluded": ["clinical validation", "real institution splits", "network timing", "hyperparameter tuning"]}


def summarize_public(results):
    """Separate datasets and condition strata; pair uncertainty only across seeds."""
    from scipy.stats import t
    paired_runs(results)
    datasets, strata = {}, []
    for name in sorted({r["condition"]["dataset"] for r in results}):
        rows = [r for r in results if r["condition"]["dataset"] == name]
        summary = summarize(rows)
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method]
            summary[method].update(
                final_test_accuracy_mean=float(np.mean([r["heldout_test"]["accuracy"] for r in selected])),
                final_test_loss_mean=float(np.mean([r["heldout_test"]["loss"] for r in selected])),
                sustained_target_runs=sum(r["sustained_target"] is not None for r in selected))
        datasets[name] = summary
        conditions = sorted({(r["condition"]["alpha"], r["condition"]["local_steps"], r["condition"]["participation"]) for r in rows})
        for alpha, steps, fraction in conditions:
            matching = [r for r in rows if (r["condition"]["alpha"], r["condition"]["local_steps"], r["condition"]["participation"]) == (alpha, steps, fraction)]
            aligned = paired_runs(matching)
            for method in METHODS:
                differences = np.array([r["heldout_test"]["accuracy"]-b["heldout_test"]["accuracy"]
                                        for r,b in zip(aligned[method], aligned["ema"])])
                n = len(differences)
                mean = float(differences.mean())
                half = float(t.ppf(.975, n-1)*differences.std(ddof=1)/np.sqrt(n)) if n > 1 else None
                strata.append({"dataset": name, "alpha": alpha, "local_steps": steps, "participation": fraction,
                               "method": method, "seeds": n, "paired_test_accuracy_delta": mean,
                               "paired_t_interval_95": [mean-half, mean+half] if half is not None else None})
    return {"datasets": datasets, "condition_strata": strata,
            "accuracy_unit": "fraction; multiply differences by 100 for percentage points",
            "interval_note": "descriptive paired t intervals within fixed conditions; few seeds, no multiple-comparison correction"}
