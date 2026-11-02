"""Paired synthetic logistic FL benchmark; no downloaded or clinical data.

The complete preset is 3 heterogeneity levels x 3 local step counts x 3
participation rates x 2 drift conditions x 5 seeds x 3 first-moment methods.
"""
from itertools import product
import hashlib
import math
import time

import numpy as np


METHODS = ("ema", "kalman", "adaptive-kalman")


def make_clients(alpha, seed, clients=10):
    """Dirichlet class proportions; independently generated train/test examples."""
    rng = np.random.default_rng(seed)
    datasets = []
    for _ in range(clients):
        proportions = rng.dirichlet([alpha, alpha])
        sets = []
        for count in (48, 96):
            y = rng.choice(2, size=count, p=proportions).astype(float)
            x = rng.normal(size=(count, 6))
            x[:, 0] += 2 * y - 1
            sets.append((np.column_stack([x, np.ones(count)]), y))
        datasets.append(sets)
    return datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def evaluate(w, clients, flipped):
    accuracies, losses = [], []
    for _, (x, original_y) in clients:
        y = 1 - original_y if flipped else original_y
        logits = x @ w
        accuracies.append(float(np.mean((logits >= 0) == y)))
        losses.append(float(np.mean(np.logaddexp(0, logits) - y * logits)))
    return {"accuracy": float(np.mean(accuracies)), "loss": float(np.mean(losses)),
            "client_accuracy_p10": float(np.quantile(accuracies, .1))}


def run_case(*, alpha, local_steps, participation, drift, seed, method, rounds=40, target=.75):
    """Run one method with deterministic client data and participation schedule."""
    from defai.kalman import KalmanConfig
    from defai.strategies import FedAdamKalman
    if method not in METHODS:
        raise ValueError("unknown benchmark method")
    if (not math.isfinite(alpha) or alpha <= 0 or isinstance(local_steps, bool) or
            not isinstance(local_steps, int) or local_steps < 1 or not 0 < participation <= 1 or
            isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2 or
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or not 0 < target <= 1 or not isinstance(drift, bool)):
        raise ValueError("invalid benchmark condition")
    clients = make_clients(alpha, seed)
    rng = np.random.default_rng(seed + 10000)
    strategy = FedAdamKalman(server_learning_rate=.03, kalman_blend=0 if method == "ema" else 1,
                            kalman=KalmanConfig(process_adaptation=.2 if method == "adaptive-kalman" else 0))
    strategy._global = {"w": np.zeros(7)}
    history, schedule = [], []
    reached = recovery = None
    previous_move = np.zeros(7)
    start = time.perf_counter()
    for rnd in range(1, rounds + 1):
        flipped = drift and rnd > rounds // 2
        selected = rng.choice(len(clients), size=max(1, int(len(clients)*participation)), replace=False)
        schedule.append(selected.tolist())
        w = strategy._global["w"]
        updates = []
        for index in selected:
            (x, original_y), _ = clients[index]
            y = 1 - original_y if flipped else original_y
            local = w.copy()
            for _ in range(local_steps):
                local -= .1 * (x.T @ (sigmoid(x @ local) - y) / len(y))
            updates.append(local - w)
        server_start = time.perf_counter()
        output = strategy.update_global(rnd, {"w": np.mean(updates, axis=0)})
        server_seconds = time.perf_counter() - server_start
        move = output["w"] - w
        strategy._global = output
        metrics = evaluate(output["w"], clients, flipped)
        elapsed = time.perf_counter()-start
        if metrics["accuracy"] >= target:
            if reached is None and not flipped: reached = {"round": rnd, "seconds": elapsed}
            if recovery is None and flipped: recovery = {"rounds_after_shift": rnd-rounds//2, "seconds_since_start": elapsed}
        history.append({"round": rnd, **metrics, "elapsed_seconds": elapsed, "server_seconds": server_seconds,
                        "update_oscillation_l2": float(np.linalg.norm(move - previous_move)),
                        "training_payload_bytes": len(selected)*2*w.nbytes,
                        "evaluation_payload_bytes": len(clients)*(w.nbytes+16),
                        **strategy._round_metrics})
        previous_move = move
    return {"condition": {"alpha": alpha, "local_steps": local_steps, "participation": participation,
                          "drift": drift, "seed": seed}, "method": method,
            "schedule_sha256": hashlib.sha256(np.asarray(schedule, dtype=np.int64).tobytes()).hexdigest(),
            "target": target, "target_reached": reached, "post_shift_recovery": recovery,
            "optimizer_numeric_bytes": strategy.optimizer_state_bytes(), "history": history}


def run_benchmark(*, full=False, rounds=40):
    axes = ((.1, .5, 10.), (1, 5, 20), (.1, .5, 1.), (False, True), range(5)) if full else ((.5,), (5,), (.5,), (False, True), range(2))
    results = [run_case(alpha=a, local_steps=s, participation=p, drift=d, seed=seed, method=m, rounds=rounds)
               for a, s, p, d, seed in product(*axes) for m in METHODS]
    return {"status": "research_synthetic_benchmark", "data": "generated synthetic binary logistic clients",
            "network_transport_measured": False, "privacy_enabled": False,
            "axes": {"alpha": list(axes[0]), "local_steps": list(axes[1]), "participation": list(axes[2]),
                     "drift": list(axes[3]), "seeds": list(axes[4])}, "rounds": rounds,
            "results": results, "excluded": ["real datasets", "clinical/manufacturing validation", "Bayesian search",
                                                 "network latency", "hyperparameter tuning", "privacy claims"]}
