"""Local estimator latency/allocation diagnostics, without training dependencies."""
import platform
import time
import tracemalloc

import numpy as np

from defai.kalman import KalmanConfig, layer_step


def profile_estimator(*, sizes=(16, 10000, 1000000), repeats=20):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
        raise ValueError("repeats must be an integer >= 3")
    if not sizes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError("sizes must contain positive integers")
    if tracemalloc.is_tracing(): raise ValueError("stop existing allocation tracing before profiling")
    rows = []
    for count in sizes:
        observation = np.random.default_rng(7).normal(0, .01, count)
        for mode in ("fixed", "adaptive"):
            config = KalmanConfig(process_adaptation=.2 if mode == "adaptive" else 0.)
            state = None
            for _ in range(3): state = layer_step(observation, state, config)
            elapsed = []
            for _ in range(repeats):
                start = time.perf_counter()
                state = layer_step(observation, state, config)
                elapsed.append(time.perf_counter()-start)
            tracemalloc.start()
            try:
                state = layer_step(observation, state, config)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            rows.append({"elements": count, "mode": mode, "repeats": repeats,
                         "median_seconds": float(np.median(elapsed)), "min_seconds": min(elapsed),
                         "p90_seconds": float(np.quantile(elapsed, .9)), "traced_peak_bytes": peak,
                         "state_numeric_bytes": state.mean.nbytes+5*8})
    return {"status": "research_local_kernel_profile", "python": platform.python_version(), "numpy": np.__version__,
            "results": rows, "limits": "Local kernel calls; traced allocation is not whole-process RSS, training time or network latency"}
