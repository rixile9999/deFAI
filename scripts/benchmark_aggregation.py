"""Compare the former stacked weighted mean with the validated streaming mean.

Synthetic arrays only. Measures kernel runtime and peak traced allocations, not
training accuracy, network throughput, or total process resident memory.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

import numpy as np

from defai.aggregation import weighted_mean


def benchmark(clients=32, parameters=200_000, repeats=5):
    if clients < 2 or parameters < 1 or repeats < 2:
        raise ValueError("clients >= 2, parameters >= 1 and repeats >= 2 are required")
    rng = np.random.default_rng(42)
    updates = [{"w": rng.normal(size=parameters).astype(np.float32)} for _ in range(clients)]
    weights = rng.uniform(.1, 2, clients)

    def stacked():
        stack = np.stack([u["w"] for u in updates]).astype(np.float64)
        return np.tensordot(weights/weights.sum(), stack, axes=1).astype(np.float32)

    def streaming():
        return weighted_mean(updates, weights)["w"]

    expected, actual = stacked(), streaming()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    result = {"scope": "weighted_mean_kernel_synthetic_arrays", "python": platform.python_version(),
              "numpy": np.__version__, "clients": clients, "parameters_per_client": parameters,
              "dtype": "float32", "repeats": repeats, "max_abs_difference": float(np.max(np.abs(actual-expected))),
              "measurements": {}}
    operations = {"stacked": stacked, "streaming": streaming}
    measurements = {name: {"durations": [], "peaks": []} for name in operations}
    for operation in operations.values():
        operation()
    for repeat in range(repeats):
        names = list(operations) if repeat % 2 == 0 else list(reversed(operations))
        for name in names:
            tracemalloc.start()
            start = time.perf_counter()
            operations[name]()
            measurements[name]["durations"].append(time.perf_counter()-start)
            measurements[name]["peaks"].append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
    result["order"] = "alternating_interleaved"
    for name, measured in measurements.items():
        result["measurements"][name] = {"median_seconds": statistics.median(measured["durations"]),
                                        "peak_traced_bytes": max(measured["peaks"])}
    result["peak_allocation_reduction_percent"] = 100 * (1-result["measurements"]["streaming"]["peak_traced_bytes"] / result["measurements"]["stacked"]["peak_traced_bytes"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
