"""Compare the current estimator with the pre-follow-up Git revision, locally."""
import argparse
import importlib.util
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tracemalloc

import numpy as np

from defai import kalman
from defai.io import write_json

REFERENCE = "35e202718400ebfe61569d893768d79762b2ff7e"


def compare(root, repeats=40):
    if repeats < 3: raise ValueError("repeats must be >= 3")
    source = subprocess.check_output(["git", "show", REFERENCE+":defai/kalman.py"], cwd=root)
    with tempfile.TemporaryDirectory(prefix="kalman-reference-") as directory:
        path = Path(directory)/"reference.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("kalman_profile_reference", path)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
    rows = []
    for size in (16, 10000, 1000000):
        for mode in ("fixed", "adaptive"):
            observation = np.random.default_rng(7).normal(0, .01, size)
            modules = (old, kalman)
            configs = [m.KalmanConfig(process_adaptation=.2 if mode == "adaptive" else 0) for m in modules]
            states, timings = [None, None], [[], []]
            for _ in range(3): states = [m.layer_step(observation, s, c) for m,s,c in zip(modules, states, configs)]
            for repeat in range(repeats):
                for index in ((0, 1) if repeat%2 == 0 else (1, 0)):
                    start = time.perf_counter()
                    states[index] = modules[index].layer_step(observation, states[index], configs[index])
                    timings[index].append(time.perf_counter()-start)
            np.testing.assert_allclose(states[0].mean, states[1].mean, rtol=1e-13, atol=1e-15)
            peaks = []
            for m,s,c in zip(modules, states, configs):
                tracemalloc.start()
                try:
                    state = m.layer_step(observation, s, c)
                    peaks.append(tracemalloc.get_traced_memory()[1])
                    assert state.mean.shape == observation.shape
                finally: tracemalloc.stop()
            medians = [float(np.median(t)) for t in timings]
            rows.append({"elements": size, "mode": mode, "paired_repeats": repeats,
                         "reference_median_seconds": medians[0], "current_median_seconds": medians[1],
                         "latency_reduction_percent": 100*(1-medians[1]/medians[0]),
                         "reference_traced_peak_bytes": peaks[0], "current_traced_peak_bytes": peaks[1],
                         "reference_seconds": timings[0], "current_seconds": timings[1]})
    return {"reference": REFERENCE, "numpy": np.__version__, "python": platform.python_version(), "results": rows,
            "limits": "Counterbalanced kernel calls in one process; not training/network acceleration or total RSS"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = compare(Path(__file__).resolve().parents[1], args.repeats)
    write_json(args.output, result, overwrite=False)
    print(str(args.output.resolve()))
