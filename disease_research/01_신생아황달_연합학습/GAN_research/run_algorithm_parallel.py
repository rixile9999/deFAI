"""Run independent data/bias experiments concurrently, then unlock test once."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/algorithm_study"


def main():
    subprocess.run([sys.executable, "-u", "run_algorithm_study.py", "--plan-only"], cwd=ROOT, check=True)
    execution = {"independent_context_processes": 4, "cpu_threads_per_context": 2,
                 "fl_clients_per_experiment": "3 sequential, full participation",
                 "test_barrier": "all four context workers complete before original coordinator test evaluation"}
    (OUTPUT / "parallel_execution.json").write_text(json.dumps(execution, indent=2))
    processes, logs = [], []
    try:
        for context in range(4):
            log = (OUTPUT / f"context_{context}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, "-u", "train_algorithm_context.py", "--context-index", str(context)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in [None, 0] for p in processes):
                raise RuntimeError("Experiment worker failed; inspect context logs; completed runs are retained")
            time.sleep(3)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("An experiment worker failed")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
    # Reuses every completed validation run, reconstructs selections deterministically,
    # and performs the original global test barrier and inference.
    subprocess.run([sys.executable, "-u", "run_algorithm_study.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
