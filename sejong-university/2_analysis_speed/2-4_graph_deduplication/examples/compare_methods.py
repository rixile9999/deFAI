from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from defai_research.compression.runner import run_benchmark

METHODS = {
    "legacy_dynamic": dict(codec="legacy", dynamic=True),
    "compact_dynamic": dict(codec="compact", dynamic=True),
    "transformer_static": dict(codec="transformer", dynamic=False),
    "transformer_dynamic": dict(codec="transformer", dynamic=True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", choices=("digits", "breast_cancer"), nargs="+", default=["digits", "breast_cancer"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(19001, 19013)))
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--drift-round", type=int, default=41)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new output directory to preserve existing comparison results")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(methods=METHODS, datasets=args.datasets, seeds=args.seeds,
                    rounds=args.rounds, drift_round=args.drift_round, test_evaluation=args.evaluate_test,
                    selection="Fixed methods and supplied artifact; no candidate or test-based selection")
    (args.output / "comparison_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for method, settings in METHODS.items():
                result = run_benchmark(output=args.output / "runs" / f"{dataset}_{seed}_{method}",
                    dataset=dataset, seed=seed, rounds=args.rounds, drift_round=args.drift_round,
                    artifact=args.artifact if settings["codec"] == "transformer" else None,
                    allow_test=args.evaluate_test, **settings)
                rows.append({"method": method, **result})
                with (args.output / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(dict(completed=len(rows), planned=len(args.datasets)*len(args.seeds)*len(METHODS),
                                      dataset=dataset, seed=seed, method=method)), flush=True)


if __name__ == "__main__":
    main()
