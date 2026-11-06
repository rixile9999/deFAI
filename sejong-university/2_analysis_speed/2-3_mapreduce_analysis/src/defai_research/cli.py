from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import __version__

STUDIES = {
    "multimodal": "purpose_multimodal_study",
    "system": "purpose_system_study",
    "privacy": "purpose_privacy_study",
    "recovery": "purpose_recovery_stress",
    "crossnode": "supplement_crossnode_information",
    "verify-multimodal": "verify_purpose_multimodal",
    "verify-system": "verify_purpose_system",
    "verify-privacy": "verify_purpose_privacy_study",
    "cross-verify-multimodal": "cross_verify_purpose_multimodal",
    "transport-echo": "purpose_transport_echo",
    "export-system": "purpose_system_export",
    "export-privacy": "purpose_privacy_export",
    "privacy-control": "purpose_privacy_attack_control",
}


def _emit(value, output=None):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(content, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify-aggregation", help="Check weighted MapReduce and sufficient-statistic identities")
    check.add_argument("--full", action="store_true", help="Run 110 vector and 30 statistic conditions")
    check.add_argument("--output", type=Path)
    audit = sub.add_parser("verify-run", help="Independently check saved byte ledgers, masks and model predictions")
    audit.add_argument("run", type=Path)
    audit.add_argument("--output", type=Path)
    collect = sub.add_parser("collect-calibration", help="Collect real training-only Flower update blocks")
    collect.add_argument("--output", required=True, type=Path)
    collect.add_argument("--rounds", type=int, default=24)
    collect.add_argument("--drift-round", type=int, default=13)
    collect.add_argument("--max-blocks-per-role", type=int, default=12000)
    train = sub.add_parser("train-codec", help="Train and export the selected Transformer architecture")
    train.add_argument("--calibration", required=True, type=Path, help="Directory created by collect-calibration")
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=60)
    run = sub.add_parser("benchmark", help="Run actual GraphSAGE and decoded two-level Flower aggregation")
    run.add_argument("--dataset", choices=("digits", "breast_cancer"), default="digits")
    run.add_argument("--seed", type=int, default=19001)
    run.add_argument("--rounds", type=int, default=80)
    run.add_argument("--drift-round", type=int, default=None)
    run.add_argument("--codec", choices=("transformer", "compact", "legacy"), default="transformer",
                     help="legacy is a reference comparator, not the default transmission path")
    run.add_argument("--artifact", type=Path)
    run.add_argument("--static", action="store_true", help="Keep the candidate graph fixed as a matched control")
    run.add_argument("--evaluate-test", action="store_true", help="Explicitly evaluate the held-out test after validation checkpoint selection")
    run.add_argument("--output", required=True, type=Path)
    schema = sub.add_parser("metrics-schema", help="Print the sixteen metric definitions and units")
    schema.add_argument("--output", type=Path)
    metrics = sub.add_parser("metrics", help="Evaluate predictions and supplied purpose observations")
    metrics.add_argument("--input", type=Path, help="NPZ with labels and probabilities; loading uses allow_pickle=False")
    metrics.add_argument("--observations", type=Path, help="JSON of measured values indexed by metric field")
    metrics.add_argument("--output", type=Path)
    study = sub.add_parser("metric-study", help="Run a retained purpose study or its independent verifier")
    study.add_argument("study", choices=tuple(STUDIES))
    study.add_argument("--workspace", type=Path, default=None, help="Generated output root, defaults to current directory")
    study.add_argument("arguments", nargs=argparse.REMAINDER, help="Study arguments, optionally after --")
    args = parser.parse_args(argv)
    if args.command == "verify-aggregation":
        from .validation import verify_aggregation
        _emit(verify_aggregation(full=args.full), args.output)
    elif args.command == "verify-run":
        from .audit import verify_benchmark
        result = verify_benchmark(args.run)
        _emit(result, args.output)
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "collect-calibration":
        from .compression.calibration import collect_calibration
        _emit(collect_calibration(args.output, args.max_blocks_per_role,
                                  rounds=args.rounds, drift_round=args.drift_round))
    elif args.command == "train-codec":
        from .compression import TransformerCodecConfig, fit_transformer_codec, export_inference_artifact
        manifest = json.loads((args.calibration / "manifest.json").read_text(encoding="utf-8"))
        trained = fit_transformer_codec(args.calibration / "calibration_blocks.npz", args.output,
                                        config=TransformerCodecConfig(epochs=args.epochs), manifest=manifest, verbose=True)
        _emit({"training": trained, "inference": export_inference_artifact(trained["artifact_path"])})
    elif args.command == "benchmark":
        from .compression.runner import run_benchmark
        drift = args.drift_round if args.drift_round is not None else (41 if args.rounds >= 41 else args.rounds // 2 + 1)
        _emit(run_benchmark(output=args.output, dataset=args.dataset, seed=args.seed, rounds=args.rounds,
                            drift_round=drift, codec=args.codec, artifact=args.artifact,
                            dynamic=not args.static, allow_test=args.evaluate_test))
    elif args.command == "metrics-schema":
        from .metrics import metric_definitions
        _emit(metric_definitions(), args.output)
    elif args.command == "metrics":
        import numpy as np
        from .metrics import evaluate16
        labels = probabilities = None
        if args.input:
            with np.load(args.input, allow_pickle=False) as source:
                labels, probabilities = source["labels"].copy(), source["probabilities"].copy()
        observations = json.loads(args.observations.read_text(encoding="utf-8")) if args.observations else None
        _emit(evaluate16(labels=labels, probabilities=probabilities, observations=observations), args.output)
    elif args.command == "metric-study":
        environment = os.environ.copy()
        if args.workspace:
            environment["DEFAI_WORKSPACE"] = str(args.workspace.resolve())
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        completed = subprocess.run([sys.executable, "-m", f"defai_research.metrics.studies.{STUDIES[args.study]}", *arguments],
                                   env=environment, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
