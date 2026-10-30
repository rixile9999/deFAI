"""One command surface for algorithms, data workflows and packaged research."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from defai import __version__


def _json(value):
    print(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))


def _output(path, value):
    from defai.io import write_json
    write_json(Path(path), value, overwrite=False)


def parser():
    root = argparse.ArgumentParser(description="Federated learning and disease research (research stage).")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the workflow families and optional install groups")
    doctor = commands.add_parser("doctor", help="Inspect dependencies without initializing GPU libraries")
    doctor.add_argument("--probe-gpu", action="store_true")
    init = commands.add_parser("init", help="Create a writable workspace with all code and source references")
    init.add_argument("directory", type=Path)
    demo = commands.add_parser("demo", help="Run all workflow families with small synthetic inputs; no downloads")
    demo.add_argument("--output", type=Path, required=True)
    data = commands.add_parser("data", help="Inspect public sources and explicitly download checksum-verified files")
    data_actions = data.add_subparsers(dest="data_action", required=True)
    data_actions.add_parser("list")
    fetch = data_actions.add_parser("fetch")
    fetch.add_argument("dataset")
    fetch.add_argument("--directory", type=Path, required=True)
    prepare = data_actions.add_parser("prepare-neojaundice")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("--directory", type=Path, required=True)
    run = commands.add_parser("run", help="Train any common algorithm using real clients in one process")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--clients", type=int, default=4)
    run.add_argument("--rounds", type=int, default=2)
    run.add_argument("--strategy", default="fedavg")
    run.add_argument("--dataset", default="synthetic")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--set", action="append", default=[], metavar="KEY=TOML_VALUE")
    research = commands.add_parser("research", help="Discover and run every packaged study in a workspace")
    actions = research.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list")
    listing.add_argument("--domain", choices=("neonatal", "policy", "generation", "hba1c"))
    show = actions.add_parser("show")
    show.add_argument("feature")
    execution = actions.add_parser("run")
    execution.add_argument("feature")
    execution.add_argument("--workspace", type=Path, required=True)
    execution.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    execution.add_argument("--timeout", type=float)
    generation = commands.add_parser("generation", help="Fit a federated Burr distribution to interval CSV")
    generation.add_argument("--input", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    policy = commands.add_parser("policy", help="Analyze explicitly calendar-aligned regional daily counts")
    policy.add_argument("--input", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy.add_argument("--start", default="2024-01-01")
    policy.add_argument("--policy-date", default="2024-01-31")
    policy.add_argument("--end", default="2024-04-09")
    policy.add_argument("--date-column", default="발병일")
    mobile = commands.add_parser("mobile", help="Serve the round-safe Android parameter protocol")
    mobile.add_argument("--output", type=Path, required=True)
    mobile.add_argument("--host", default="127.0.0.1")
    mobile.add_argument("--port", type=int, default=8001)
    mobile.add_argument("--clients", type=int, default=1)
    android = commands.add_parser("android", help="Build the Android app in a portable temporary directory")
    android.add_argument("--sdk", type=Path, required=True)
    android.add_argument("--output", type=Path, required=True)
    android.add_argument("--java-home", type=Path)
    return root


def dispatch(args, forwarded):
    if forwarded and (args.command != "research" or args.action != "run"):
        raise ValueError("Additional arguments are supported only after 'research run ... --'")
    if args.command == "list":
        _json({"version": __version__, "workflows": {
            "common": "defai run --strategy NAME --set KEY=VALUE [flower]",
            "neonatal": "defai research list --domain neonatal [research]",
            "policy": "defai policy --input DIRECTORY --output result.json [analysis]",
            "generation": "defai generation --input intervals.csv --output result.json [analysis]",
            "hba1c": "defai research list --domain hba1c [research]",
            "mobile": "defai mobile --output DIRECTORY [mobile]"},
            "python_api": ["defai.strategies", "defai.aggregation", "defai.privacy", "defai.compression",
                           "defai.research.neonatal", "defai.research.policy", "defai.research.generation", "defai.research.hba1c"]})
    elif args.command == "doctor":
        packages = ("numpy", "scipy", "pandas", "torch", "torchvision", "flwr", "flwr-datasets", "tenseal", "flask", "waitress")
        versions = {}
        for name in packages:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        result = {"defai": __version__, "python": sys.version.split()[0], "executable": sys.executable, "packages": versions}
        if args.probe_gpu:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            result["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        _json(result)
    elif args.command == "init":
        from defai.research import create_workspace
        _json({"workspace": str(create_workspace(args.directory))})
    elif args.command == "demo":
        from defai.demo import run_demo
        result = run_demo(args.output)
        _json({"output": str(args.output.resolve()), "elapsed_seconds": result["elapsed_seconds"],
               "data": result["data"], "clinical_validation": False})
    elif args.command == "data":
        from defai.data import datasets, fetch_dataset, prepare_neojaundice
        if args.data_action == "list":
            _json({key: {"name": rows[0]["dataset"], "files": len(rows),
                          "sources": sorted({row["source_url"] for row in rows})}
                   for key, rows in datasets().items()})
        elif args.data_action == "fetch":
            _json(fetch_dataset(args.dataset, args.directory))
        else:
            _json(prepare_neojaundice(args.archive, args.directory))
    elif args.command == "run":
        from defai.simulation import run_simulation
        cfg = {"strategy": args.strategy, "dataset": args.dataset, "num-server-rounds": args.rounds}
        for item in args.set:
            key, separator, value = item.partition("=")
            if not separator or not key:
                raise ValueError("--set needs KEY=TOML_VALUE")
            cfg[key] = tomllib.loads("value=" + value)["value"]
        summary = run_simulation(args.output, overrides=cfg, clients=args.clients, device=args.device, threads=args.threads)
        _json({"output": str(args.output.resolve()), "completed_rounds": len(summary["history"])-1,
               "elapsed_seconds": summary["elapsed_seconds"], "backend": summary["backend"]})
    elif args.command == "research":
        from defai.research import list_features, run_feature
        if args.action == "list":
            _json(list_features(args.domain))
        elif args.action == "show":
            matches = [f for f in list_features() if f["id"] == args.feature]
            if not matches:
                raise ValueError("Unknown feature; use 'defai research list'")
            _json(matches[0])
        else:
            return run_feature(args.feature, args.workspace, forwarded, device=args.device, timeout=args.timeout)
    elif args.command == "generation":
        from defai.research.generation import build_hierarchy, federated_fit, load_district_pairs
        samples = load_district_pairs(args.input, strict=True)
        districts, provinces = build_hierarchy(samples)
        parameters, nll, rounds, result = federated_fit(provinces)
        if not result.success:
            raise ValueError(f"Optimizer did not converge: {result.message}")
        _output(args.output, {"stage": "research_only", "parameters": dict(zip(("c", "k", "scale_days"), parameters)),
                              "negative_log_likelihood": float(nll), "rounds": rounds,
                              "districts": len(districts), "observations": sum(d.n for d in districts)})
        _json({"output": str(args.output.resolve())})
    elif args.command == "policy":
        from defai.research.policy import (AnalysisConfig, analyze, daily_counts, district_identity,
                                           district_vector, province_summary, read_district_file)
        import numpy as np
        config = AnalysisConfig(args.start, args.policy_date, args.end, args.date_column)
        paths = sorted([*args.input.glob("*.csv"), *args.input.glob("*.xlsx")])
        if not paths:
            raise ValueError("No regional CSV/XLSX files found")
        groups, identities = {}, set()
        for path in paths:
            frame = read_district_file(path)
            required = {"record_format", "date", "cases", "시군구", "treated"}
            if not required <= set(frame.columns) or not {"시도", "거주지시도"} & set(frame.columns):
                raise ValueError("Regional panel schema missing date/counts/province/district/treated columns")
            if args.date_column == "진단일" and "diagnosed_cases" not in frame:
                raise ValueError("Diagnosis-date panels require diagnosed_cases")
            region, district = district_identity(frame)
            if (region, district) in identities:
                raise ValueError("Duplicate district input")
            identities.add((region, district))
            if "treated" not in frame or frame.treated.nunique() != 1 or frame.treated.isna().any() or frame.treated.iloc[0] not in (0, 1):
                raise ValueError("Every district needs an explicit, constant treated=0/1 column")
            if "record_format" not in frame:
                raise ValueError("Use complete daily_counts_v1 panels; missing surveillance dates cannot be assumed zero")
            treated = int(frame.treated.iloc[0])
            if region in groups and groups[region][0] != treated:
                raise ValueError("Districts within one province must share treatment assignment")
            groups.setdefault(region, (treated, []))[1].append(district_vector(daily_counts(frame, config), config))
        summaries = [province_summary(np.sum(vectors, axis=0), region, treated, config, len(vectors))
                     for region, (treated, vectors) in sorted(groups.items())]
        _output(args.output, analyze(summaries))
        _json({"output": str(args.output.resolve())})
    elif args.command == "mobile":
        from defai.mobile import create_app
        from waitress import serve
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        serve(create_app(args.output, expected_clients=args.clients), host=args.host, port=args.port)
    elif args.command == "android":
        from defai.android import build_android
        _json(build_android(args.output, args.sdk, java_home=args.java_home))
    return 0


def main(argv=None):
    root = parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    forwarded = []
    if "--" in raw:
        at = raw.index("--")
        raw, forwarded = raw[:at], raw[at+1:]
    args = root.parse_args(raw)
    try:
        return dispatch(args, forwarded)
    except (ValueError, OSError, ModuleNotFoundError, tomllib.TOMLDecodeError) as error:
        print(f"defai: {error}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("defai: research execution timed out; see workspace/run_receipts", file=sys.stderr)
        return 124
    except subprocess.CalledProcessError as error:
        print(f"defai: build command failed with exit code {error.returncode}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
