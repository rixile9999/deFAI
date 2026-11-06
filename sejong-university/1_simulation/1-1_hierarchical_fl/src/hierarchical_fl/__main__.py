import argparse
import json
import sys

from fl_core.config import parse_sets


def cmd_run(args):
    from hierarchical_fl.server import run_hierarchical

    overrides = json.loads(open(args.config, encoding="utf-8").read()) if args.config else {}
    from fl_core.config import merge

    overrides = merge(overrides, parse_sets(args.set))
    if args.edges:
        overrides.update({"transport": "http", "edge-urls": args.edges.split(",")})
    if args.spawn_edges:
        overrides.update({"transport": "http", "spawn-edges": True})
    out = run_hierarchical(overrides)
    print(json.dumps({k: out[k] for k in ("run_id", "final_acc", "best_acc", "P_last3", "virtual_ms_total",
                                          "bytes_total")}, indent=2))
    print(f"results: {out['config']['output-dir']}/{out['run_id']}")


def cmd_edge(args):
    from hierarchical_fl.edge_server import main

    main(["--edge-id", str(args.edge_id), "--host", args.host, "--port", str(args.port)])


def cmd_dashboard(args):
    from hierarchical_fl.dashboard import main

    main(["--results", args.results, "--host", args.host, "--port", str(args.port)])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hierarchical_fl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config")
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--edges")
    r.add_argument("--spawn-edges", action="store_true")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("edge")
    e.add_argument("--edge-id", type=int, required=True)
    e.add_argument("--host", default="127.0.0.1")
    e.add_argument("--port", type=int, required=True)
    e.set_defaults(fn=cmd_edge)
    d = sub.add_parser("dashboard")
    d.add_argument("--results", default="results")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=5000)
    d.set_defaults(fn=cmd_dashboard)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
