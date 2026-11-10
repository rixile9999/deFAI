import argparse

from fl_core.config import merge, parse_sets
from scenario_test.runner import run_suite
from scenario_test.scenarios import SCENARIOS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scenario_test")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--backend", default="builtin")
    ap.add_argument("--clients", type=int, default=12)
    ap.add_argument("--edges", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--output-dir", default="results/scenarios")
    args = ap.parse_args(argv)
    base = merge({"backend": args.backend, "num-clients": args.clients, "num-edges": args.edges,
                  "global-rounds": args.rounds, "backend-config": {}}, parse_sets(args.set))
    run_suite(args.scenarios.split(","), [int(s) for s in args.seeds.split(",")], base, args.output_dir)


if __name__ == "__main__":
    main()
