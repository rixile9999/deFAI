import argparse
import csv
import itertools
import json
from pathlib import Path

from fl_core.backends import get_backend
from fl_core.config import parse_sets
from fl_core.plugins import run_flat
from fl_core.runner import final_accuracy, quiet, write_json
from security_sim.plugin import SecurityPlugin


def run_security(attack="none", defense=(), mode="flat", backend="builtin", backend_config=None, clients=10,
                 edges=3, rounds=5, strategy="fedavg", plugin_config=None, output_dir="results/security",
                 transport="local"):
    quiet()
    cfg = dict(plugin_config or {}, attack=attack, defense=list(defense))
    plugin = SecurityPlugin(cfg)
    tag = f"{mode}_{attack}_{'+'.join(defense) or 'none'}"
    if mode == "flat":
        b = get_backend(backend, backend_config)
        out = run_flat(b, [plugin], clients, rounds, strategy)
        rows = out["rows"]
        summary = {"mode": mode, "attack": attack, "defense": list(defense), "rows": rows, "alerts": out["alerts"],
                   "security": plugin.summary(None), "final_acc": final_accuracy(rows),
                   "P_last3": final_accuracy(rows, 3)}
    else:
        from hierarchical_fl.server import HierarchicalServer

        server = HierarchicalServer({"backend": backend, "backend-config": backend_config or {},
                                     "num-clients": clients, "num-edges": edges, "global-rounds": rounds,
                                     "edge-strategy": strategy, "transport": transport,
                                     "spawn-edges": transport == "http",
                                     "run-id": f"sec_{tag}", "output-dir": str(Path(output_dir) / "runs")},
                                    plugins=[plugin])
        res = server.run()
        summary = {"mode": mode, "attack": attack, "defense": list(defense), "rows": res["rounds"],
                   "alerts": res["alerts"], "security": res["plugins"].get("SecurityPlugin", {}),
                   "final_acc": res["final_acc"], "P_last3": res["P_last3"], "run_dir": str(server.run_dir)}
    write_json(Path(output_dir) / f"{tag}.json", summary)
    return summary


def describe(s):
    sec = s["security"]
    det = sec.get("client_detection") or {}
    edet = sec.get("edge_detection") or {}
    cost = sum(c["total_mb"] for c in sec.get("cost", []))
    return {"mode": s["mode"], "attack": s["attack"], "defense": "+".join(s["defense"]) or "none",
            "final_acc": round(s["final_acc"], 4), "P_last3": round(s["P_last3"], 4),
            "client_precision": det.get("precision"), "client_recall": det.get("recall"),
            "edge_precision": edet.get("precision"), "edge_recall": edet.get("recall"),
            "alerts": len(s["alerts"]), "net_cost_mb": round(cost, 2)}


def cmd_run(args):
    s = run_security(args.attack, [d for d in args.defense.split(",") if d], args.mode, args.backend,
                     parse_sets(args.set), args.clients, args.edges, args.rounds, args.strategy,
                     {"ratio": args.ratio}, args.output_dir, args.transport)
    print(json.dumps(describe(s), indent=2, ensure_ascii=False))
    for a in s["alerts"][:10]:
        print(f"  [R{a['round']}] {a['level']} {a['source']}: {a['message']}")


SCENARIOS = {
    "y1": [("data_noise", "data_dist"), ("model_poison", "stat"), ("sybil", "cert")],
    "y2": [("edge_relay", "hmac"), ("relay_tamper", "hmac"), ("hier_collusion", "kmeans"),
           ("phishing", "mfa,monitor"), ("model_poison", "ocsvm")],
    "y3": [("insider_collusion", "monitor"), ("adaptive_collusion", "monitor,stat")],
}


def cmd_matrix(args):
    rows = []
    groups = args.groups.split(",")
    pairs = list(itertools.chain.from_iterable(SCENARIOS[g] for g in groups))
    for attack, defense in pairs:
        mode = "hier" if attack in ("edge_relay", "hier_collusion", "phishing") else args.mode
        for d in ("", defense):
            s = run_security(attack, [x for x in d.split(",") if x], mode, args.backend, parse_sets(args.set),
                             args.clients, args.edges, args.rounds, "fedavg", {"ratio": args.ratio}, args.output_dir)
            rows.append(describe(s))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    base = run_security("none", [], args.mode, args.backend, parse_sets(args.set), args.clients, args.edges,
                        args.rounds, "fedavg", {}, args.output_dir)
    rows.insert(0, describe(base))
    out = Path(args.output_dir) / "matrix.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


def cmd_inversion(args):
    from security_sim.inversion import run_inversion

    quiet()
    b = get_backend(args.backend, parse_sets(args.set))
    rows = run_inversion(b, queries=args.queries, invert_steps=args.steps)
    write_json(Path(args.output_dir) / "inversion.json", rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m security_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default="builtin")
    common.add_argument("--set", action="append", default=[])
    common.add_argument("--clients", type=int, default=10)
    common.add_argument("--edges", type=int, default=3)
    common.add_argument("--rounds", type=int, default=5)
    common.add_argument("--ratio", type=float, default=0.2)
    common.add_argument("--mode", choices=("flat", "hier"), default="flat")
    common.add_argument("--output-dir", default="results/security")
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--attack", default="none")
    r.add_argument("--defense", default="")
    r.add_argument("--strategy", default="fedavg")
    r.add_argument("--transport", default="local")
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("matrix", parents=[common])
    m.add_argument("--groups", default="y1,y2,y3")
    m.set_defaults(fn=cmd_matrix)
    i = sub.add_parser("inversion", parents=[common])
    i.add_argument("--queries", type=int, default=2000)
    i.add_argument("--steps", type=int, default=200)
    i.set_defaults(fn=cmd_inversion)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
