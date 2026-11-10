import argparse
import csv
from pathlib import Path


from fl_core.backends import get_backend
from fl_core.config import parse_sets
from fl_core.plugins import run_flat
from fl_core.runner import final_accuracy, quiet, write_json
from network_cost.plugin import OPTIMIZATIONS, NetworkCostPlugin

VARIANTS = {"baseline": [], "edge_compute": ["edge_compute"], "cache": ["cache"], "dag": ["dag"],
            "all": list(OPTIMIZATIONS)}


def run_variant(name, optimizations, mode, backend, backend_config, clients, edges, rounds, plugin_config,
                output_dir="results/network_cost"):
    plugin = NetworkCostPlugin(dict(plugin_config or {}, optimizations=optimizations))
    if mode == "flat":
        out = run_flat(get_backend(backend, backend_config), [plugin], clients, rounds)
        acc, p3 = final_accuracy(out["rows"]), final_accuracy(out["rows"], 3)
        summary = plugin.summary(None)
    else:
        from hierarchical_fl.server import HierarchicalServer

        server = HierarchicalServer({"backend": backend, "backend-config": backend_config or {},
                                     "num-clients": clients, "num-edges": edges, "global-rounds": rounds,
                                     "run-id": f"netcost_{name}", "output-dir": str(Path(output_dir) / "runs")},
                                    plugins=[plugin])
        res = server.run()
        acc, p3 = res["final_acc"], res["P_last3"]
        summary = res["plugins"]["NetworkCostPlugin"]
    return {"variant": name, "final_acc": acc, "P_last3": p3, **{k: v for k, v in summary.items() if k != "rounds"},
            "rounds": summary["rounds"]}


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    base = results[0]["rounds"]
    ax[0].plot([r["round"] for r in base], [r["bytes_base"] / 2 ** 20 for r in base], "k--", label="no optimisation")
    ax[1].plot([r["round"] for r in base], [r["cost_base"] for r in base], "k--", label="no optimisation")
    for res in results:
        if res["variant"] == "baseline":
            continue
        ax[0].plot([r["round"] for r in res["rounds"]], [r["bytes_opt"] / 2 ** 20 for r in res["rounds"]],
                   marker="o", label=res["variant"])
        ax[1].plot([r["round"] for r in res["rounds"]], [r["cost_opt"] for r in res["rounds"]], marker="o",
                   label=res["variant"])
    ax[0].set(title="transferred data per round (MB, all hops)", xlabel="round")
    ax[1].set(title="network cost per round", xlabel="round")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m network_cost")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--mode", choices=("flat", "hier"), default="flat")
    ap.add_argument("--backend", default="builtin")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--edges", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--output-dir", default="results/network_cost")
    args = ap.parse_args(argv)
    quiet()
    results = []
    for name in args.variants.split(","):
        res = run_variant(name, VARIANTS[name], args.mode, args.backend, parse_sets(args.set), args.clients,
                          args.edges, args.rounds, {"ratio": args.ratio}, args.output_dir)
        results.append(res)
        print(f"{name:13s} acc={res['final_acc']:.4f} MB {res['bytes_base_mb']:.1f}->{res['bytes_opt_mb']:.1f} "
              f"(-{100 * res['bytes_saving']:.1f}%) cost {res['cost_base']:.2f}->{res['cost_opt']:.2f} "
              f"(-{100 * res['cost_saving']:.1f}%) cache_hit={res['cache_hit_rate']:.2f}", flush=True)
    out = Path(args.output_dir)
    write_json(out / "network_cost.json", results)
    with open(out / "network_cost.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "final_acc", "MB_base", "MB_opt", "bytes_saving", "cost_base", "cost_opt",
                    "cost_saving", "cache_hit_rate"])
        for r in results:
            w.writerow([r["variant"], round(r["final_acc"], 4), round(r["bytes_base_mb"], 2),
                        round(r["bytes_opt_mb"], 2), round(r["bytes_saving"], 4), round(r["cost_base"], 3),
                        round(r["cost_opt"], 3), round(r["cost_saving"], 4), round(r["cache_hit_rate"], 3)])
    if results and results[0]["variant"] == "baseline":
        plot(results, out / "network_cost.png")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
