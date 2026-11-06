import csv
import json
from pathlib import Path

import numpy as np

from fl_core.runner import quiet, write_json
from fl_core.task import test
from hierarchical_fl.server import HierarchicalServer
from scenario_test.scenarios import SCENARIOS, ScenarioPlugin


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    if v.sum() <= 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def contributions(history, n):
    score = np.zeros(n)
    for c in history:
        score[c["cid"]] += c.get("n", 0) * max(0.0, 2 * c.get("q", 0.5) - 1)
    total = score.sum()
    return (score / total).tolist() if total > 0 else score.tolist()


def per_client_accuracy(server):
    accs = []
    for cid in range(server.n):
        x, y = server.backend.client_data(cid, server.n, "val")
        accs.append(test(server.model, x, y)[1] if len(y) else float("nan"))
    return accs


def run_scenario(kind, base=None, seed=0, output_dir="results/scenarios", extra=None):
    quiet()
    base = dict(base or {})
    bc = dict(base.get("backend-config") or {}, seed=seed)
    cfg = dict(base, **{"backend-config": bc, "run-id": f"scn_{bc.get('dataset', 'synthetic')}_{kind}_s{seed}",
                        "output-dir": str(Path(output_dir) / "runs")})
    plugin = ScenarioPlugin(dict(extra or {}, scenario=kind))
    server = HierarchicalServer(cfg, plugins=[plugin])
    out = server.run()
    accs = per_client_accuracy(server)
    contrib = contributions(server.client_history, server.n)
    rows = out["rounds"]
    return {"scenario": kind, "title": SCENARIOS[kind]["title"], "seed": seed, "dataset": server.cfg["dataset"],
            "P": out["P_last3"], "final_acc": out["final_acc"], "rounds": rows,
            "virtual_ms": [r.get("virtual_ms") for r in rows if r["round"] > 0],
            "participants": [r.get("matched") for r in rows if r["round"] > 0],
            "late": [r.get("late") for r in rows if r["round"] > 0],
            "client_acc": accs, "client_acc_min": float(np.nanmin(accs)), "client_acc_std": float(np.nanstd(accs)),
            "contribution": contrib, "contribution_gini": gini(contrib),
            "n_samples": server.n_samples.tolist(), "alerts": len(out["alerts"]),
            "trace": plugin.trace, "run_dir": str(server.run_dir), "completed": len(rows) - 1 == cfg.get(
                "global-rounds", server.cfg["global-rounds"])}


def adaptability(results):
    ref = {(r["dataset"], r["seed"]): r["P"] for r in results if r["scenario"] == "ref"}
    ratios = {}
    for r in results:
        if r["scenario"] == "ref" or (r["dataset"], r["seed"]) not in ref:
            continue
        ratios.setdefault(r["scenario"], []).append(r["P"] / ref[(r["dataset"], r["seed"])])
    per = {k: float(np.mean(v)) for k, v in ratios.items()}
    allv = [x for v in ratios.values() for x in v]
    return {"A_adapt": float(np.mean(allv)) if allv else None, "per_scenario": per,
            "scenarios_completed": sorted({r["scenario"] for r in results if r["scenario"] != "ref" and r["completed"]})}


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for r in results:
        rounds = [x["round"] for x in r["rounds"]]
        axes[0].plot(rounds, [x.get("test_acc") for x in r["rounds"]], label=f"{r['scenario']} s{r['seed']}")
        axes[1].plot(rounds[1:], np.asarray(r["virtual_ms"], dtype=float) / 1000, label=r["scenario"])
        axes[2].plot(rounds[1:], r["participants"], label=r["scenario"])
    axes[0].set(title="test accuracy", xlabel="round")
    axes[1].set(title="virtual round time (s)", xlabel="round")
    axes[2].set(title="clients aggregated", xlabel="round")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_suite(scenarios=None, seeds=(0,), base=None, output_dir="results/scenarios", log=print):
    scenarios = list(scenarios or SCENARIOS)
    if "ref" not in scenarios:
        scenarios = ["ref"] + scenarios
    results = []
    for seed in seeds:
        for kind in scenarios:
            r = run_scenario(kind, base, seed, output_dir)
            results.append(r)
            log(f"{kind:14s} seed={seed} P={r['P']:.4f} final={r['final_acc']:.4f} "
                f"client_min={r['client_acc_min']:.3f} gini={r['contribution_gini']:.3f} "
                f"virtual_s={np.nansum(np.asarray(r['virtual_ms'], dtype=float)) / 1000:.1f}")
    summary = adaptability(results)
    out = Path(output_dir)
    write_json(out / "scenario_results.json", {"results": results, "summary": summary})
    with open(out / "scenario_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "dataset", "P", "P/P_ref", "client_acc_min", "client_acc_std",
                    "contribution_gini", "virtual_s", "alerts"])
        ref = {(r["dataset"], r["seed"]): r["P"] for r in results if r["scenario"] == "ref"}
        for r in results:
            w.writerow([r["scenario"], r["seed"], r["dataset"], round(r["P"], 4),
                        round(r["P"] / ref[(r["dataset"], r["seed"])], 4), round(r["client_acc_min"], 4),
                        round(r["client_acc_std"], 4), round(r["contribution_gini"], 4),
                        round(float(np.nansum(np.asarray(r["virtual_ms"], dtype=float))) / 1000, 1), r["alerts"]])
    plot(results, out / "scenario_curves.png")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return results, summary
