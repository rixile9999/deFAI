import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data_imbalance.plugin import ImbalancePlugin
from data_imbalance.sampling import imbalance_ratio, resample
from fl_core.backends import get_backend
from fl_core.config import merge, parse_sets
from fl_core.plugins import run_flat
from fl_core.runner import final_accuracy, quiet, write_json
from fl_core.task import predict, test, train

VARIANTS = {"fedavg": ("none", "size"), "sampling": ("combo", "size"), "weighting": ("none", "size_loss"),
            "sampling+weighting": ("combo", "size_loss")}


def class_metrics(model, x, y, num_classes=10):
    labels, preds = predict(model, x, y)
    recall, f1 = [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        r = tp / (tp + fn) if tp + fn else 0.0
        p = tp / (tp + fp) if tp + fp else 0.0
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"macro_f1": float(np.mean(f1)), "min_recall": float(np.min(recall)), "recall": recall}


def local_study(backend, clients, method, epochs=2, seed=0):
    x_test, y_test = backend.test_set()
    rows = []
    for cid in range(clients):
        x, y = backend.client_data(cid, clients)
        xv, yv = backend.client_data(cid, clients, "val")
        xs, ys = resample(x, y, method, seed=seed + cid)
        torch.manual_seed(seed)
        model = backend.build_model()
        train(model, xs, ys, np.arange(len(ys)), epochs=epochs, seed=seed)
        rows.append({"client": cid, "n_before": len(y), "n_after": len(ys), "ir_before": imbalance_ratio(y),
                     "ir_after": imbalance_ratio(ys), "local_val_acc": test(model, xv, yv)[1] if len(yv) else None,
                     "global_test_acc": test(model, x_test, y_test)[1]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m data_imbalance")
    ap.add_argument("--backend", default="builtin")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--alpha-dirichlet", type=float, default=0.3)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--local-study", action="store_true")
    ap.add_argument("--output-dir", default="results/data_imbalance")
    args = ap.parse_args(argv)
    quiet()
    bc = merge({"partitioner": "dirichlet", "dirichlet-alpha": args.alpha_dirichlet}, parse_sets(args.set))
    out = Path(args.output_dir)
    results = []
    for name in args.variants.split(","):
        sampling, weighting = VARIANTS[name]
        backend = get_backend(args.backend, bc)
        plugin = ImbalancePlugin({"sampling": sampling, "weighting": weighting})
        res = run_flat(backend, [plugin], args.clients, args.rounds)
        x_test, y_test = backend.test_set()
        cm = class_metrics(res["model"], x_test, y_test)
        row = {"variant": name, "sampling": sampling, "weighting": weighting,
               "final_acc": final_accuracy(res["rows"]), "P_last3": final_accuracy(res["rows"], 3), **cm,
               "curve": [r.get("test_acc") for r in res["rows"]], "weights": plugin.summary(None)["weight_log"]}
        results.append(row)
        print(f"{name:20s} acc={row['final_acc']:.4f} macroF1={cm['macro_f1']:.4f} minRecall={cm['min_recall']:.4f}",
              flush=True)
    local = []
    if args.local_study:
        backend = get_backend(args.backend, bc)
        for method in ("none", "combo"):
            for r in local_study(backend, args.clients, method):
                local.append(dict(r, sampling=method))
        for method in ("none", "combo"):
            rows = [r for r in local if r["sampling"] == method]
            print(f"local {method:5s} IR {np.mean([r['ir_before'] for r in rows]):.1f}->"
                  f"{np.mean([r['ir_after'] for r in rows]):.1f} local_acc={np.mean([r['local_val_acc'] for r in rows]):.4f} "
                  f"global_acc={np.mean([r['global_test_acc'] for r in rows]):.4f}")
    write_json(out / "imbalance.json", {"federated": results, "local": local})
    with open(out / "imbalance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "sampling", "weighting", "final_acc", "P_last3", "macro_f1", "min_recall"])
        for r in results:
            w.writerow([r["variant"], r["sampling"], r["weighting"], round(r["final_acc"], 4), round(r["P_last3"], 4),
                        round(r["macro_f1"], 4), round(r["min_recall"], 4)])
    print(f"saved {out}")


if __name__ == "__main__":
    main()
