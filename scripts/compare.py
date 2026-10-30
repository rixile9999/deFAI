"""Summarise results/*/history.json: one row per configuration.

Runs whose configuration differs only in ``seed`` (and ``output-dir``) are grouped, so repeated
runs show mean ± std over seeds.

usage: python scripts/compare.py results/*/history.json
       python scripts/compare.py results/*/history.json --target 0.95 --csv table.csv
       python scripts/compare.py results/*/history.json --extra eval_acc_min comm-mb-total --markdown
       python scripts/compare.py results/*/history.json --tail 3

columns: runs, final / best test accuracy, first round reaching --target accuracy (mean over the
runs that reached it, "-" if none did) and, for differentially private runs, the final epsilon.
``--extra`` adds one column per history key with its last recorded value (mean ± std over seeds).
``--tail K`` adds the test accuracy averaged over the last K rounds, which is steadier than the final round
when the accuracy fluctuates from round to round (typical for non-IID data).
"""
import argparse
import csv
import json
import statistics
import sys

IGNORED = ("seed", "output-dir")
IN_LABEL = ("strategy", "dataset", "partitioner", "attack", "num-malicious", "dp", "dp-side", "dp-noise-multiplier")


def load(path):
    with open(path) as f:
        data = json.load(f)
    return data["config"], data["history"], data.get("privacy")


def label(config):
    parts = [str(config["strategy"]), str(config["dataset"]), str(config["partitioner"])]
    if config.get("attack", "none") != "none" and config.get("num-malicious", 0) > 0:
        parts.append(f"{config['attack']}x{config['num-malicious']}")
    if config.get("dp", "none") != "none":
        parts.append(f"dp-{config['dp']}-{config['dp-side']}(z={config['dp-noise-multiplier']})")
    return " ".join(parts)


def last_value(history, key):
    """The key's value in the last round that has one (None if no round does)."""
    values = [r[key] for r in history if r.get(key) is not None]
    return values[-1] if values else None


def run_stats(history, target, extra=(), tail=0):
    accs = [(r["round"], r["test_acc"]) for r in history if "test_acc" in r]
    reached = next((rnd for rnd, acc in accs if rnd > 0 and acc >= target), None)
    return {"final": accs[-1][1], "best": max(a for _, a in accs), "reached": reached,
            "tail": statistics.fmean(a for _, a in accs[-tail:]) if tail else None,
            "extra": {key: last_value(history, key) for key in extra}}


def mean_std(values):
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def fmt(mean, std, runs):
    if mean is None:
        return "-"
    return f"{mean:.3f}" if runs == 1 else f"{mean:.3f} ± {std:.3f}"


def varying_settings(configs):
    """Settings that take different values across ``configs`` and are not already in the label."""
    keys = {k for c in configs for k in c if k not in IGNORED and k not in IN_LABEL}
    return sorted(k for k in keys if len({json.dumps(c.get(k)) for c in configs}) > 1)


def unique_labels(configs):
    """Label per config; configs with the same label get the settings that tell them apart."""
    base = [label(c) for c in configs]
    labels = list(base)
    for text in set(base):
        same = [i for i, b in enumerate(base) if b == text]
        if len(same) > 1:
            extra = varying_settings([configs[i] for i in same])
            for i in same:
                labels[i] = " ".join([text, *(f"{k}={configs[i].get(k)}" for k in extra)])
    return labels


def group_runs(paths):
    """Runs that differ only in ``seed`` / ``output-dir`` grouped: [{"config": ..., "runs": [(history, privacy)]}]."""
    groups = {}
    for path in paths:
        config, history, privacy = load(path)
        key = json.dumps({k: v for k, v in config.items() if k not in IGNORED}, sort_keys=True)
        groups.setdefault(key, {"config": config, "runs": []})["runs"].append((history, privacy))
    return list(groups.values())


def summarise(paths, target, extra=(), tail=0):
    groups = group_runs(paths)
    for group in groups:
        group["runs"] = [(run_stats(history, target, extra, tail), privacy) for history, privacy in group["runs"]]
    names = unique_labels([g["config"] for g in groups])
    rows = []
    for name, group in zip(names, groups):
        runs = group["runs"]
        finals = [s["final"] for s, _ in runs]
        bests = [s["best"] for s, _ in runs]
        tails = [s["tail"] for s, _ in runs if s["tail"] is not None]
        reached = [s["reached"] for s, _ in runs if s["reached"] is not None]
        eps = [float("inf") if p.get("epsilon_unbounded") else p["epsilon"] for _, p in runs if p]
        extras = {}
        for key in extra:
            values = [s["extra"][key] for s, _ in runs if s["extra"][key] is not None]
            extras[key] = fmt(*mean_std(values), len(values))
        rows.append({
            **extras,
            "run": name,
            "runs": len(runs),
            "final": fmt(*mean_std(finals), len(runs)),
            "best": fmt(*mean_std(bests), len(runs)),
            "tail": fmt(*mean_std(tails), len(tails)),
            "rounds_to_target": "-" if not reached else f"{statistics.fmean(reached):.1f} ({len(reached)}/{len(runs)})",
            "epsilon": "-" if not eps else f"{statistics.fmean(eps):.4g}",
        })
    return sorted(rows, key=lambda r: r["run"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--target", type=float, default=0.9, help="accuracy for the rounds-to-target column")
    p.add_argument("--extra", nargs="+", default=[], metavar="KEY", help="history keys to add as columns (last value)")
    p.add_argument("--tail", type=int, default=0, metavar="K", help="add the test accuracy averaged over the last K rounds")
    p.add_argument("--markdown", action="store_true", help="print a Markdown table")
    p.add_argument("--csv", help="also write the table to this CSV file")
    args = p.parse_args(argv)

    rows = summarise(args.files, args.target, args.extra, args.tail)
    columns = ["run", "runs", "final", *(["tail"] if args.tail else []), "best", "rounds_to_target", "epsilon", *args.extra]
    headers = {"run": "run", "runs": "runs", "final": "final acc", "tail": f"last-{args.tail} mean", "best": "best acc",
               "rounds_to_target": f"rounds to {args.target:g}", "epsilon": "epsilon", **{k: k for k in args.extra}}
    widths = {c: max(len(headers[c]), *(len(str(r[c])) for r in rows)) for c in columns}
    if args.markdown:
        print("| " + " | ".join(headers[c].ljust(widths[c]) for c in columns) + " |")
        print("|" + "|".join(" " + "-" * widths[c] + " " for c in columns) + "|")
        for r in rows:
            print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in columns) + " |")
    else:
        print("  ".join(headers[c].ljust(widths[c]) for c in columns))
        for r in rows:
            print("  ".join(str(r[c]).ljust(widths[c]) for c in columns))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
