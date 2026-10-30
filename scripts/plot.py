"""Plot a metric per round from one or more results/*/history.json files.

usage: python scripts/plot.py results/*/history.json -o results/acc.png
       python scripts/plot.py results/*/history.json -m dp-epsilon -o results/eps.png
       python scripts/plot.py results/*/history.json --group -o results/acc.png

Any key of a history row can be plotted, e.g. test_acc (default), test_loss, eval_acc,
train_loss, dp-epsilon, dp-clip-norm. Rounds without a (finite) value for the metric are skipped.
With --group, runs that differ only in the seed (see scripts/compare.py) become one curve: the mean
over the seeds with a band of one standard deviation.
"""
import argparse
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")  # File output works without a desktop/Tk installation.
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402


def per_round_values(histories, metric):
    """{round: [value of every run that has one]}; null = infinite / missing."""
    values = {}
    for history in histories:
        for row in history:
            if row.get(metric) is not None:
                values.setdefault(row["round"], []).append(row[metric])
    return values


def plot_groups(files, metric):
    groups = compare.group_runs(files)
    for label, group in zip(compare.unique_labels([g["config"] for g in groups]), groups):
        values = per_round_values([history for history, _ in group["runs"]], metric)
        rounds = sorted(values)
        mean = [statistics.fmean(values[r]) for r in rounds]
        std = [statistics.stdev(values[r]) if len(values[r]) > 1 else 0.0 for r in rounds]
        (line,) = plt.plot(rounds, mean, label=f"{label} (n={len(group['runs'])})")
        plt.fill_between(rounds, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                         color=line.get_color(), alpha=0.15, linewidth=0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", default="results/acc.png")
    p.add_argument("-m", "--metric", default="test_acc")
    p.add_argument("--group", action="store_true", help="one curve per configuration: mean ± std over the seeds")
    p.add_argument("--title", help="figure title")
    args = p.parse_args(argv)

    if args.group:
        plot_groups(args.files, args.metric)
    else:
        for path in args.files:
            with open(path) as f:
                h = json.load(f)["history"]
            label = os.path.basename(os.path.dirname(path))
            rows = [x for x in h if x.get(args.metric) is not None]  # null = infinite / missing
            plt.plot([x["round"] for x in rows], [x[args.metric] for x in rows], label=label)

    plt.xlabel("Round")
    plt.ylabel(args.metric)
    if args.title:
        plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    figure = plt.gcf()
    plt.close(figure)
    print(f"saved: {args.out}")
    return figure


if __name__ == "__main__":
    main()
