"""Show how a partitioning splits the classes across clients.

usage: python scripts/partition_stats.py --partitioner dirichlet --alpha 0.1 --num-partitions 10
       python scripts/partition_stats.py --dataset cifar10 --partitioner dirichlet --plot results/partition.png

Prints one row per client (examples and per-class counts) and the mean share of each client's
most frequent class: 0.1 for a perfectly uniform split of 10 classes, towards 1.0 when clients
hold almost a single class. Use it to pick a dirichlet-alpha for an experiment.
"""
import argparse
import sys


from defai.task import DATASETS, NUM_CLASSES, label_skew, partition_label_counts


def format_table(counts):
    lines = ["client  examples  " + " ".join(f"{c:>5d}" for c in range(NUM_CLASSES)) + "  max-share"]
    for i, row in enumerate(counts):
        lines.append(f"{i:>6d}  {row.sum():>8d}  " + " ".join(f"{n:>5d}" for n in row) + f"  {row.max() / row.sum():>9.2f}")
    lines.append(f"mean max-class share: {label_skew(counts):.2f}  (uniform = {1 / NUM_CLASSES:.2f})")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="synthetic", choices=sorted(DATASETS))
    p.add_argument("--partitioner", default="dirichlet", choices=["iid", "dirichlet"])
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (smaller = more skewed)")
    p.add_argument("--num-partitions", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", help="also save a client x class heat map to this PNG file")
    args = p.parse_args(argv)

    cfg = {"dataset": args.dataset, "partitioner": args.partitioner, "dirichlet-alpha": args.alpha, "seed": args.seed}
    counts = partition_label_counts(args.num_partitions, cfg)
    print(f"dataset={args.dataset} partitioner={args.partitioner}"
          + (f" alpha={args.alpha}" if args.partitioner == "dirichlet" else "")
          + f" clients={args.num_partitions} seed={args.seed}")
    print(format_table(counts))

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 0.4 * args.num_partitions + 1.5))
        image = ax.imshow(counts / counts.sum(axis=1, keepdims=True), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("class")
        ax.set_ylabel("client")
        fig.colorbar(image, label="share of the client's examples")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"saved: {args.plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
