import argparse

import numpy as np
from defai.strategies.base import FedAvg

from defai_research.aggregation import map_reduce


class MapReduceFedAvg(FedAvg):
    name = "mapreduce"

    def __init__(self, *, groups=4, block_size=256, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.block_size = block_size
        self.last_error = None

    def aggregate_updates(self, updates):
        keys = list(updates[0].delta)
        flat = [np.concatenate([u.delta[k].ravel() for k in keys]) for u in updates]
        weights = [max(1, int(round(u.weight))) for u in updates]
        groups = [self._rank.get(u.node_id, i) % self.groups for i, u in enumerate(updates)]
        result, _, _, self.last_error = map_reduce(flat, weights, groups, self.block_size)
        out, pos = {}, 0
        for k in keys:
            shape = updates[0].delta[k].shape
            size = int(np.prod(shape)) if shape else 1
            out[k] = result[pos:pos + size].reshape(shape)
            pos += size
        self._round_metrics["mapreduce-max-abs-error"] = float(self.last_error)
        return out


def main(argv=None):
    from fl_core.backends import get_backend
    from fl_core.plugins import run_flat

    ap = argparse.ArgumentParser(prog="python examples/kist_mapreduce_strategy.py")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--groups", type=int, default=4)
    args = ap.parse_args(argv)
    backend = get_backend("kist", {"dataset": "synthetic"})
    strategy = MapReduceFedAvg(groups=args.groups, fraction_train=1.0, fraction_evaluate=0.0, min_train_nodes=1,
                               min_evaluate_nodes=0, min_available_nodes=1, seed=0)
    out = run_flat(backend, [], args.clients, args.rounds, strategy=strategy)
    for row in out["rows"]:
        print(row["round"], round(row["test_acc"], 4))
    print("max |MapReduce - weighted mean| =", strategy.last_error)


if __name__ == "__main__":
    main()
