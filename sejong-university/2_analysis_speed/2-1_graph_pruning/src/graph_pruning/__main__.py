import argparse
from pathlib import Path

import numpy as np
import torch

from fl_core.backends import get_backend
from fl_core.config import parse_sets
from fl_core.plugins import run_flat
from fl_core.runner import quiet, write_json
from fl_core.task import test, train
from graph_pruning.dynamic import count_macs, count_params, latency_ms, prune_dynamic, uniform_l1_matched
from graph_pruning.federated import GraphPruningPlugin
from graph_pruning.graph import build_graph, check_model, save_pruned
from graph_pruning.sparsity import bn_l1, bn_scale_stats


def cost_row(name, model, input_shape, x, y):
    loss, acc = test(model, x, y)
    return {"model": name, "params": count_params(model), "macs": count_macs(model, input_shape),
            "latency_ms": latency_ms(model, input_shape), "test_loss": loss, "test_acc": acc}


def cmd_graph(args):
    b = get_backend(args.backend, parse_sets(args.set))
    model = b.build_model()
    print(build_graph(model, b.input_shape()).describe())


def cmd_central(args):
    quiet()
    b = get_backend(args.backend, parse_sets(args.set))
    x, y = b.client_data(0, 1)
    x_test, y_test = b.test_set()
    half = len(y_test) // 2
    xv, yv, xt, yt = x_test[:half], y_test[:half], x_test[half:], y_test[half:]
    torch.manual_seed(0)
    model = b.build_model()
    extra = (lambda m, xb, yb, out: bn_l1(m, args.bn_l1)) if args.bn_l1 > 0 else None
    train(model, x, y, np.arange(len(y)), epochs=args.epochs, lr=args.lr, extra_loss=extra)
    shape = b.input_shape()
    rows = [cost_row("base", model, shape, xt, yt)]
    print(rows[-1], bn_scale_stats(model))

    def recover(m, steps, chunk):
        train(m, x, y, np.arange(len(y)), epochs=steps * 32 / len(y), lr=args.lr / 2, seed=1000 + chunk)

    res = prune_dynamic(model, recover, lambda m: test(m, xv, yv), shape, percentile=args.percentile, eta=args.eta,
                        tol=args.tol, max_iters=args.iters, recovery_steps=args.recovery_steps,
                        importance=args.importance, target=args.target,
                        batches=[(xv[i:i + 128], yv[i:i + 128]) for i in range(0, 512, 128)])
    pruned = res["model"]
    check_model(pruned, shape)
    rows.append(cost_row("graph_dynamic", pruned, shape, xt, yt))
    base_l1 = uniform_l1_matched(model, shape, count_params(pruned))
    recover(base_l1, args.recovery_steps * max(1, sum(r.get("accepted", 0) for r in res["history"][1:])), 0)
    rows.append(cost_row("uniform_l1", base_l1, shape, xt, yt))
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    out = Path(args.output_dir)
    save_pruned(pruned, str(out / "pruned_model.pt"), {"history": res["history"]})
    write_json(out / "central_pruning.json", {"rows": rows, "history": res["history"], "stop": res["stop"]})


def cmd_federated(args):
    quiet()
    b = get_backend(args.backend, parse_sets(args.set))
    plugin = GraphPruningPlugin({"start": args.start, "every": args.every, "percentile": args.percentile,
                                 "eta": args.eta, "tol": args.tol, "importance": args.importance,
                                 "bn-l1": args.bn_l1, "target": args.target})
    out = run_flat(b, [plugin], args.clients, args.rounds)
    strat = plugin.strategy
    masked, compact = strat.final_models(out["result"].arrays)
    x_test, y_test = b.test_set()
    shape = b.input_shape()
    torch.manual_seed(0)
    full = b.build_model()
    full.load_state_dict(out["result"].arrays.to_torch_state_dict())
    rows = [cost_row("federated_masked", masked, shape, x_test, y_test),
            cost_row("federated_compact", compact, shape, x_test, y_test)]
    rows[0]["params"] = count_params(full)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("rounds:", [(r["round"], round(r.get("test_acc", 0), 4), round(r.get("prune-params-red", 0), 3))
                      for r in out["rows"]])
    outdir = Path(args.output_dir)
    save_pruned(compact, str(outdir / "federated_pruned_model.pt"))
    write_json(outdir / "federated_pruning.json", {"rows": rows, "rounds": out["rows"], "events": strat.history})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m graph_pruning")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default="builtin")
    common.add_argument("--set", action="append", default=[])
    common.add_argument("--percentile", type=float, default=30.0)
    common.add_argument("--eta", type=float, default=0.5)
    common.add_argument("--tol", type=float, default=0.15)
    common.add_argument("--importance", choices=("path", "taylor"), default="path")
    common.add_argument("--bn-l1", type=float, default=0.0)
    common.add_argument("--target", type=float, default=None)
    common.add_argument("--output-dir", default="results/graph_pruning")
    g = sub.add_parser("graph", parents=[common])
    g.set_defaults(fn=cmd_graph)
    c = sub.add_parser("central", parents=[common])
    c.add_argument("--epochs", type=float, default=2.0)
    c.add_argument("--lr", type=float, default=0.05)
    c.add_argument("--iters", type=int, default=6)
    c.add_argument("--recovery-steps", type=int, default=60)
    c.set_defaults(fn=cmd_central)
    f = sub.add_parser("federated", parents=[common])
    f.add_argument("--clients", type=int, default=8)
    f.add_argument("--rounds", type=int, default=8)
    f.add_argument("--start", type=int, default=2)
    f.add_argument("--every", type=int, default=2)
    f.set_defaults(fn=cmd_federated)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
