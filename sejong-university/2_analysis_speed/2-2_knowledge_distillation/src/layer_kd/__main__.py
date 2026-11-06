import argparse
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord

from fl_core.backends import get_backend
from fl_core.config import parse_sets
from fl_core.runner import final_accuracy, quiet, run_federated, write_json
from fl_core.task import test, train
from graph_pruning.dynamic import count_params, latency_ms
from layer_kd.federated import TeacherStrategy, make_kd_handler
from layer_kd.trainer import METHODS, KDConfig, KDTrainer


def train_teachers(backend, count, epochs, seed=0):
    x, y = backend.client_data(0, 1)
    teachers = []
    for k in range(count):
        torch.manual_seed(seed + k)
        t = backend.build_model(width=1.0)
        train(t, x, y, np.arange(len(y)), epochs=epochs, lr=0.05, seed=seed + k)
        teachers.append(t.eval())
    return teachers


def subset(x, y, fraction, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))[: max(64, int(fraction * len(y)))]
    t = torch.as_tensor(np.sort(idx))
    return x[t], y[t]


def cmd_central(args):
    quiet()
    b = get_backend(args.backend, parse_sets(args.set))
    x_test, y_test = b.test_set()
    teachers = train_teachers(b, max(1, args.teachers), args.teacher_epochs)
    shape = b.input_shape()
    rows = []
    for k, t in enumerate(teachers):
        rows.append({"model": f"teacher{k}", "params": count_params(t), "latency_ms": latency_ms(t, shape),
                     "test_acc": test(t, x_test, y_test)[1]})
        print(rows[-1], flush=True)
    x, y = subset(*b.client_data(0, 1), args.student_data)
    for method in args.methods.split(","):
        torch.manual_seed(0)
        student = b.build_model(width=args.student_width)
        cfg = KDConfig(method=method)
        trainer = KDTrainer(cfg, teachers if method == "ensemble" else teachers[:1], student, shape)
        logs, steps = trainer.fit(x, y, epochs=args.epochs, lr=args.lr)
        trainer.close()
        acc = test(student, x_test, y_test)[1]
        rows.append({"model": f"student_{method}", "params": count_params(student),
                     "latency_ms": latency_ms(student, shape), "test_acc": acc,
                     "size_ratio": count_params(student) / count_params(teachers[0]),
                     **{f"loss_{k}": v for k, v in logs.items()}})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    write_json(Path(args.output_dir) / "kd_central.json", rows)


def cmd_federated(args):
    quiet()
    b = get_backend(args.backend, parse_sets(args.set))
    teacher = train_teachers(b, 1, args.teacher_epochs)[0]
    x_test, y_test = b.test_set()
    t_arrays = ArrayRecord(teacher.state_dict())
    print({"teacher_acc": test(teacher, x_test, y_test)[1]}, flush=True)
    results = []
    for method in args.methods.split(","):
        handler = make_kd_handler(b, KDConfig(method=method), args.student_width)
        grid = b.grid(args.clients, train=handler, evaluate=b.custom_evaluate_handler(
            model_fn=lambda cfg: b.build_model(width=args.student_width)))
        strategy = b.build_strategy("fedavg")
        if method != "student":
            strategy = TeacherStrategy(strategy, t_arrays)
        torch.manual_seed(0)
        student = b.build_model(width=args.student_width)
        out = run_federated(b, strategy, args.clients, args.rounds, grid=grid, model=student)
        row = {"method": method, "final_acc": final_accuracy(out["history"]),
               "P_last3": final_accuracy(out["history"], 3), "params": count_params(student),
               "curve": [r.get("test_acc") for r in out["history"]]}
        results.append(row)
        print({k: v for k, v in row.items() if k != "curve"}, [round(a, 3) for a in row["curve"]], flush=True)
    write_json(Path(args.output_dir) / "kd_federated.json", results)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m layer_kd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default="builtin")
    common.add_argument("--set", action="append", default=[])
    common.add_argument("--methods", default=",".join(METHODS))
    common.add_argument("--student-width", type=float, default=0.25)
    common.add_argument("--teacher-epochs", type=float, default=3.0)
    common.add_argument("--output-dir", default="results/layer_kd")
    c = sub.add_parser("central", parents=[common])
    c.add_argument("--teachers", type=int, default=2)
    c.add_argument("--student-data", type=float, default=0.2)
    c.add_argument("--epochs", type=float, default=3.0)
    c.add_argument("--lr", type=float, default=0.02)
    c.set_defaults(fn=cmd_central)
    f = sub.add_parser("federated", parents=[common])
    f.add_argument("--clients", type=int, default=6)
    f.add_argument("--rounds", type=int, default=5)
    f.set_defaults(fn=cmd_federated)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
