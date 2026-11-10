import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "src")))

import numpy as np
import flwr as fl
from flwr.server import ServerConfig

from defai_addons.common import data as D
from defai_addons.fl import flower as FL
from defai_addons.fl.algorithms import evaluate
from defai_addons.monitor.deploy import MonitoringStrategy
from defai_addons.fl.models import MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default="127.0.0.1:8080")
    ap.add_argument("--algorithm", default="fedavg", choices=["fedavg", "fedprox", "qfedavg"])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dashboard", default=None, help="시각화 도구 주소 (예: http://127.0.0.1:5000)")
    a = ap.parse_args()

    _, _, Xte, yte = D.load_split("mnist")
    model = MLP(784, 10, hidden=[a.hidden])
    keys = FL.state_keys(model)

    def evaluate_global(nd):
        FL.set_ndarrays(model, keys, nd)
        m, pred = evaluate(model, Xte, yte, 10, return_pred=True)
        m["class_acc"] = [float((pred[yte == k] == k).mean()) for k in range(10)]
        return m

    fit_cfg = lambda rnd: {"local_epochs": a.local_epochs, "lr": a.lr, "batch_size": 32, "seed": rnd}
    base = FL.make_strategy(a.algorithm, FL.get_ndarrays(model, keys), mu=a.mu, q=a.q, lr=a.lr, on_fit_config=fit_cfg)
    base.min_fit_clients = base.min_available_clients = a.n_clients
    strategy = MonitoringStrategy(base, evaluate_global, a.dashboard, name="flower-%s" % a.algorithm,
                                  config={"algorithm": a.algorithm, "rounds": a.rounds, "n_clients": a.n_clients,
                                          "dataset": "mnist", "partition": "label_groups", "framework": "flwr"})
    fl.server.start_server(server_address=a.address, config=ServerConfig(num_rounds=a.rounds), strategy=strategy)
    strategy.finish()
    if strategy.history:
        h = strategy.history[-1]["global"]
        print("최종 전역 모델 (0~9 전체): acc %.4f  F1 %.4f" % (h["accuracy"], h["f1"]))


if __name__ == "__main__":
    main()
