import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "src")))

import numpy as np
import torch
import flwr as fl

from defai_addons.monitor.deploy import mnist_client_data
from defai_addons.fl.models import MLP
from defai_addons.monitor.simulator import MNISTClient


class DeployedClient(MNISTClient):
    def __init__(self, *a, n_clients: int, **kw):
        super().__init__(*a, **kw)
        self.n_clients = n_clients

    def fit(self, parameters, config):
        new, n, m = super().fit(parameters, config)
        m.update({"client_id": self.cid, "n_clients": self.n_clients,
                  "class_counts": json.dumps(np.bincount(self.data["y"], minlength=10).tolist())})
        return new, n, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--malicious", action="store_true", help="부호 반전 공격 클라이언트로 동작")
    a = ap.parse_args()
    torch.manual_seed(0)
    data = mnist_client_data(a.cid, a.n_clients)
    print("client %d: 숫자 %s, 학습 %d / 검증 %d" % (a.cid, sorted(set(data["y"].tolist())), len(data["y"]), len(data["yv"])))
    client = DeployedClient(a.cid, data, MLP(784, 10, hidden=[a.hidden]), 10, malicious=a.malicious,
                            n_clients=a.n_clients)
    fl.client.start_client(server_address=a.server, client=client.to_client())


if __name__ == "__main__":
    main()
