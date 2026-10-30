"""Two-round, three-client Flower smoke test using training patients only."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import flwr as fl
import numpy as np
import torch
from runtime import DEVICE, device_name, require_device

from run_federated_study import ROOT, MultiModalNet, make_tensors, local_update
from src.server import NeonatalFedAvg


class CheckClient(fl.client.NumPyClient):
    def __init__(self, client):
        self.data = np.load(ROOT / "data/real_data.npz")
        self.tensors = make_tensors(self.data)
        self.indices = np.flatnonzero((self.data["split"] == "train") & (self.data["client"] == client))[:64]
        self.model = MultiModalNet(pretrained=False).to(DEVICE)
        self.client = client

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def fit(self, parameters, config):
        reference = self.model.state_dict()
        self.model.load_state_dict(OrderedDict((key, torch.as_tensor(array, dtype=value.dtype))
                                               for (key, value), array in zip(reference.items(), parameters)))
        local_update(self.model, self.tensors, self.indices, None, None, 0, 32, 1, 100+self.client)
        print(f"device={device_name()} client={self.client} training_images={len(self.indices)}", flush=True)
        return self.get_parameters({}), len(self.indices), {}


class CheckStrategy(NeonatalFedAvg):
    def aggregate_fit(self, server_round, results, failures):
        value = super().aggregate_fit(server_round, results, failures)
        print(f"AGGREGATED_ROUND {server_round} CLIENTS {len(results)}", flush=True)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["server", "client", "check"], default="check")
    parser.add_argument("--port", type=int)
    parser.add_argument("--client", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.role == "server":
        torch.manual_seed(100)
        model = MultiModalNet()
        strategy = CheckStrategy(initial_parameters=fl.common.ndarrays_to_parameters([v.numpy() for v in model.state_dict().values()]),
                                 min_fit_clients=3, min_available_clients=3, fraction_evaluate=0,
                                 min_evaluate_clients=0, accept_failures=False)
        fl.server.start_server(server_address=f"127.0.0.1:{args.port}",
                               config=fl.server.ServerConfig(num_rounds=2, round_timeout=120), strategy=strategy)
        return
    if args.role == "client":
        fl.client.start_client(server_address=f"127.0.0.1:{args.port}", client=CheckClient(args.client).to_client())
        return
    output = ROOT / "results/network"; output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    processes, handles = [], []
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    def launch(name, arguments):
        handle = (output / f"{name}.log").open("w", encoding="utf-8"); handles.append(handle)
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *arguments, "--port", str(port)],
                                   cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        processes.append(process)
    try:
        launch("server", ["--role", "server"])
        deadline = time.monotonic()+60
        while True:
            if processes[0].poll() is not None:
                raise RuntimeError("Server failed; see network/server.log")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Server did not start")
                time.sleep(.2)
        for client in range(3):
            launch(f"client_{client}", ["--role", "client", "--client", str(client)])
        for process in processes:
            if process.wait(timeout=180):
                raise RuntimeError("Flower process failed; inspect network logs")
        text = (output / "server.log").read_text(encoding="utf-8")
        if "AGGREGATED_ROUND 2 CLIENTS 3" not in text:
            raise AssertionError("Two complete three-client rounds were not observed")
        result = {"clients": 3, "processes": 4, "rounds": 2, "all_exit_codes_zero": True,
                  "device": device_name(), "data": "64 training images/client; smoke test, not reported performance"}
        (output / "check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(result)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait(timeout=10)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
