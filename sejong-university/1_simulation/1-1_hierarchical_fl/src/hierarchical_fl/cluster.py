import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def free_ports(n, start):
    ports, p = [], start
    while len(ports) < n:
        if port_free(p):
            ports.append(p)
        p += 1
    return ports


class EdgeCluster:
    def __init__(self, n_edges, base_port=6101, log_dir=None, host="127.0.0.1"):
        self.n = n_edges
        self.host = host
        self.ports = free_ports(n_edges, base_port)
        self.log_dir = Path(log_dir) if log_dir else None
        self.procs = []
        self.logs = []

    @property
    def urls(self):
        return [f"http://{self.host}:{p}" for p in self.ports]

    def start(self, timeout=120.0):
        env = dict(os.environ)
        for i, port in enumerate(self.ports):
            out = subprocess.DEVNULL
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                out = open(self.log_dir / f"edge{i}.log", "w", encoding="utf-8")
                self.logs.append(out)
            self.procs.append(subprocess.Popen(
                [sys.executable, "-m", "hierarchical_fl.edge_server", "--edge-id", str(i), "--host", self.host,
                 "--port", str(port)], env=env, stdout=out, stderr=subprocess.STDOUT))
        deadline = time.time() + timeout
        for url, proc in zip(self.urls, self.procs):
            while True:
                if proc.poll() is not None:
                    self.stop()
                    raise RuntimeError(f"edge server {url} exited with code {proc.returncode}")
                try:
                    if requests.get(url + "/api/v1/health", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.time() > deadline:
                    self.stop()
                    raise TimeoutError(f"edge server {url} did not start")
                time.sleep(0.3)
        return self.urls

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in self.logs:
            f.close()
        self.procs, self.logs = [], []
