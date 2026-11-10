import json
import time
import urllib.request
from typing import Callable, Dict, List, Optional

import numpy as np
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import Strategy

from ..common import data as D
from ..common.metrics import data_ratio, entropy, fairness, normalized_entropy
from ..fl.recorder import _jsonable


def _post(url: str, body: Dict) -> Dict:
    req = urllib.request.Request(url, data=json.dumps(_jsonable(body)).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def mnist_client_data(cid: int, n_clients: int, classes_per_client: int = 2, seed: int = 0,
                      max_train: Optional[int] = None) -> Dict[str, np.ndarray]:
    Xtr, ytr, _, _ = D.load_split("mnist", seed=seed, max_train=max_train)
    parts = D.partition(ytr, n_clients, "label_groups", seed=seed, classes_per_client=classes_per_client)
    return D.split_clients(Xtr, ytr, parts, val_ratio=0.2, seed=seed)[cid]


class MonitoringStrategy(Strategy):
    def __init__(self, base: Strategy, evaluate_global: Callable[[List[np.ndarray]], Dict[str, object]],
                 dashboard: Optional[str] = None, name: str = "flower", config: Optional[Dict] = None,
                 n_classes: int = 10):
        self.base, self.evaluate_global = base, evaluate_global
        self.dashboard = dashboard.rstrip("/") if dashboard else None
        self.name, self.config, self.n_classes = name, config or {}, n_classes
        self.run_id: Optional[str] = None
        self.clients_seen: Dict[str, Dict] = {}
        self.t_round = time.time()
        self.history: List[Dict] = []

    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        self.t_round = time.time()
        return self.base.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        return self.base.evaluate(server_round, parameters)

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = self.base.aggregate_fit(server_round, results, failures)
        if params is not None:
            try:
                self._record(server_round, params, results, failures)
            except Exception as e:
                print("[monitor] 전송 실패:", e)
        return params, metrics

    def _ensure_run(self, results):
        if self.run_id or not self.dashboard:
            return
        n_total = max(int(r.metrics.get("n_clients", len(results))) for _, r in results)
        counts = [[0] * self.n_classes for _ in range(n_total)]
        for _, r in results:
            k = int(r.metrics["client_id"])
            counts[k] = json.loads(r.metrics.get("class_counts", "[]")) or counts[k]
        n = [sum(c) for c in counts]
        ratio = data_ratio(n) if sum(n) else np.zeros(n_total)
        clients = [{"id": i, "n": n[i], "ratio": float(ratio[i]), "entropy": entropy(counts[i]) if n[i] else 0.0,
                    "norm_entropy": normalized_entropy(counts[i]) if n[i] else 0.0, "class_counts": counts[i],
                    "malicious": False} for i in range(n_total)]
        topology = {"server": {"id": "server"}, "edges": [{"id": 0, "latency_ms": 0.0}], "n_classes": self.n_classes,
                    "model_bytes": int(sum(np.asarray(a).nbytes for a in parameters_to_ndarrays(results[0][1].parameters))),
                    "framework": "flwr (gRPC)"}
        self.n_total = n_total
        self.run_id = _post(self.dashboard + "/api/ext/runs", {"name": self.name, "config": self.config,
                                                               "topology": topology, "clients": clients})["id"]
        print("[monitor] 대시보드 실행 id:", self.run_id)

    def _record(self, server_round, params, results, failures):
        nd = parameters_to_ndarrays(params)
        g = self.evaluate_global(nd)
        mbytes = int(sum(a.nbytes for a in nd))
        self._ensure_run(results)
        n_total = getattr(self, "n_total", len(results))
        crec = [{"id": i, "edge": 0, "participated": False, "status": "idle", "n": 0, "weight": 0.0} for i in range(n_total)]
        tot = sum(r.num_examples for _, r in results)
        for _, r in results:
            m = r.metrics
            k = int(m["client_id"])
            crec[k].update({"participated": True, "status": "ok", "n": r.num_examples,
                            "train_loss": m.get("train_loss"), "val_acc": m.get("val_acc"),
                            "loss_before": m.get("loss_before"), "compute_ms": m.get("compute_ms"),
                            "latency_ms": m.get("compute_ms"), "bytes_up": mbytes, "bytes_down": mbytes,
                            "weight": r.num_examples / tot, "quality": m.get("val_acc"),
                            "global_acc": m.get("val_acc")})
        rec = {"round": server_round, "wall_s": time.time() - self.t_round, "sim_time_ms": (time.time() - self.t_round) * 1e3,
               "global": {k: g[k] for k in ("accuracy", "precision", "recall", "f1", "loss")},
               "fairness": fairness([c["val_acc"] for c in crec if c.get("val_acc") is not None]),
               "class_acc": g.get("class_acc"), "participants": len(results), "accepted": len(results),
               "bytes_total": 2 * mbytes * len(results), "resource": {}, "rematched": False,
               "clients": crec, "edges": [{"id": 0, "assigned": n_total, "accepted": len(results), "timeouts": len(failures),
                                           "outliers": 0, "latency_ms": 0.0, "bytes_up": mbytes,
                                           "status": "ok" if not failures else "degraded"}]}
        self.history.append(rec)
        print("[server] round %d  acc %.4f  f1 %.4f  clients %d" % (server_round, g["accuracy"], g["f1"], len(results)))
        if self.dashboard:
            _post(self.dashboard + "/api/ext/runs/%s/rounds" % self.run_id, rec)

    def finish(self):
        if self.dashboard and self.run_id:
            _post(self.dashboard + "/api/ext/runs/%s/finish" % self.run_id, {"summary": {"framework": "flwr"}})
