import copy
import math
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from ..common import data as D
from ..common.metrics import data_ratio, entropy, fairness, normalized_entropy
from ..common.preprocessing import Normalizer
from ..common.sysmon import ResourceSampler
from ..fl import flower as FL
from ..fl.algorithms import apply_attack, detect_update_outliers, evaluate, local_train
from .matching import GameMatcher
from ..fl.models import MLP
from ..fl.recorder import RunRecorder

DEFAULTS: Dict[str, Any] = dict(
    dataset="mnist", algorithm="fedavg", rounds=30, n_clients=20, n_edges=3,
    partition="label_groups", classes_per_client=2, alpha=0.5, quantity_sigma=0.3, max_train=None,
    local_epochs=1, lr=0.05, batch_size=32, mu=0.1, q=0.2, qL=None,
    participation=1.0, participation_mode="fixed", participation_min=0.3,
    rematch_every=5, ema_beta=0.7,
    bandwidth_mbps=50.0, base_latency_ms=5.0, device_ms_per_sample=0.02, delay_growth_ms=0.0, delay_frac=0.3, timeout_ms=None,
    n_malicious=0, attack="signflip", defense=False, outlier_z=3.5,
    hidden=[128], seed=0, sleep=0.0, sketch_dim=32,
)


def make_config(**kw) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in kw.items():
        if v is not None:
            cfg[k] = v
    if cfg["qL"] is None:
        cfg["qL"] = 1.0 / cfg["lr"]
    return cfg


class MNISTClient(NumPyClient):
    def __init__(self, cid: int, data: Dict[str, np.ndarray], model: torch.nn.Module, n_classes: int,
                 malicious: bool = False, attack: str = "signflip", seed: int = 0):
        self.cid, self.data, self.model, self.n_classes = cid, data, model, n_classes
        self.keys = FL.state_keys(model)
        self.malicious, self.attack = malicious, attack
        self.rng = np.random.RandomState(seed + 7919 * cid)

    def get_parameters(self, config):
        return FL.get_ndarrays(self.model, self.keys)

    def fit(self, parameters, config):
        FL.set_ndarrays(self.model, self.keys, parameters)
        d = self.data
        loss_before = evaluate(self.model, d["X"], d["y"], self.n_classes)["loss"]
        g = torch.as_tensor(FL.flatten(parameters), dtype=torch.float32)
        t0 = time.time()
        tr = local_train(self.model, d["X"], d["y"], int(config.get("local_epochs", 1)), float(config.get("lr", 0.05)),
                         int(config.get("batch_size", 32)), mu=float(config.get("proximal_mu", 0.0)), global_vec=g,
                         seed=int(config.get("seed", 0)) * 1000 + self.cid)
        compute_ms = (time.time() - t0) * 1e3
        new = FL.get_ndarrays(self.model, self.keys)
        if self.malicious:
            upd = torch.as_tensor(FL.flatten(new) - FL.flatten(parameters), dtype=torch.float32)
            upd = apply_attack(upd, self.attack, self.rng).double().numpy()
            new = FL.unflatten(FL.flatten(parameters) + upd, parameters)
            FL.set_ndarrays(self.model, self.keys, new)
        va = evaluate(self.model, d["Xv"], d["yv"], self.n_classes)
        return new, len(d["y"]), {"loss_before": float(loss_before), "train_loss": tr["train_loss"],
                                  "val_acc": va["accuracy"], "val_loss": va["loss"], "compute_ms": compute_ms}

    def evaluate(self, parameters, config):
        FL.set_ndarrays(self.model, self.keys, parameters)
        m = evaluate(self.model, self.data["Xv"], self.data["yv"], self.n_classes)
        return m["loss"], len(self.data["yv"]), {"accuracy": m["accuracy"]}


class HierarchicalFLSimulator:
    def __init__(self, config: Dict[str, Any], recorder: Optional[RunRecorder] = None):
        self.cfg = make_config(**config)
        self.rec = recorder or RunRecorder(self.cfg)
        self._stop = threading.Event()
        torch.manual_seed(self.cfg["seed"])
        self.rng = np.random.RandomState(self.cfg["seed"])
        self._setup()

    def _setup(self):
        c = self.cfg
        Xtr, ytr, Xte, yte = D.load_split(c["dataset"], seed=c["seed"], max_train=c["max_train"])
        self.n_classes = int(max(ytr.max(), yte.max())) + 1
        parts = D.partition(ytr, c["n_clients"], c["partition"], alpha=c["alpha"], quantity_sigma=c["quantity_sigma"],
                            seed=c["seed"], classes_per_client=c["classes_per_client"])
        clients = D.split_clients(Xtr, ytr, parts, val_ratio=0.2, seed=c["seed"])
        norm = Normalizer().fit_stats([Normalizer.sufficient_stats(cl["X"]) for cl in clients])
        for cl in clients:
            cl["X"], cl["Xv"] = norm.transform(cl["X"]), norm.transform(cl["Xv"])
        self.clients = clients
        self.Xte, self.yte = norm.transform(Xte), yte
        self.model = MLP(Xtr.shape[1], self.n_classes, hidden=c["hidden"])
        self.keys = FL.state_keys(self.model)
        self.global_nd = FL.get_ndarrays(self.model, self.keys)
        self.mbytes = FL.nbytes(self.global_nd)
        self.n_param = int(sum(a.size for a in self.global_nd))

        nc, ne = c["n_clients"], c["n_edges"]
        self.n_samples = np.array([len(cl["y"]) for cl in clients])
        self.dist_counts = np.stack([D.class_distribution(cl["y"], self.n_classes) for cl in clients])
        self.pos_c = self.rng.uniform(0, 100, size=(nc, 2))
        ang = np.linspace(0, 2 * np.pi, ne, endpoint=False)
        self.pos_e = np.stack([50 + 30 * np.cos(ang), 50 + 30 * np.sin(ang)], 1) if ne > 1 else np.array([[50.0, 50.0]])
        cap = np.full(ne, int(math.ceil(nc / ne * 1.3)))
        rewards = self.rng.uniform(20, 40, ne) * max(1.0, self.n_samples.mean() / 70.0)
        costs = self.rng.uniform(0.002, 0.01, nc)
        self.matcher = GameMatcher(nc, ne, self.pos_c, self.pos_e, self.n_samples, cap, rewards, costs,
                                   beta=c["ema_beta"])
        self.bandwidth = self.rng.lognormal(np.log(c["bandwidth_mbps"]), 0.5, nc)
        self.device_ms = self.rng.lognormal(np.log(c["device_ms_per_sample"]), 0.4, nc)
        self.edge_latency = 20.0 + self.rng.uniform(0, 15, ne)
        n_delay = int(round(c["delay_frac"] * nc))
        self.delayed = set(self.rng.choice(nc, n_delay, replace=False).tolist()) if n_delay else set()
        self.malicious = set(self.rng.choice(nc, c["n_malicious"], replace=False).tolist()) if c["n_malicious"] else set()
        self.assign = -np.ones(nc, dtype=np.int64)
        self.proj = torch.randn(self.n_param, c["sketch_dim"], generator=torch.Generator().manual_seed(1)) \
            / math.sqrt(c["sketch_dim"])
        self.cum_update = torch.zeros(nc, self.n_param)

        self.proxies = [FL.as_proxy(i, MNISTClient(i, cl, copy.deepcopy(self.model), self.n_classes,
                                                   i in self.malicious, c["attack"], c["seed"]))
                        for i, cl in enumerate(clients)]
        fit_cfg = lambda rnd: {"local_epochs": c["local_epochs"], "lr": c["lr"], "batch_size": c["batch_size"],
                               "seed": c["seed"] * 100000 + rnd}
        self.edge_strategies = [FL.make_strategy(c["algorithm"], self.global_nd, mu=c["mu"], q=c["q"], lr=1.0 / c["qL"],
                                                 on_fit_config=fit_cfg) for _ in range(ne)]
        self.server_strategy = FedAvg(**FL._all(0), initial_parameters=ndarrays_to_parameters(self.global_nd))
        self.edge_proxies = [FL.LocalClientProxy("edge-%d" % j, None) for j in range(ne)]

        totals = self.dist_counts.sum(0)
        ratio = data_ratio(self.n_samples)
        clients_static = [{
            "id": i, "n": int(self.n_samples[i]), "ratio": float(ratio[i]),
            "entropy": entropy(self.dist_counts[i]), "norm_entropy": normalized_entropy(self.dist_counts[i]),
            "class_counts": self.dist_counts[i].astype(int).tolist(), "pos": self.pos_c[i].tolist(),
            "classes": [int(k) for k in np.where(self.dist_counts[i] > 0)[0]],
            "bandwidth_mbps": float(self.bandwidth[i]), "cost": float(costs[i]),
            "malicious": i in self.malicious, "delayed": i in self.delayed} for i in range(nc)]
        topology = {"server": {"id": "server", "pos": [50, 50]},
                    "edges": [{"id": j, "pos": self.pos_e[j].tolist(), "capacity": int(cap[j]),
                               "reward": float(rewards[j]), "latency_ms": float(self.edge_latency[j])}
                              for j in range(ne)],
                    "n_classes": self.n_classes, "class_totals": totals.astype(int).tolist(),
                    "model_bytes": self.mbytes, "n_params": self.n_param, "framework": "flwr"}
        self.rec.set_topology(topology, clients_static)

    def stop(self):
        self._stop.set()

    def _participants(self, t: int) -> np.ndarray:
        c, nc = self.cfg, self.cfg["n_clients"]
        frac = self.rng.uniform(c["participation_min"], 1.0) if c["participation_mode"] == "random" else c["participation"]
        k = max(1, int(round(frac * nc)))
        return np.sort(self.rng.choice(nc, k, replace=False))

    def _link(self, i: int, j: int, t: int, nbytes: int) -> float:
        c = self.cfg
        lat = c["base_latency_ms"] + 0.3 * self.matcher.dist[i, j] + self.rng.exponential(3.0)
        if i in self.delayed:
            lat += c["delay_growth_ms"] * t
        return lat + nbytes * 8 / (self.bandwidth[i] * 1e6) * 1e3

    def run(self) -> RunRecorder:
        c = self.cfg
        sampler = ResourceSampler(0.05).start()
        self.rec.set_status("running")
        self.rec.log(0, "info", "start", algorithm=c["algorithm"], n_clients=c["n_clients"], n_edges=c["n_edges"],
                     framework="flwr")
        try:
            for t in range(c["rounds"]):
                if self._stop.is_set():
                    self.rec.log(t, "warn", "stopped")
                    break
                self._round(t, sampler)
                if c["sleep"] > 0:
                    time.sleep(c["sleep"])
            self.rec.set_status("finished", summary=self.summary())
        except Exception as e:  # pragma: no cover
            import traceback
            self.rec.set_status("error", error=traceback.format_exc())
            raise e
        finally:
            sampler.stop()
        return self.rec

    def _round(self, t: int, sampler: ResourceSampler):
        c, nc, ne = self.cfg, self.cfg["n_clients"], self.cfg["n_edges"]
        rnd = t + 1
        t0 = time.time()
        rematched = t % c["rematch_every"] == 0
        if rematched:
            prev = self.assign.copy()
            match_info = self.matcher.match()
            self.assign = match_info["assign"]
            moved = int((prev >= 0).sum() and (prev != self.assign).sum())
            self.rec.log(t, "info", "rematch", moved=moved, blocking=int(match_info["blocking"]))

        part = set(self._participants(t).tolist())
        gparams = ndarrays_to_parameters(self.global_nd)
        gflat = FL.flatten(self.global_nd)
        crec: List[Dict[str, Any]] = [{"id": i, "edge": int(self.assign[i]), "participated": i in part,
                                       "status": "idle", "n": int(self.n_samples[i])} for i in range(nc)]

        edge_results: Dict[int, list] = {}
        edge_failures: Dict[int, list] = {}
        updates, owners = [], []
        for j in range(ne):
            proxies = [self.proxies[i] for i in range(nc) if self.assign[i] == j and i in part]
            results, failures = FL.run_fit(self.edge_strategies[j], rnd, gparams, proxies)
            ok = []
            for proxy, res in results:
                i = int(proxy.cid)
                m = res.metrics
                up_ms, down_ms = self._link(i, j, t, self.mbytes), 0.5 * self._link(i, j, t, self.mbytes)
                sim_compute = len(self.clients[i]["y"]) * c["local_epochs"] * self.device_ms[i]
                lat = up_ms + down_ms + sim_compute
                timeout = c["timeout_ms"] is not None and lat > c["timeout_ms"]
                crec[i].update({"status": "timeout" if timeout else "ok", "loss_before": m["loss_before"],
                                "train_loss": m["train_loss"], "val_acc": m["val_acc"], "val_loss": m["val_loss"],
                                "latency_ms": lat, "sim_compute_ms": sim_compute, "compute_ms": m["compute_ms"], "bytes_down": self.mbytes,
                                "bytes_up": 0 if timeout else self.mbytes})
                if i in self.malicious:
                    self.rec.log(t, "warn", "malicious_update", client=i, attack=c["attack"])
                if timeout:
                    failures.append((proxy, res))
                    self.rec.log(t, "warn", "timeout", client=i, latency_ms=round(lat, 1))
                    continue
                ok.append((proxy, res))
                upd = FL.flatten(parameters_to_ndarrays(res.parameters)) - gflat
                updates.append(torch.as_tensor(upd, dtype=torch.float32))
                owners.append(i)
            edge_results[j], edge_failures[j] = ok, failures

        od = detect_update_outliers(updates, c["outlier_z"])
        flagged = set()
        for k, i in enumerate(owners):
            crec[i].update({"update_norm": float(od["norm"][k]), "cos": float(od["cos"][k]),
                            "z": float(od["z"][k]), "outlier": bool(od["outlier"][k])})
            if od["outlier"][k]:
                flagged.add(i)
                self.rec.log(t, "alert", "outlier", client=i, z=round(float(od["z"][k]), 2), excluded=bool(c["defense"]))
            self.cum_update[i] += updates[k]

        for i in owners:
            crec[i]["quality"] = self.matcher.quality.update(i, 0.0 if i in flagged else crec[i]["val_acc"])
        for r in crec:
            r.setdefault("quality", float(self.matcher.quality.Q[r["id"]]))

        server_results, edges_rec = [], []
        for j in range(ne):
            res_j = [(p, r) for p, r in edge_results[j] if not (c["defense"] and int(p.cid) in flagged)]
            assigned = [i for i in range(nc) if self.assign[i] == j]
            erec = {"id": j, "assigned": len(assigned), "accepted": len(res_j),
                    "timeouts": sum(1 for i in assigned if crec[i]["status"] == "timeout"),
                    "outliers": sum(1 for i in assigned if i in flagged),
                    "latency_ms": float(self.edge_latency[j] + self.rng.exponential(2.0)),
                    "bytes_up": 0, "bytes_in": sum(crec[i].get("bytes_up", 0) for i in assigned)}
            if res_j:
                agg, _ = self.edge_strategies[j].aggregate_fit(rnd, res_j, edge_failures[j])
                n_edge = int(sum(r.num_examples for _, r in res_j))
                server_results.append((self.edge_proxies[j], FL.fit_res(parameters_to_ndarrays(agg), n_edge)))
                erec.update({"bytes_up": self.mbytes, "n_samples": n_edge})
                w = self._edge_weights(j, res_j)
                for (p, _), wi in zip(res_j, w):
                    crec[int(p.cid)]["edge_weight"] = float(wi)
            busy = erec["timeouts"] + erec["outliers"]
            erec["status"] = "idle" if not assigned else ("down" if not res_j else ("degraded" if busy > 0 else "ok"))
            edges_rec.append(erec)
        if server_results:
            agg, _ = self.server_strategy.aggregate_fit(rnd, server_results, [])
            self.global_nd = parameters_to_ndarrays(agg)
            tot = sum(r.num_examples for _, r in server_results)
            sw = {p.cid: r.num_examples / tot for p, r in server_results}
            for erec in edges_rec:
                if erec["accepted"]:
                    erec["server_weight"] = float(sw["edge-%d" % erec["id"]])
            for r in crec:
                if "edge_weight" in r:
                    r["weight"] = r["edge_weight"] * edges_rec[r["edge"]]["server_weight"]
        for r in crec:
            r.setdefault("weight", 0.0)

        FL.set_ndarrays(self.model, self.keys, self.global_nd)
        g = evaluate(self.model, self.Xte, self.yte, self.n_classes)
        per_client = []
        for i, cl in enumerate(self.clients):
            a = evaluate(self.model, cl["Xv"], cl["yv"], self.n_classes)["accuracy"]
            crec[i]["global_acc"] = a
            per_client.append(a)
        _, pred = evaluate(self.model, self.Xte, self.yte, self.n_classes, return_pred=True)
        class_acc = [float((pred[self.yte == k] == k).mean()) for k in range(self.n_classes)]

        lat = [r["latency_ms"] for r in crec if r["status"] == "ok"]
        rnd_rec = {
            "round": rnd, "wall_s": time.time() - t0,
            "sim_time_ms": (max(lat) if lat else 0.0) + max([e["latency_ms"] for e in edges_rec] or [0.0]),
            "global": g, "fairness": fairness(per_client), "class_acc": class_acc,
            "participants": len(part), "accepted": sum(e["accepted"] for e in edges_rec),
            "bytes_total": int(sum(r.get("bytes_up", 0) + r.get("bytes_down", 0) for r in crec)
                               + sum(e["bytes_up"] for e in edges_rec) + self.mbytes * ne),
            "clients": crec, "edges": edges_rec, "resource": sampler.drain(), "rematched": rematched,
        }
        if rematched:
            rnd_rec["matching"] = {k: match_info[k] for k in ("assign", "rank", "edge_rank", "Q", "profit", "score", "blocking")}
        rnd_rec["rank"] = (np.argsort(np.argsort(-self.matcher.quality.Q)) + 1).tolist()
        self.rec.add_round(rnd_rec)
        sk = {}
        for i in range(nc):
            v = self.cum_update[i]
            if float(v.norm()) > 0:
                s = v @ self.proj
                sk[i] = (s / (s.norm() + 1e-12)).numpy()
        self.rec.set_sketch(sk)

    def _edge_weights(self, j: int, results) -> np.ndarray:
        if self.cfg["algorithm"] == "qfedavg":
            w = np.array([(r.metrics["loss_before"] + 1e-10) ** self.cfg["q"] for _, r in results])
        else:
            w = np.array([r.num_examples for _, r in results], dtype=np.float64)
        return w / w.sum()

    def summary(self) -> Dict[str, Any]:
        rs = self.rec.rounds
        if not rs:
            return {}
        from ..common.metrics import convergence_round
        accs = [r["global"]["accuracy"] for r in rs]
        last = rs[-1]
        return {"final_accuracy": accs[-1], "best_accuracy": max(accs),
                "final_f1": last["global"]["f1"], "final_precision": last["global"]["precision"],
                "final_recall": last["global"]["recall"], "convergence_round": convergence_round(accs),
                "fairness": last["fairness"], "class_acc": last.get("class_acc"),
                "total_bytes": int(sum(r["bytes_total"] for r in rs)),
                "sim_time_s": sum(r["sim_time_ms"] for r in rs) / 1e3, "framework": "flwr"}


def run_simulation(config: Dict[str, Any], recorder: Optional[RunRecorder] = None) -> RunRecorder:
    return HierarchicalFLSimulator(config, recorder).run()
