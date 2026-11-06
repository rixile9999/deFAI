import json
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord

from fl_core.backends import ensure_identity, get_backend
from fl_core.records import record_bytes, to_arrays
from fl_core.runner import centralized_evaluator, history_from_result, quiet, write_json
from fl_core.strategies import StrategyWrapper
from hierarchical_fl.config import make_config
from hierarchical_fl.edge_grid import EdgeGrid, EdgeLink, HttpTransport, LocalTransport
from hierarchical_fl.matching import Matcher, QualityEMA, priority_rank
from hierarchical_fl.network import NetworkModel
from fl_core.plugins import load_plugins, spec_of
from hierarchical_fl.registry import Registry


class RoundHook(StrategyWrapper):
    def __init__(self, inner, server):
        super().__init__(inner)
        self.server = server

    def configure_train(self, server_round, arrays, config, grid):
        self.server.prepare_round(server_round, arrays)
        return self.inner.configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        arrays, metrics = self.inner.aggregate_train(server_round, replies)
        self.server.train_metrics = dict(metrics or {})
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        metrics = self.inner.aggregate_evaluate(server_round, replies)
        self.server.train_metrics.update({k: v for k, v in dict(metrics or {}).items() if k != "num-examples"})
        return metrics


class HierarchicalServer:
    def __init__(self, overrides=None, plugins=None):
        quiet()
        ensure_identity()
        cfg = make_config(overrides)
        self.plugins = load_plugins(cfg["plugins"]) + list(plugins or [])
        known = len(cfg["plugins"])
        for p in list(self.plugins):
            cfg = p.configure(cfg)
        self.plugins += load_plugins(cfg["plugins"][known:])
        cfg["plugins"] = [spec_of(p) for p in self.plugins]
        self.cfg = cfg
        self.run_id = cfg["run-id"]
        self.run_dir = Path(cfg["output-dir"]) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.registry = Registry(self.run_dir / "hfl.sqlite")
        self.backend = get_backend(cfg["backend"], cfg["backend-config"])
        cfg["dataset"] = self.backend.cfg["dataset"]
        self.n, self.m = cfg["num-clients"], cfg["num-edges"]
        self.capacity = [cfg["edge-capacity"]] * self.m
        self.net = NetworkModel(self.n, self.m, self.backend.cfg["seed"])
        self.rng = np.random.default_rng(self.backend.cfg["seed"] + 101)
        self.ema = QualityEMA(self.n, cfg["matching"]["ema-beta"])
        self.cluster = None
        self.links = []
        self.plans = {}
        self.round_info = {}
        self.rows = []
        self.train_metrics = {}
        self.client_stats = []
        self.model_bytes = 0
        self.roles = {}
        self.alerts = []
        self.last_reports = {}
        self.client_history = []

    def event(self, rnd, level, source, message):
        self.registry.event(self.run_id, rnd, level, source, message)
        self.alerts.append({"round": rnd, "level": level, "source": source, "message": message})

    def alert(self, level, message, source="global-defense"):
        self.event(self.round_info.get("round", 0), level, source, message)

    def _transports(self):
        cfg = self.cfg
        if cfg["transport"] == "local":
            return [LocalTransport(i) for i in range(self.m)]
        urls = list(cfg["edge-urls"])
        if not urls and cfg["spawn-edges"]:
            from hierarchical_fl.cluster import EdgeCluster

            self.cluster = EdgeCluster(self.m, base_port=cfg["edge-port"], log_dir=self.run_dir / "edge_logs")
            urls = self.cluster.start()
        if len(urls) < self.m:
            raise ValueError(f"transport=http needs {self.m} edge URLs (or spawn-edges=true)")
        return [HttpTransport(u) for u in urls[:self.m]]

    def setup(self):
        cfg = self.cfg
        self.registry.start_run(self.run_id, cfg)
        payload = {"backend": cfg["backend"], "backend-config": self.backend.cfg, "num-clients": self.n,
                   "num-edges": self.m,
                   "edge-rounds": cfg["edge-rounds"], "edge-strategy": cfg["edge-strategy"],
                   "client-workers": cfg["client-workers"], "plugins": cfg["plugins"]}
        for eid, t in enumerate(self._transports()):
            status, body = t.post_json("/api/v1/sessions", dict(payload, sid=self.run_id))
            if status != 200:
                raise RuntimeError(f"edge{eid} refused the session: {status} {body}")
            self.links.append(EdgeLink(eid, t, self.run_id))
        self.registry.add_edges(self.run_id, [{"eid": l.edge_id, "url": l.transport.url, "capacity": c}
                                              for l, c in zip(self.links, self.capacity)])
        _, stats = self.links[0].transport.get(self.links[0].base + "/clients/stats")
        self.client_stats = stats
        for p in self.plugins:
            p.setup(self)
        for row in stats:
            row["role"] = self.roles.get(row["cid"], "benign")
        self.registry.add_clients(self.run_id, stats)
        self.n_samples = np.asarray([r["n_samples"] for r in stats], dtype=float)
        self.matcher = Matcher(cfg["matching"]["mode"], self.net, self.n_samples, self.capacity, cfg["matching"],
                               self.backend.cfg["seed"])
        self.grid = EdgeGrid(self.links, self, cfg["http-timeout"])
        write_json(self.run_dir / "network.json", self.net.describe())

    def _faults(self, rnd, state):
        for link in self.links:
            down = [rnd] if link.edge_id in state["down_edges"] else []
            hang = [rnd] if link.edge_id in state["hang_edges"] else []
            link.transport.post_json(link.base + "/fault", {"down": down, "hang": hang,
                                                            "hang-sec": state.get("hang_sec", 5.0)})
        available = []
        for link in self.links:
            status, _ = link.transport.get(link.base + "/health", {"round": rnd})
            if status == 200:
                available.append(link.edge_id)
            else:
                self.event(rnd, "warning", f"edge{link.edge_id}", f"health check failed (HTTP {status})")
        return available

    def _deadline(self, expected):
        d = self.cfg["deadline"]
        if d["policy"] == "fixed":
            return float(d["ms"])
        if d["policy"] == "adaptive":
            return float(np.quantile(expected.min(axis=1), d["quantile"]) * d["margin"])
        return float("inf")

    def prepare_round(self, rnd, arrays):
        self.model_bytes = record_bytes(arrays)
        self.current_global = to_arrays(arrays)
        state = {"participating": set(range(self.n)), "delay_factor": 1.0, "spikes": None, "down_edges": set(),
                 "hang_edges": set(), "drop": set()}
        for p in self.plugins:
            state = p.round_state(self, rnd, state)
        available = self._faults(rnd, state)
        participating = sorted(state["participating"])
        e_steps = np.maximum(1, np.round(float(self.backend.cfg["local-epochs"]) * self.n_samples
                                         / float(self.backend.cfg["batch-size"])))
        comm_exp = self.net.comm_ms(self.model_bytes, None, state["delay_factor"])
        expected = (comm_exp + self.net.compute_ms(e_steps)[:, None]) * self.cfg["edge-rounds"]
        deadline = self._deadline(expected[participating] if participating else expected)
        q_hat = self.ema.q.copy()
        assign, info = self.matcher.assign(q_hat, expected, comm_exp, deadline, available, participating)
        comm_real = self.net.comm_ms(self.model_bytes, self.rng, state["delay_factor"], state["spikes"])
        realized = (comm_real + self.net.compute_ms(e_steps)[:, None]) * self.cfg["edge-rounds"]
        plans = {i: {"clients": [], "late": [], "drop": []} for i in range(self.m)}
        rows, client_ms = [], np.zeros(self.n)
        for j in range(self.n):
            i = int(assign[j])
            status = "idle" if j not in state["participating"] else "unmatched"
            if i >= 0:
                client_ms[j] = realized[j, i]
                if realized[j, i] > deadline:
                    status = "late"
                    plans[i]["late"].append(j)
                else:
                    status = "dropped" if j in state["drop"] else "matched"
                    plans[i]["clients"].append(j)
                    if j in state["drop"]:
                        plans[i]["drop"].append(j)
            rows.append({"cid": j, "eid": i, "utility": float(info["utility"][j, i]) if i >= 0 else None,
                         "priority": float(info["priority"][i, j]) if i >= 0 else None, "q_hat": float(q_hat[j]),
                         "status": status})
        self.registry.add_assignments(self.run_id, rnd, rows)
        self.plans = plans
        self.grid.available = {i + 1 for i in available if plans[i]["clients"]}
        edge_ms = {i: float(max([client_ms[j] for j in plans[i]["clients"]], default=0.0)
                            + self.net.edge_global_ms[i]) for i in range(self.m)}
        self.round_info = {"round": rnd, "deadline_ms": deadline if np.isfinite(deadline) else None,
                           "participants": len(participating), "edges_up": len(available),
                           "matched": int(sum(len(p["clients"]) for p in plans.values())),
                           "late": int(sum(len(p["late"]) for p in plans.values())),
                           "dropped": int(len(state["drop"] & set(j for p in plans.values() for j in p["clients"]))),
                           "unmatched": int(sum(1 for r in rows if r["status"] == "unmatched")),
                           "blocking_pairs": info["blocking_pairs"], "delay_factor": state["delay_factor"],
                           "edge_ms": edge_ms,
                           "matching": {"utility": np.round(info["utility"], 4).tolist(),
                                        "priority_rank": priority_rank(info["priority"]),
                                        "assign": assign.tolist()}}

    def plan_for(self, edge_id):
        return self.plans.get(edge_id, {"clients": []})

    def transport_hook(self, rnd, edge_id, reply):
        for p in self.plugins:
            reply = p.transport(self, rnd, edge_id, reply)
        return reply

    def after_exchange(self, rnd, replies, infos):
        rows = []
        self.last_reports = {}
        virtual = 0.0
        for reply, info in zip(replies, infos):
            eid = info["eid"]
            report = {}
            if not reply.has_error() and "report" in reply.content:
                report = json.loads(str(reply.content["report"]["json"]))
                self.last_reports[eid] = report
                for c in report.get("clients", []):
                    if "q" in c:
                        self.ema.update(c["cid"], c["q"])
                        self.client_history.append(dict(c, round=rnd, eid=eid))
                for a in report.get("alerts", []):
                    self.event(rnd, a["level"], a["source"], a["message"])
            else:
                self.event(rnd, "warning", f"edge{eid}", reply.error.reason if reply.has_error() else "no report")
            ms = self.round_info["edge_ms"].get(eid, 0.0) if info["status"] == "ok" else 0.0
            virtual = max(virtual, ms)
            rows.append({"eid": eid, "status": info["status"], "bytes_up": info["bytes_up"],
                         "bytes_down": info["bytes_down"], "seconds": info["seconds"], "virtual_ms": ms,
                         "clients": len(report.get("clients", [])), "data": {"clients": report.get("clients", [])}})
        self.registry.add_edge_rounds(self.run_id, rnd, rows)
        self.round_info.update(virtual_ms=virtual, bytes_up=int(sum(r["bytes_up"] for r in rows)),
                               bytes_down=int(sum(r["bytes_down"] for r in rows)),
                               edges_ok=int(sum(r["status"] == "ok" for r in rows)))

    def _on_evaluate(self, rnd, arrays, record):
        if rnd == 0:
            row = {"round": 0}
        else:
            row = {k: v for k, v in self.round_info.items() if k not in ("edge_ms", "matching")}
            row.update({k: v for k, v in self.train_metrics.items() if isinstance(v, (int, float))})
        row.update(dict(record))
        row["wall_seconds"] = time.perf_counter() - self.t0
        for p in self.plugins:
            p.on_round_end(self, rnd, row)
        self.rows.append(row)
        stored = dict(row)
        if rnd > 0:
            stored["matching"] = self.round_info.get("matching")
        self.registry.add_round(self.run_id, rnd, stored)

    def run(self):
        status = "failed"
        try:
            self.setup()
            torch.manual_seed(int(self.backend.cfg["seed"]))
            model = self.backend.build_model()
            strategy = self.backend.build_strategy(self.cfg["global-strategy"], 1.0,
                                                   1.0 if self.cfg["global-evaluate"] else 0.0)
            for p in self.plugins:
                strategy = p.wrap_strategy(strategy, "global", self)
            strategy = RoundHook(strategy, self)
            self.t0 = time.perf_counter()
            result = strategy.start(grid=self.grid, initial_arrays=ArrayRecord(model.state_dict()),
                                    num_rounds=self.cfg["global-rounds"], timeout=self.cfg["http-timeout"],
                                    train_config=self.backend.train_config(),
                                    evaluate_fn=centralized_evaluator(self.backend, model, self._on_evaluate))
            model.load_state_dict(result.arrays.to_torch_state_dict())
            self.model = model
            self.result = result
            status = "finished"
            return self.summary(history_from_result(result))
        finally:
            summary = getattr(self, "_summary", {})
            self.registry.finish_run(self.run_id, status, summary)
            self.close()

    def summary(self, flower_history):
        accs = [r["test_acc"] for r in self.rows if r.get("test_acc") is not None]
        out = {"run_id": self.run_id, "config": self.cfg, "rounds": self.rows,
               "final_acc": accs[-1] if accs else None, "best_acc": max(accs) if accs else None,
               "P_last3": float(np.mean(accs[-3:])) if accs else None,
               "virtual_ms_total": float(sum(r.get("virtual_ms", 0.0) for r in self.rows)),
               "bytes_total": int(sum(r.get("bytes_up", 0) + r.get("bytes_down", 0) for r in self.rows)),
               "alerts": self.alerts, "flower_history": flower_history,
               "plugins": {type(p).__name__: p.summary(self) for p in self.plugins}}
        self._summary = {k: out[k] for k in ("final_acc", "best_acc", "P_last3", "virtual_ms_total", "bytes_total")}
        write_json(self.run_dir / "summary.json", out)
        return out

    def close(self):
        for link in self.links:
            link.transport.delete(link.base)
        if self.cluster is not None:
            self.cluster.stop()
        self.registry.close()


def run_hierarchical(overrides=None, plugins=None):
    return HierarchicalServer(overrides, plugins).run()
