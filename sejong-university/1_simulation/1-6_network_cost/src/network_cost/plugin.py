import hashlib
import json
import threading

import numpy as np
from flwr.app import ConfigRecord, MessageType, MetricRecord

from fl_core.plugins import Middleware, Plugin
from fl_core.records import to_arrays, to_record
from fl_core.strategies import StrategyWrapper
from network_cost.compress import compress_arrays
from network_cost.topology import LRUCache, Topology

OPTIMIZATIONS = ("edge_compute", "cache", "dag")
DEFAULTS = {"optimizations": list(OPTIMIZATIONS), "ratio": 0.25, "bits": 8, "error-feedback": True, "cache-mb": 64,
            "chunk-kb": 64, "aps": 4, "routers": 3, "weights": {"price": 1.0, "time": 1.0, "congestion": 0.5},
            "seed": 0}
FIELDS = ("down_bytes_base", "down_bytes_opt", "up_bytes_base", "up_bytes_opt", "cost_base", "cost_opt",
          "time_ms_base", "time_ms_opt", "cache_hits", "cache_misses", "messages")


def metric_key(field):
    return "net-" + field.replace("_", "-")


def chunks(arrays, chunk_bytes):
    out = []
    for k, v in arrays.items():
        raw = np.ascontiguousarray(v).tobytes()
        for s in range(0, max(len(raw), 1), chunk_bytes):
            piece = raw[s:s + chunk_bytes]
            out.append((hashlib.sha1(k.encode() + piece).hexdigest(), len(piece)))
    return out


class Accountant:
    def __init__(self):
        self.rows = {}
        self.lock = threading.Lock()

    def add(self, rnd, **values):
        with self.lock:
            row = self.rows.setdefault(rnd, {f: 0.0 for f in FIELDS})
            for k, v in values.items():
                if k.startswith("time_ms"):
                    row[k] = max(row[k], v)
                else:
                    row[k] += v

    def get(self, rnd):
        return dict(self.rows.get(rnd, {f: 0.0 for f in FIELDS}))


class CostCollector(StrategyWrapper):
    def __init__(self, inner, accountant):
        super().__init__(inner)
        self.acc = accountant

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        for r in replies:
            if r.has_error():
                continue
            m = r.content["metrics"]
            values = {f: float(m[metric_key(f)]) for f in FIELDS if metric_key(f) in m}
            if values:
                self.acc.add(server_round, **values)
        return self.inner.aggregate_train(server_round, replies)


class CostMiddleware(Middleware):
    def __init__(self, topology, dest, options):
        self.topo, self.dest, self.o = topology, dest, options
        self.pending = {}
        self.opts = set(options["optimizations"])
        self.caches = {}
        self.residuals = {}
        self.round = None
        self.loads = {"base": {}, "opt": {}}
        self.lock = threading.Lock()

    def _cache(self, node):
        if node not in self.caches:
            self.caches[node] = LRUCache(self.o["cache-mb"] * 2 ** 20)
        return self.caches[node]

    def _route(self, client, nbytes):
        static = self.topo.static_path(client, self.dest)
        best = (self.topo.best_path(client, self.dest, nbytes, self.o["weights"], self.loads["opt"])
                if "dag" in self.opts else static)
        return static, best

    def _charge(self, ledger, links, nbytes_per_hop):
        loads = self.loads[ledger]
        cost, ms = 0.0, 0.0
        for link, nb in zip(links, nbytes_per_hop):
            cost += link.cost(nb, self.o["weights"], loads.get(link.key, 0.0))
            ms += link.latency_ms + nb * 8 / (link.bw_mbps * 1e6) * 1e3
            loads[link.key] = loads.get(link.key, 0.0) + nb
        return cost, ms

    def _tick(self, rnd):
        with self.lock:
            if rnd != self.round:
                self.round = rnd
                self.loads = {"base": {}, "opt": {}}

    def before(self, msg, node):
        if msg.metadata.message_type != MessageType.TRAIN or not msg.has_content() or "arrays" not in msg.content:
            return None
        rnd = int(msg.content["config"].get("server-round", 0)) if "config" in msg.content else 0
        self._tick(rnd)
        client = f"c{int(node.context.node_config['partition-id'])}"
        arrays = to_arrays(msg.content["arrays"])
        parts = chunks(arrays, self.o["chunk-kb"] * 1024)
        total = sum(b for _, b in parts)
        with self.lock:
            static, best = self._route(client, total)
            down_static = list(reversed(static))
            down_best = list(reversed(best))
            base_links = [self.topo.links[(b, a)] for a, b in zip(down_static, down_static[1:])]
            base_cost, base_ms = self._charge("base", base_links, [total] * len(base_links))
            per_hop = []
            hits = misses = 0
            for a, b in zip(down_best, down_best[1:]):
                sent = 0
                cache = self._cache(b) if "cache" in self.opts else None
                for h, nb in parts:
                    if cache is not None and cache.get(h):
                        hits += 1
                        continue
                    misses += 1
                    sent += nb
                    if cache is not None:
                        cache.put(h, nb)
                per_hop.append(sent)
            opt_links = [self.topo.links[(b, a)] for a, b in zip(down_best, down_best[1:])]
            opt_cost, opt_ms = self._charge("opt", opt_links, per_hop)
        self.pending[(client, rnd)] = dict(down_bytes_base=total * (len(static) - 1), down_bytes_opt=sum(per_hop),
                                           cost_base=base_cost, cost_opt=opt_cost, time_ms_base=base_ms,
                                           time_ms_opt=opt_ms, cache_hits=hits, cache_misses=misses, messages=1)
        return None

    def after(self, msg, reply, node):
        if reply.has_error() or msg.metadata.message_type != MessageType.TRAIN or "arrays" not in reply.content:
            return reply
        rnd = int(msg.content["config"].get("server-round", 0))
        pid = int(node.context.node_config["partition-id"])
        client = f"c{pid}"
        local = to_arrays(reply.content["arrays"])
        full = int(sum(v.nbytes for v in local.values()))
        sent = full
        if "edge_compute" in self.opts:
            reference = to_arrays(msg.content["arrays"])
            residual = self.residuals.get(pid) if self.o["error-feedback"] else None
            decoded, sent, new_residual = compress_arrays(local, reference, self.o["ratio"], self.o["bits"], residual)
            if self.o["error-feedback"]:
                self.residuals[pid] = new_residual
            reply.content["arrays"] = to_record(decoded)
        with self.lock:
            static, best = self._route(client, sent)
            base_cost, base_ms = self._charge("base", self.topo.hops(static), [full] * (len(static) - 1))
            opt_cost, opt_ms = self._charge("opt", self.topo.hops(best), [sent] * (len(best) - 1))
        stats = self.pending.pop((client, rnd), {f: 0.0 for f in FIELDS})
        stats.update(up_bytes_base=full * (len(static) - 1), up_bytes_opt=sent * (len(best) - 1),
                     cost_base=stats.get("cost_base", 0.0) + base_cost, cost_opt=stats.get("cost_opt", 0.0) + opt_cost,
                     time_ms_base=stats.get("time_ms_base", 0.0) + base_ms,
                     time_ms_opt=stats.get("time_ms_opt", 0.0) + opt_ms)
        metrics = dict(reply.content["metrics"])
        metrics.update({metric_key(k): float(v) for k, v in stats.items()})
        reply.content["metrics"] = MetricRecord(metrics)
        return reply


class NetworkCostPlugin(Plugin):
    name = "network_cost"

    def __init__(self, config=None):
        super().__init__({**DEFAULTS, **(config or {})})
        if isinstance(self.config["optimizations"], str):
            self.config["optimizations"] = [o for o in self.config["optimizations"].split(",") if o]
        bad = set(self.config["optimizations"]) - set(OPTIMIZATIONS)
        if bad:
            raise ValueError(f"unknown optimizations {sorted(bad)}; choose from {', '.join(OPTIMIZATIONS)}")
        self.accountant = Accountant()
        self.rows = []

    def setup(self, owner):
        n = owner.num_clients if hasattr(owner, "num_clients") else owner.n
        edges = max(1, getattr(owner, "m", 1) or 1)
        if hasattr(owner, "payload"):
            edges = max(1, int(owner.payload.get("num-edges", 1)))
        self.topology = Topology(n, edges, self.config["aps"], self.config["routers"], self.config["seed"])
        self.dest = f"e{owner.edge_id}" if getattr(owner, "edge_id", None) is not None else "cloud"

    def client_middleware(self, owner):
        return [CostMiddleware(self.topology, self.dest, self.config)]

    def wrap_strategy(self, strategy, tier, owner):
        if tier in ("flat", "edge"):
            return CostCollector(strategy, self.accountant)
        return strategy

    def edge_reply(self, session, rnd, content):
        report = json.loads(str(content["report"]["json"]))
        rows = [self.accountant.get(r) for r in range((rnd - 1) * session.edge_rounds + 1,
                                                     rnd * session.edge_rounds + 1)]
        merged = {f: (max(x[f] for x in rows) if f.startswith("time_ms") else sum(x[f] for x in rows)) for f in FIELDS}
        report["network_cost"] = merged
        content["report"] = ConfigRecord({"json": json.dumps(report)})
        return content

    def on_round_end(self, owner, rnd, row):
        if rnd == 0:
            return
        if hasattr(owner, "last_reports"):
            parts = [rep.get("network_cost", {}) for rep in owner.last_reports.values()]
            stats = {f: (max([p.get(f, 0.0) for p in parts], default=0.0) if f.startswith("time_ms")
                         else sum(p.get(f, 0.0) for p in parts)) for f in FIELDS}
        else:
            stats = self.accountant.get(rnd)
        stats["round"] = rnd
        stats["bytes_base"] = stats["down_bytes_base"] + stats["up_bytes_base"]
        stats["bytes_opt"] = stats["down_bytes_opt"] + stats["up_bytes_opt"]
        self.rows.append(stats)
        row["net_mb_base"] = stats["bytes_base"] / 2 ** 20
        row["net_mb_opt"] = stats["bytes_opt"] / 2 ** 20
        row["net_cost_base"] = stats["cost_base"]
        row["net_cost_opt"] = stats["cost_opt"]

    def summary(self, owner):
        tb = sum(r["bytes_base"] for r in self.rows)
        to = sum(r["bytes_opt"] for r in self.rows)
        cb = sum(r["cost_base"] for r in self.rows)
        co = sum(r["cost_opt"] for r in self.rows)
        hits = sum(r["cache_hits"] for r in self.rows)
        miss = sum(r["cache_misses"] for r in self.rows)
        return {"optimizations": self.config["optimizations"], "rounds": self.rows,
                "bytes_base_mb": tb / 2 ** 20, "bytes_opt_mb": to / 2 ** 20,
                "bytes_saving": 1 - to / tb if tb else 0.0, "cost_base": cb, "cost_opt": co,
                "cost_saving": 1 - co / cb if cb else 0.0,
                "cache_hit_rate": hits / (hits + miss) if hits + miss else 0.0}
