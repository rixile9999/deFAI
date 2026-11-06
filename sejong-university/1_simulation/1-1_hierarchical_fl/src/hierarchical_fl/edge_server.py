import argparse
import json
import logging
import os
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, request
from flwr.app import ConfigRecord, Message, MessageType, MetricRecord, RecordDict

from fl_core import wire
from fl_core.backends import ensure_identity, get_backend
from fl_core.grid import error_reply
from fl_core.records import delta, flatten, float_keys, to_arrays
from fl_core.runner import quiet
from fl_core.plugins import Middleware, load_plugins, merge_hooks


def entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / max(p.sum(), 1)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


class PlanMiddleware(Middleware):
    def __init__(self, session):
        self.session = session

    def before(self, msg, node):
        pid = int(node.context.node_config["partition-id"])
        if msg.metadata.message_type == MessageType.TRAIN and pid in self.session.plan.get("drop", []):
            return error_reply(msg, "client dropped out during local training")
        return None


def client_quality(global_arrays, replies):
    keys = float_keys(global_arrays)
    rows, vecs = [], []
    for r in replies:
        if r.has_error() or "arrays" not in r.content:
            continue
        local = to_arrays(r.content["arrays"])
        vecs.append(flatten(delta(local, global_arrays), keys))
        m = r.content["metrics"]
        rows.append({"cid": int(r.metadata.src_node_id) - 1, "loss": float(m.get("train_loss", 0.0)),
                     "n": int(m.get("num-examples", 0)), "steps": int(m.get("local-steps", 0))})
    if not vecs:
        return rows
    u = np.stack(vecs)
    ref = np.median(u, axis=0)
    norms = np.linalg.norm(u, axis=1)
    cos = u @ ref / (norms * np.linalg.norm(ref) + 1e-12)
    for row, c, nrm in zip(rows, cos, norms):
        row["q"] = float((1.0 + c) / 2.0)
        row["norm"] = float(nrm)
    return rows


class EdgeSession:
    def __init__(self, sid, edge_id, payload):
        ensure_identity()
        self.sid, self.edge_id, self.payload = sid, edge_id, payload
        self.backend = get_backend(payload["backend"], payload.get("backend-config"))
        self.num_clients = int(payload["num-clients"])
        self.edge_rounds = int(payload.get("edge-rounds", 1))
        self.plan = {"clients": [], "drop": []}
        self.round = 0
        self.lock = threading.Lock()
        self.fault = {"down": set(), "hang": set(), "hang-sec": 5.0}
        self.reports = []
        self.alerts = []
        self.plugins = load_plugins(payload.get("plugins"))
        for p in self.plugins:
            p.setup(self)
        hooks = merge_hooks(self.plugins, self)
        train = self.backend.custom_train_handler(**hooks) if hooks else self.backend.train_handler
        middleware = [PlanMiddleware(self)] + [m for p in self.plugins for m in p.client_middleware(self)]
        self.grid = self.backend.grid(self.num_clients, workers=int(payload.get("client-workers", 1)),
                                      middleware=middleware, train=train)
        strategy = self.backend.build_strategy(payload.get("edge-strategy", "fedavg"), 1.0, 0.0)
        for p in self.plugins:
            strategy = p.wrap_strategy(strategy, "edge", self)
        self.strategy = strategy

    def alert(self, level, message):
        self.alerts.append({"round": self.round, "level": level, "source": f"edge{self.edge_id}",
                            "message": message})

    def status(self, rnd):
        if rnd in self.fault["down"]:
            return "down"
        if rnd in self.fault["hang"]:
            return "hang"
        return "up"

    def handle(self, msg):
        if msg.metadata.message_type == MessageType.TRAIN:
            return self.train(msg)
        return self.evaluate(msg)

    def train(self, msg):
        config = msg.content["config"]
        rnd = int(config["server-round"])
        plan = json.loads(str(config.get("plan", "{}")))
        self.round, self.plan = rnd, plan
        active = [c for c in plan.get("clients", [])]
        if not active:
            return error_reply(msg, f"edge{self.edge_id}: no clients matched")
        self.grid.active = {c + 1 for c in active}
        arrays = msg.content["arrays"]
        self.global_in = to_arrays(arrays)
        extra = {k: v for k, v in msg.content.items() if k not in ("arrays", "config")}
        train_cfg = ConfigRecord({k: v for k, v in config.items() if k not in ("plan", "server-round")})
        start = time.perf_counter()
        clients, losses, examples, metrics = {}, [], 0, {}
        for e in range(self.edge_rounds):
            edge_round = (rnd - 1) * self.edge_rounds + e + 1
            messages = list(self.strategy.configure_train(edge_round, arrays, train_cfg, self.grid))
            for m in messages:
                for k, v in extra.items():
                    m.content[k] = v
            replies = list(self.grid.send_and_receive(messages))
            stats = client_quality(to_arrays(arrays), replies)
            for row in stats:
                prev = clients.get(row["cid"], {"steps": 0})
                row["steps"] += prev["steps"]
                clients[row["cid"]] = row
            failed = [int(r.metadata.src_node_id) - 1 for r in replies if r.has_error()]
            for cid in failed:
                clients.setdefault(cid, {"cid": cid, "failed": True})
            new_arrays, agg = self.strategy.aggregate_train(edge_round, replies)
            if new_arrays is None:
                break
            arrays = new_arrays
            metrics = dict(agg or {})
            examples = sum(r["n"] for r in stats)
            losses = [(r["loss"], r["n"]) for r in stats]
        if not examples:
            return error_reply(msg, f"edge{self.edge_id}: no valid client update")
        train_loss = sum(l * n for l, n in losses) / max(sum(n for _, n in losses), 1)
        report = {"edge": self.edge_id, "round": rnd, "clients": list(clients.values()),
                  "seconds": time.perf_counter() - start, "edge_metrics": metrics, "alerts": self.alerts}
        self.alerts = []
        self.reports.append({k: v for k, v in report.items() if k != "alerts"})
        content = RecordDict({
            "arrays": arrays,
            "metrics": MetricRecord({"num-examples": int(examples), "train_loss": float(train_loss),
                                     "edge-clients": len(clients)}),
            "report": ConfigRecord({"json": json.dumps(report)}),
        })
        for p in self.plugins:
            content = p.edge_reply(self, rnd, content)
        return Message(content=content, reply_to=msg)

    def evaluate(self, msg):
        clients = self.plan.get("clients", [])
        if not clients:
            return error_reply(msg, f"edge{self.edge_id}: no clients to evaluate")
        self.grid.active = {c + 1 for c in clients}
        messages = [Message(content=RecordDict({"arrays": msg.content["arrays"],
                                                "config": msg.content.get("config", ConfigRecord())}),
                            dst_node_id=c + 1, message_type=MessageType.EVALUATE) for c in clients]
        replies = [r for r in self.grid.send_and_receive(messages) if not r.has_error()]
        if not replies:
            return error_reply(msg, f"edge{self.edge_id}: evaluation failed")
        n = np.asarray([r.content["metrics"]["num-examples"] for r in replies], dtype=float)
        out = {"num-examples": int(n.sum())}
        for key in ("eval_loss", "eval_acc"):
            vals = np.asarray([r.content["metrics"].get(key, 0.0) for r in replies], dtype=float)
            out[key] = float((vals * n).sum() / n.sum())
        return Message(content=RecordDict({"metrics": MetricRecord(out)}), reply_to=msg)

    def client_stats(self):
        rows, counts = [], []
        for cid in range(self.num_clients):
            _, y = self.backend.client_data(cid, self.num_clients)
            c = np.bincount(np.asarray(y), minlength=10)
            counts.append(c)
        total = max(sum(int(c.sum()) for c in counts), 1)
        for cid, c in enumerate(counts):
            rows.append({"cid": cid, "n_samples": int(c.sum()), "label_counts": c.tolist(),
                         "data_ratio": float(c.sum() / total), "entropy": entropy(c)})
        return rows


class EdgeService:
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.sessions = {}
        self.requests = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.started = time.time()

    def create(self, payload):
        self.sessions[payload["sid"]] = EdgeSession(payload["sid"], self.edge_id, payload)

    def drop(self, sid):
        self.sessions.pop(sid, None)


def create_app(edge_id):
    quiet()
    app = Flask(f"edge{edge_id}")
    svc = EdgeService(edge_id)
    app.config["service"] = svc
    try:
        import psutil

        proc = psutil.Process(os.getpid())
    except ImportError:
        proc = None

    def session_or_404(sid):
        s = svc.sessions.get(sid)
        if s is None:
            return None, (jsonify(error="unknown session"), 404)
        return s, None

    @app.get("/api/v1/health")
    def health():
        info = dict(edge_id=edge_id, status="up", uptime=time.time() - svc.started, sessions=list(svc.sessions),
                    requests=svc.requests, bytes_in=svc.bytes_in, bytes_out=svc.bytes_out)
        if proc is not None:
            info.update(cpu_percent=proc.cpu_percent(interval=None), rss_mb=proc.memory_info().rss / 2 ** 20)
        return jsonify(info)

    @app.post("/api/v1/sessions")
    def create_session():
        svc.bytes_in += request.content_length or 0
        svc.create(request.get_json(force=True))
        return jsonify(ok=True, edge_id=edge_id)

    @app.delete("/api/v1/sessions/<sid>")
    def delete_session(sid):
        svc.drop(sid)
        return jsonify(ok=True)

    @app.post("/api/v1/sessions/<sid>/fault")
    def set_fault(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        body = request.get_json(force=True)
        s.fault = {"down": set(body.get("down", [])), "hang": set(body.get("hang", [])),
                   "hang-sec": float(body.get("hang-sec", 5.0))}
        return jsonify(ok=True)

    @app.get("/api/v1/sessions/<sid>/health")
    def session_health(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        state = s.status(int(request.args.get("round", 0)))
        if state == "down":
            return jsonify(edge_id=edge_id, status="down"), 503
        return jsonify(edge_id=edge_id, status=state, busy=s.lock.locked())

    @app.get("/api/v1/sessions/<sid>/clients/stats")
    def client_stats(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        return jsonify(s.client_stats())

    @app.get("/api/v1/sessions/<sid>/history")
    def history(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        return jsonify(s.reports)

    @app.post("/api/v1/sessions/<sid>/messages")
    def messages(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        svc.requests += 1
        data = request.get_data()
        svc.bytes_in += len(data)
        incoming = wire.unpack(data)
        rnd = 0
        if incoming and incoming[0].has_content() and "config" in incoming[0].content:
            rnd = int(incoming[0].content["config"].get("server-round", 0))
        state = s.status(rnd)
        if state == "down":
            return jsonify(error="edge down"), 503
        if not s.lock.acquire(blocking=False):
            return jsonify(error="edge busy"), 409
        try:
            replies = [s.handle(m) for m in incoming]
            if state == "hang":
                time.sleep(s.fault["hang-sec"])
        finally:
            s.lock.release()
        body = wire.pack(replies)
        svc.bytes_out += len(body)
        return Response(body, mimetype=wire.CONTENT_TYPE)

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hierarchical_fl.edge_server")
    ap.add_argument("--edge-id", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args(argv)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    create_app(args.edge_id).run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
