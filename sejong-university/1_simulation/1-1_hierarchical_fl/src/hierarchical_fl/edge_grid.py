import asyncio
import json
import time

import requests
from flwr.app import ConfigRecord, Message, MessageType, RecordDict

from fl_core import wire
from fl_core.grid import BaseGrid, error_reply


class LocalTransport:
    def __init__(self, edge_id):
        from hierarchical_fl.edge_server import create_app

        self.app = create_app(edge_id)
        self.client = self.app.test_client()
        self.url = f"local://edge{edge_id}"

    def get(self, path, params=None, timeout=None):
        r = self.client.get(path, query_string=params or {})
        return r.status_code, r.get_json(silent=True)

    def post_json(self, path, obj, timeout=None):
        r = self.client.post(path, json=obj)
        return r.status_code, r.get_json(silent=True)

    def post_bytes(self, path, data, timeout=None):
        r = self.client.post(path, data=data, content_type=wire.CONTENT_TYPE)
        return r.status_code, r.data

    def delete(self, path):
        self.client.delete(path)


class HttpTransport:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.http = requests.Session()

    def get(self, path, params=None, timeout=10):
        try:
            r = self.http.get(self.url + path, params=params, timeout=timeout)
            return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        except requests.RequestException:
            return 0, None

    def post_json(self, path, obj, timeout=300):
        r = self.http.post(self.url + path, json=obj, timeout=timeout)
        return r.status_code, r.json()

    def post_bytes(self, path, data, timeout=900):
        try:
            r = self.http.post(self.url + path, data=data, headers={"Content-Type": wire.CONTENT_TYPE},
                               timeout=timeout)
            return r.status_code, r.content
        except requests.Timeout:
            return 408, b""
        except requests.RequestException:
            return 0, b""

    def delete(self, path):
        try:
            self.http.delete(self.url + path, timeout=10)
        except requests.RequestException:
            pass


class EdgeLink:
    def __init__(self, edge_id, transport, sid):
        self.edge_id = edge_id
        self.transport = transport
        self.sid = sid

    @property
    def base(self):
        return f"/api/v1/sessions/{self.sid}"

    def exchange(self, message, timeout):
        body = wire.pack([message])
        start = time.perf_counter()
        status, data = self.transport.post_bytes(self.base + "/messages", body, timeout=timeout)
        info = {"eid": self.edge_id, "bytes_down": len(body), "bytes_up": len(data or b""),
                "seconds": time.perf_counter() - start, "http": status}
        if status != 200:
            reason = {0: "unreachable", 408: "timeout", 503: "edge down", 409: "edge busy"}.get(status, f"HTTP {status}")
            info["status"] = reason
            return error_reply(message, f"edge{self.edge_id}: {reason}"), info
        info["status"] = "ok"
        return wire.unpack(data)[0], info


class EdgeGrid(BaseGrid):
    def __init__(self, links, server=None, timeout=900.0):
        super().__init__()
        self.links = {link.edge_id + 1: link for link in links}
        self.server = server
        self.timeout = timeout
        self.available = set(self.links)
        self.last_exchange = []

    def get_node_ids(self):
        return sorted(n for n in self.links if n in self.available)

    def _prepare(self, msg, plan):
        content = RecordDict()
        for k, v in msg.content.items():
            if k == "config":
                cfg = ConfigRecord(dict(v))
                if plan is not None:
                    cfg["plan"] = json.dumps(plan)
                content[k] = cfg
            else:
                content[k] = v
        return Message(content=content, dst_node_id=msg.metadata.dst_node_id,
                       message_type=msg.metadata.message_type, group_id=msg.metadata.group_id)

    async def _gather(self, jobs, timeout):
        return await asyncio.gather(*[asyncio.to_thread(link.exchange, m, timeout) for link, m in jobs])

    def send_and_receive(self, messages, *, timeout=None):
        messages = list(messages)
        if not messages:
            return []
        timeout = min(timeout or self.timeout, self.timeout)
        jobs, originals, replies = [], [], []
        rnd = None
        for msg in messages:
            link = self.links.get(msg.metadata.dst_node_id)
            if link is None:
                replies.append(error_reply(msg, "unknown edge"))
                continue
            plan = None
            if msg.metadata.message_type == MessageType.TRAIN:
                rnd = int(msg.content["config"].get("server-round", 0))
                plan = self.server.plan_for(link.edge_id) if self.server is not None else None
            jobs.append((link, self._prepare(msg, plan)))
            originals.append(msg)
        results = asyncio.run(self._gather(jobs, timeout)) if jobs else []
        infos = []
        for orig, (reply, info) in zip(originals, results):
            if self.server is not None and orig.metadata.message_type == MessageType.TRAIN:
                reply = self.server.transport_hook(rnd, info["eid"], reply)
            if reply.has_error():
                replies.append(Message(reply.error, reply_to=orig))
            else:
                replies.append(Message(content=reply.content, reply_to=orig))
            infos.append(info)
        self.last_exchange = infos
        if self.server is not None and messages[0].metadata.message_type == MessageType.TRAIN:
            self.server.after_exchange(rnd, replies, infos)
        return replies
