import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "src")))

from defai_addons.fl.recorder import RunRecorder, _jsonable
from defai_addons.monitor.simulator import HierarchicalFLSimulator


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(_jsonable(body)).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


class RestRecorder(RunRecorder):
    def __init__(self, base, config):
        super().__init__(config, name="external")
        self.base, self.remote_id, self._sent_logs = base, None, 0

    def set_topology(self, topology, clients_static):
        super().set_topology(topology, clients_static)
        self.remote_id = post(self.base + "/api/ext/runs", {"name": "external-" + self.config.get("algorithm", ""),
                                                            "config": self.config, "topology": topology,
                                                            "clients": clients_static})["id"]
        for c in clients_static[:3]:
            r = post(self.base + "/api/ext/client_stats", {"client_id": c["id"], "n": c["n"], "class_counts": c["class_counts"]})
            print("client_stats", r["client_id"], "ratio=%.3f entropy=%.3f" % (r["ratio"], r["entropy"]))

    def add_round(self, rec):
        super().add_round(rec)
        body = dict(rec)
        body["logs"] = self.logs[self._sent_logs:]
        body["sketch"] = self.sketch
        self._sent_logs = len(self.logs)
        post(self.base + "/api/ext/runs/%s/rounds" % self.remote_id, body)

    def set_status(self, status, error=None, summary=None):
        super().set_status(status, error, summary)
        if status == "finished" and self.remote_id:
            post(self.base + "/api/ext/runs/%s/finish" % self.remote_id, {"summary": summary or {}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--algorithm", default="fedprox")
    a = ap.parse_args()
    cfg = {"rounds": a.rounds, "algorithm": a.algorithm, "n_malicious": 1, "sleep": 0.3}
    rec = RestRecorder(a.url.rstrip("/"), cfg)
    sim = HierarchicalFLSimulator(cfg, rec)
    sim.run()
    print("완료: 대시보드에서 실행 id %s 를 선택하세요" % rec.remote_id)


if __name__ == "__main__":
    main()
