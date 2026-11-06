import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from hierarchical_fl.registry import fetch, read_only


def find_runs(root):
    out = []
    for db_path in sorted(Path(root).rglob("hfl.sqlite")):
        try:
            db = read_only(db_path)
            rows = fetch(db, "SELECT run_id, status, started, finished, summary FROM runs")
            db.close()
        except Exception:
            continue
        for r in rows:
            r["path"] = str(db_path)
            r["summary"] = json.loads(r["summary"]) if r["summary"] else {}
            out.append(r)
    return sorted(out, key=lambda r: r["started"] or 0, reverse=True)


def load_run(path, run_id):
    db = read_only(path)
    try:
        run = fetch(db, "SELECT * FROM runs WHERE run_id=?", (run_id,))[0]
        rounds = [json.loads(r["data"]) for r in fetch(db, "SELECT data FROM rounds WHERE run_id=? ORDER BY round",
                                                         (run_id,))]
        clients = fetch(db, "SELECT * FROM clients WHERE run_id=? ORDER BY cid", (run_id,))
        edges = fetch(db, "SELECT * FROM edges WHERE run_id=? ORDER BY eid", (run_id,))
        last = fetch(db, "SELECT MAX(round) AS r FROM assignments WHERE run_id=?", (run_id,))[0]["r"] or 0
        assignments = fetch(db, "SELECT * FROM assignments WHERE run_id=? AND round=? ORDER BY cid", (run_id, last))
        edge_rounds = fetch(db, "SELECT run_id, round, eid, status, bytes_up, bytes_down, seconds, virtual_ms, "
                                "clients FROM edge_rounds WHERE run_id=? ORDER BY round, eid", (run_id,))
        client_hist = fetch(db, "SELECT round, data FROM edge_rounds WHERE run_id=? ORDER BY round", (run_id,))
        events = fetch(db, "SELECT * FROM events WHERE run_id=? ORDER BY ts DESC LIMIT 200", (run_id,))
        status_hist = fetch(db, "SELECT round, cid, eid, status, q_hat FROM assignments WHERE run_id=? "
                                "ORDER BY round, cid", (run_id,))
    finally:
        db.close()
    for c in clients:
        c["label_counts"] = json.loads(c["label_counts"])
    trend = {}
    for row in client_hist:
        for c in json.loads(row["data"]).get("clients", []):
            if "loss" in c:
                trend.setdefault(c["cid"], []).append({"round": row["round"], "loss": c["loss"],
                                                       "q": c.get("q"), "norm": c.get("norm")})
    run["config"] = json.loads(run["config"])
    run["summary"] = json.loads(run["summary"]) if run["summary"] else {}
    matching = next((r.get("matching") for r in reversed(rounds) if r.get("matching")), None)
    return {"run": run, "rounds": [{k: v for k, v in r.items() if k != "matching"} for r in rounds],
            "clients": clients, "edges": edges, "assignments": assignments, "assignment_round": last,
            "edge_rounds": edge_rounds, "client_trend": trend, "events": events, "matching": matching,
            "status_history": status_hist}


def outliers(trend):
    flagged = []
    for cid, pts in trend.items():
        qs = [p["q"] for p in pts if p.get("q") is not None]
        if qs and min(qs) < 0.35:
            flagged.append({"cid": cid, "min_q": min(qs)})
    return flagged


def create_app(results_root):
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
                static_folder=str(Path(__file__).parent / "static"))
    app.config["RESULTS"] = Path(results_root).resolve()
    launched = []

    def locate(run_id):
        for r in find_runs(app.config["RESULTS"]):
            if r["run_id"] == run_id:
                return r["path"]
        abort(404)

    @app.get("/")
    def index():
        return render_template("index.html", runs=find_runs(app.config["RESULTS"]), launched=launched)

    @app.get("/run/<run_id>")
    def run_page(run_id):
        locate(run_id)
        return render_template("run.html", run_id=run_id)

    @app.get("/api/runs")
    def api_runs():
        return jsonify(find_runs(app.config["RESULTS"]))

    @app.get("/api/run/<run_id>")
    def api_run(run_id):
        data = load_run(locate(run_id), run_id)
        data["outliers"] = outliers(data["client_trend"])
        return jsonify(data)

    @app.post("/launch")
    def launch():
        form = request.form
        sets = [f"output-dir={app.config['RESULTS'] / 'hfl'}"]
        for key in ("num-clients", "num-edges", "global-rounds", "edge-rounds", "edge-strategy", "global-strategy",
                    "backend"):
            if form.get(key):
                sets.append(f"{key}={form[key]}")
        for key, target in (("dataset", "backend-config.dataset"), ("partitioner", "backend-config.partitioner"),
                            ("matching", "matching.mode"), ("deadline", "deadline.policy")):
            if form.get(key):
                sets.append(f"{target}={form[key]}")
        cmd = [sys.executable, "-m", "hierarchical_fl", "run"]
        for s in sets:
            cmd += ["--set", s]
        if form.get("transport") == "http":
            cmd.append("--spawn-edges")
        log = app.config["RESULTS"] / f"launch_{int(time.time())}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(cmd, stdout=open(log, "w", encoding="utf-8"),
                                stderr=subprocess.STDOUT)
        launched.append({"pid": proc.pid, "cmd": " ".join(cmd[2:]), "log": str(log), "started": time.time()})
        return redirect(url_for("index"))

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hierarchical_fl.dashboard")
    ap.add_argument("--results", default="results")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args(argv)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"dashboard: http://{args.host}:{args.port}")
    create_app(args.results).run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
