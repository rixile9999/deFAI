import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, config TEXT, status TEXT, started REAL, finished REAL,
                                 summary TEXT);
CREATE TABLE IF NOT EXISTS clients (run_id TEXT, cid INTEGER, n_samples INTEGER, label_counts TEXT,
                                    data_ratio REAL, entropy REAL, role TEXT, PRIMARY KEY (run_id, cid));
CREATE TABLE IF NOT EXISTS edges (run_id TEXT, eid INTEGER, url TEXT, capacity INTEGER, PRIMARY KEY (run_id, eid));
CREATE TABLE IF NOT EXISTS rounds (run_id TEXT, round INTEGER, data TEXT, PRIMARY KEY (run_id, round));
CREATE TABLE IF NOT EXISTS assignments (run_id TEXT, round INTEGER, cid INTEGER, eid INTEGER, utility REAL,
                                        priority REAL, q_hat REAL, status TEXT, PRIMARY KEY (run_id, round, cid));
CREATE TABLE IF NOT EXISTS edge_rounds (run_id TEXT, round INTEGER, eid INTEGER, status TEXT, bytes_up INTEGER,
                                        bytes_down INTEGER, seconds REAL, virtual_ms REAL, clients INTEGER,
                                        data TEXT, PRIMARY KEY (run_id, round, eid));
CREATE TABLE IF NOT EXISTS events (run_id TEXT, round INTEGER, ts REAL, level TEXT, source TEXT, message TEXT);
"""


class Registry:
    def __init__(self, path):
        self.path = str(path)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def _exec(self, sql, rows):
        with self.lock:
            self.db.executemany(sql, rows)
            self.db.commit()

    def start_run(self, run_id, config):
        self._exec("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                   [(run_id, json.dumps(config), "running", time.time(), None, None)])

    def finish_run(self, run_id, status, summary):
        with self.lock:
            self.db.execute("UPDATE runs SET status=?, finished=?, summary=? WHERE run_id=?",
                            (status, time.time(), json.dumps(summary), run_id))
            self.db.commit()

    def add_clients(self, run_id, rows):
        self._exec("INSERT OR REPLACE INTO clients VALUES (?,?,?,?,?,?,?)",
                   [(run_id, r["cid"], r["n_samples"], json.dumps(r["label_counts"]), r["data_ratio"],
                     r["entropy"], r.get("role", "benign")) for r in rows])

    def add_edges(self, run_id, rows):
        self._exec("INSERT OR REPLACE INTO edges VALUES (?,?,?,?)",
                   [(run_id, r["eid"], r["url"], r["capacity"]) for r in rows])

    def add_round(self, run_id, rnd, data):
        self._exec("INSERT OR REPLACE INTO rounds VALUES (?,?,?)", [(run_id, rnd, json.dumps(data))])

    def add_assignments(self, run_id, rnd, rows):
        self._exec("INSERT OR REPLACE INTO assignments VALUES (?,?,?,?,?,?,?,?)",
                   [(run_id, rnd, r["cid"], r["eid"], r["utility"], r["priority"], r["q_hat"], r["status"])
                    for r in rows])

    def add_edge_rounds(self, run_id, rnd, rows):
        self._exec("INSERT OR REPLACE INTO edge_rounds VALUES (?,?,?,?,?,?,?,?,?,?)",
                   [(run_id, rnd, r["eid"], r["status"], r["bytes_up"], r["bytes_down"], r["seconds"],
                     r["virtual_ms"], r["clients"], json.dumps(r.get("data", {}))) for r in rows])

    def event(self, run_id, rnd, level, source, message):
        self._exec("INSERT INTO events VALUES (?,?,?,?,?,?)", [(run_id, rnd, time.time(), level, source, message)])

    def query(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        with self.lock:
            self.db.close()


def read_only(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    return db


def fetch(db, sql, args=()):
    cur = db.execute(sql, args)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
