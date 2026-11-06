import json
import math

import numpy as np
from flwr.app import ConfigRecord

from fl_core.plugins import Plugin
from fl_core.records import to_arrays, to_record
from security_sim import attacks
from security_sim.defense import DefendedStrategy
from security_sim.detectors import precision_recall
from security_sim.keys import KeyStore, sign

CLIENT_ATTACKS = ("data_noise", "label_flip", "model_poison", "sign_flip", "sybil", "relay_tamper",
                  "insider_collusion", "adaptive_collusion")
EDGE_ATTACKS = ("hier_collusion", "edge_relay", "phishing")
ATTACKS = ("none",) + CLIENT_ATTACKS + EDGE_ATTACKS
DEFENSES = ("none", "cert", "hmac", "mfa", "stat", "data_dist", "ocsvm", "kmeans", "monitor", "clip")
AUTH_BYTES = 360

DEFAULTS = {"attack": "none", "ratio": 0.2, "defense": [], "two-tier": True, "scale": 3.0, "noise-std": 1.5,
            "param-std": 0.5, "relay-std": 0.2, "jitter": 0.75, "sybils": 3, "malicious-edges": None,
            "phish-round": 2, "start-round": 1, "secret": "defai-sejong-security", "seed": 0}


class SecurityPlugin(Plugin):
    name = "security"

    def __init__(self, config=None):
        super().__init__(dict(DEFAULTS, **(config or {})))
        c = self.config
        if isinstance(c["defense"], str):
            c["defense"] = [d for d in c["defense"].split(",") if d]
        if c["attack"] not in ATTACKS:
            raise ValueError(f"unknown attack {c['attack']!r}; choose from {', '.join(ATTACKS)}")
        bad = [d for d in c["defense"] if d not in DEFENSES]
        if bad:
            raise ValueError(f"unknown defense {bad}; choose from {', '.join(DEFENSES)}")
        self.keys = KeyStore(c["secret"])
        self.malicious = []
        self.malicious_edges = []
        self.client_log = []
        self.edge_log = []
        self.global_strategy = None
        self.flat_strategy = None
        self.cost = []

    def _num_clients(self, owner):
        if hasattr(owner, "num_clients"):
            return owner.num_clients
        return owner.n

    def setup(self, owner):
        c = self.config
        n = self._num_clients(owner)
        count = int(math.ceil(c["ratio"] * n)) if c["attack"] in CLIENT_ATTACKS and c["attack"] != "sybil" else 0
        self.malicious = sorted(int(i) for i in np.random.default_rng(c["seed"] + 505).permutation(n)[:count])
        m = getattr(owner, "m", None) or len(getattr(owner, "links", [])) or 0
        default_edges = [0, 1] if c["attack"] == "hier_collusion" else [0]
        self.malicious_edges = list(c["malicious-edges"] if c["malicious-edges"] is not None else default_edges)
        if c["attack"] not in EDGE_ATTACKS:
            self.malicious_edges = []
        if hasattr(owner, "roles"):
            for pid in self.malicious:
                owner.roles[pid] = c["attack"]
        if hasattr(owner, "registry") and c["attack"] == "phishing" and self.malicious_edges:
            owner.event(c["phish-round"], "info", "attacker",
                        f"phishing mail sent to operator of edge{self.malicious_edges[0]}")
        self.n, self.m = n, m
        self.malicious_principals = [f"client{p}" for p in self.malicious]
        if c["attack"] == "sybil":
            self.malicious_principals = [f"client{100000 + k - 1}" for k in range(c["sybils"])]

    def active(self, rnd):
        return rnd >= self.config["start-round"]

    def train_hooks(self, owner):
        c = self.config
        hooks = {"report_fn": attacks.data_report} if "data_dist" in c["defense"] else {}
        if c["attack"] == "data_noise":
            hooks["data_transform"] = attacks.only_for(set(self.malicious), attacks.data_noise(c["noise-std"], c["seed"]))
        if c["attack"] == "label_flip":
            hooks["data_transform"] = attacks.only_for(set(self.malicious), attacks.label_flip())
        return hooks

    def client_middleware(self, owner):
        c = self.config
        out = []
        if c["attack"] == "relay_tamper":
            out.append(attacks.RelayTamper(self.malicious, c["relay-std"], c["seed"]))
        out.append(attacks.Signer(self.keys))
        if c["attack"] in ("model_poison", "sign_flip"):
            mode = "gaussian" if c["attack"] == "model_poison" else "sign_flip"
            out.append(attacks.ModelPoison(self.malicious, mode, c["scale"], c["param-std"], c["seed"]))
        return out

    def wrap_strategy(self, strategy, tier, owner):
        c = self.config
        if tier in ("edge", "flat"):
            strategy = DefendedStrategy(strategy, c["defense"], self.keys, "client", owner,
                                        self.malicious_principals, seed=c["seed"])
            if tier == "edge":
                owner.security_strategy = strategy
            else:
                self.flat_strategy = strategy
            if c["attack"] in ("insider_collusion", "adaptive_collusion", "sybil"):
                strategy = attacks.ClientAttackStrategy(strategy, c["attack"], self.malicious, self.keys, c["scale"],
                                                        c["jitter"], c["sybils"], c["seed"], owner)
        elif tier == "global" and c["two-tier"]:
            strategy = DefendedStrategy(strategy, c["defense"], self.keys, "edge", owner,
                                        [f"edge{e}" for e in self.malicious_edges], seed=c["seed"])
            self.global_strategy = strategy
        return strategy

    def edge_reply(self, session, rnd, content):
        c = self.config
        if c["attack"] == "hier_collusion" and session.edge_id in self.malicious_edges and self.active(rnd):
            arrays = attacks.poison_edge_model(to_arrays(content["arrays"]), session.global_in, rnd, c["scale"],
                                               c["seed"])
            content["arrays"] = to_record(arrays)
        sign(content, f"edge{session.edge_id}", rnd, self.keys)
        strat = getattr(session, "security_strategy", None)
        if strat is not None and strat.log:
            report = json.loads(str(content["report"]["json"]))
            report["security"] = [e for e in strat.log if e["round"] > (rnd - 1) * session.edge_rounds]
            content["report"] = ConfigRecord({"json": json.dumps(report)})
        return content

    def transport(self, server, rnd, edge_id, reply):
        c = self.config
        if reply.has_error() or edge_id not in self.malicious_edges or not self.active(rnd):
            return reply
        if c["attack"] == "edge_relay":
            arrays = to_arrays(reply.content["arrays"])
            rng = np.random.default_rng([c["seed"], rnd, edge_id, 5])
            arrays = {k: (v + rng.normal(0, c["relay-std"], v.shape)).astype(v.dtype)
                      if np.issubdtype(v.dtype, np.floating) else v for k, v in arrays.items()}
            reply.content["arrays"] = to_record(arrays)
        if c["attack"] == "phishing" and rnd > c["phish-round"]:
            if rnd == c["phish-round"] + 1:
                server.event(rnd, "info", "attacker", f"stolen credentials of edge{edge_id} used to submit a forged model")
            g = server.current_global
            arrays = to_arrays(reply.content["arrays"])
            forged = {k: (g[k] - 2 * c["scale"] * (v - g[k])).astype(v.dtype) if np.issubdtype(v.dtype, np.floating)
                      else v for k, v in arrays.items()}
            reply.content["arrays"] = to_record(forged)
            attacker = KeyStore("attacker-device")
            sign(reply.content, f"edge{edge_id}", rnd, self.keys, mac_key=self.keys.mac_key(f"edge{edge_id}"),
                 device_secret=attacker.device_secret("x"), device_id=attacker.device_id("x"))
            if "monitor" in c["defense"]:
                server.event(rnd, "alert", "monitor", f"edge{edge_id} credentials used from unknown device "
                                                      f"{attacker.device_id('x')}")
        return reply

    def on_round_end(self, owner, rnd, row):
        if rnd == 0:
            return
        entries = []
        if hasattr(owner, "last_reports"):
            for eid, rep in owner.last_reports.items():
                entries += rep.get("security", [])
        elif self.flat_strategy is not None:
            entries = [e for e in self.flat_strategy.log if e["round"] == rnd]
        self.client_log += entries
        flagged = sum(len(e["flagged"]) for e in entries)
        row["sec_flagged_clients"] = flagged
        if self.global_strategy is not None:
            g = [e for e in self.global_strategy.log if e["round"] == rnd]
            self.edge_log += g
            row["sec_flagged_edges"] = sum(len(e["flagged"]) for e in g)
        model_bytes = getattr(owner, "model_bytes", 0) or 0
        if not model_bytes and self.flat_strategy is not None and self.flat_strategy.global_arrays:
            model_bytes = int(sum(v.nbytes for v in self.flat_strategy.global_arrays.values()))
        participants = sum(len(e["population"]) for e in entries) or row.get("matched", 0)
        signed = participants + len(owner.last_reports) if hasattr(owner, "last_reports") else participants
        overhead = AUTH_BYTES * signed if self.config["defense"] else 0
        client_edge = 2 * model_bytes * participants
        upper = row.get("bytes_up", 0) + row.get("bytes_down", 0)
        wasted = model_bytes * flagged
        cost = {"round": rnd, "client_edge_bytes": client_edge, "edge_global_bytes": upper,
                "auth_overhead_bytes": overhead, "rejected_bytes": wasted,
                "total_mb": (client_edge + upper + overhead) / 2 ** 20}
        self.cost.append(cost)
        row["net_cost_mb"] = cost["total_mb"]
        row["sec_overhead_kb"] = overhead / 1024

    def _detection(self, log, principals):
        flagged, population = set(), set()
        for e in log:
            flagged |= set(e["flagged"])
            population |= set(e["population"])
        pr = precision_recall(flagged, principals, population)
        pr["flagged"] = sorted(flagged)
        return pr

    def summary(self, owner):
        out = {"attack": self.config["attack"], "defense": self.config["defense"], "malicious_clients": self.malicious,
               "malicious_edges": self.malicious_edges, "cost": self.cost}
        if self.client_log:
            out["client_detection"] = self._detection(self.client_log, self.malicious_principals)
        if self.edge_log:
            out["edge_detection"] = self._detection(self.edge_log, [f"edge{e}" for e in self.malicious_edges])
        return out
