import numpy as np

from fl_core.plugins import Plugin

SCENARIOS = {
    "ref": {"title": "기준 환경 (REF)"},
    "network_delay": {"title": "S1 네트워크 지연", "start": 2, "growth": 0.15, "spike-prob": 0.1, "spike-factor": 5.0,
                      "deadline": "adaptive"},
    "participation": {"title": "S2 클라이언트 참여 비율 변동", "low": 0.3, "high": 1.0, "dropout": 0.1},
    "imbalance": {"title": "S3 데이터 불균형", "partitioner": "quantity", "sigma": 1.0, "alpha": 0.3},
    "edge_failure": {"title": "S4 엣지 서버 장애", "edge": 1, "from": 2, "to": 3},
    "insider": {"title": "S5 내부자 협력 공격", "ratio": 0.2, "defense": ["monitor"]},
}


class ScenarioPlugin(Plugin):
    name = "scenario"

    def __init__(self, config=None):
        config = dict(config or {})
        kind = config.get("scenario", "ref")
        if kind not in SCENARIOS:
            raise ValueError(f"unknown scenario {kind!r}; choose from {', '.join(SCENARIOS)}")
        super().__init__({**SCENARIOS[kind], **config, "scenario": kind})
        self.trace = []

    @property
    def kind(self):
        return self.config["scenario"]

    def configure(self, cfg):
        c = self.config
        if self.kind == "network_delay" and cfg["deadline"]["policy"] == "none":
            cfg["deadline"]["policy"] = c["deadline"]
        if self.kind == "imbalance":
            bc = dict(cfg.get("backend-config") or {})
            bc["partitioner"] = c["partitioner"]
            if c["partitioner"] == "quantity":
                bc["quantity-sigma"] = c["sigma"]
            else:
                bc["dirichlet-alpha"] = c["alpha"]
            cfg["backend-config"] = bc
        if self.kind == "insider":
            cfg.setdefault("plugins", [])
            cfg["plugins"] = list(cfg["plugins"]) + [{"target": "security_sim.plugin:SecurityPlugin",
                                                      "config": {"attack": "insider_collusion", "ratio": c["ratio"],
                                                                 "defense": c["defense"]}}]
        return cfg

    def setup(self, owner):
        self.rng = np.random.default_rng([owner.backend.cfg["seed"] if hasattr(owner, "backend") else 0, 909])

    def round_state(self, server, rnd, state):
        c = self.config
        if self.kind == "network_delay" and rnd >= c["start"]:
            state["delay_factor"] = 1.0 + c["growth"] * (rnd - c["start"] + 1)
            spikes = np.where(self.rng.random(server.n) < c["spike-prob"], c["spike-factor"], 1.0)
            state["spikes"] = spikes
        if self.kind == "participation":
            rate = float(self.rng.uniform(c["low"], c["high"]))
            k = max(1, int(round(rate * server.n)))
            chosen = set(int(i) for i in self.rng.permutation(server.n)[:k])
            state["participating"] = chosen
            state["drop"] = set(i for i in chosen if self.rng.random() < c["dropout"])
            state["participation_rate"] = rate
        if self.kind == "edge_failure" and c["from"] <= rnd <= c["to"]:
            state["down_edges"] = set(state["down_edges"]) | {c["edge"]}
        self.trace.append({"round": rnd, "delay_factor": state["delay_factor"],
                           "participating": len(state["participating"]), "drop": len(state["drop"]),
                           "down_edges": sorted(state["down_edges"])})
        return state

    def on_round_end(self, owner, rnd, row):
        if rnd == 0:
            return
        t = next((x for x in self.trace if x["round"] == rnd), None)
        if t:
            row["scenario"] = self.kind
            row["participation"] = t["participating"]
            row["down_edges"] = t["down_edges"]

    def summary(self, owner):
        return {"scenario": self.kind, "title": self.config["title"], "trace": self.trace}
