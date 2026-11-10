from fl_core.plugins import Plugin
from data_imbalance.sampling import METHODS, resampling_transform
from data_imbalance.weighting import MODES, DynamicWeightStrategy

DEFAULTS = {"sampling": "combo", "weighting": "size_loss", "alpha": 1.0, "beta": 1.0, "k": 5, "target-ratio": 1.0,
            "tiers": ["flat", "edge", "global"], "seed": 0}


class ImbalancePlugin(Plugin):
    name = "data_imbalance"

    def __init__(self, config=None):
        super().__init__({**DEFAULTS, **(config or {})})
        if self.config["sampling"] not in METHODS:
            raise ValueError(f"unknown sampling {self.config['sampling']!r}")
        if self.config["weighting"] not in MODES + ("none",):
            raise ValueError(f"unknown weighting {self.config['weighting']!r}")
        self.strategies = []

    def train_hooks(self, owner):
        c = self.config
        if c["sampling"] == "none":
            return {}
        return {"data_transform": resampling_transform(c["sampling"], c["k"], c["target-ratio"], c["seed"])}

    def wrap_strategy(self, strategy, tier, owner):
        c = self.config
        if c["weighting"] in ("none", "size") or tier not in c["tiers"]:
            return strategy
        s = DynamicWeightStrategy(strategy, c["weighting"], c["alpha"], c["beta"])
        self.strategies.append(s)
        return s

    def summary(self, owner):
        return {"sampling": self.config["sampling"], "weighting": self.config["weighting"],
                "weight_log": [e for s in self.strategies for e in s.log][-5:]}
