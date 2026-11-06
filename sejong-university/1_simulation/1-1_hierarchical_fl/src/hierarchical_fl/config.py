import hashlib
import json

from fl_core.config import merge

DEFAULTS = {
    "run-id": None,
    "backend": "builtin",
    "backend-config": {},
    "num-clients": 12,
    "num-edges": 3,
    "edge-capacity": 6,
    "global-rounds": 5,
    "edge-rounds": 2,
    "edge-strategy": "fedavg",
    "global-strategy": "fedavg",
    "global-evaluate": False,
    "matching": {"mode": "matching", "ema-beta": 0.7, "cost-data": 0.05, "cost-comm": 0.1, "min-quality": 0.0,
                 "trust-rel": 0.0, "edge-reward": None},
    "deadline": {"policy": "none", "ms": 2500.0, "quantile": 0.8, "margin": 1.25},
    "transport": "local",
    "edge-urls": [],
    "spawn-edges": False,
    "edge-port": 6101,
    "client-workers": 1,
    "http-timeout": 900.0,
    "plugins": [],
    "output-dir": "results/hfl",
}


def make_config(overrides=None):
    cfg = merge(DEFAULTS, overrides or {})
    if cfg["num-edges"] * cfg["edge-capacity"] < cfg["num-clients"]:
        cfg["edge-capacity"] = -(-cfg["num-clients"] // cfg["num-edges"])
    if not cfg["run-id"]:
        cfg["run-id"] = "hfl_" + config_hash(cfg)
    return cfg


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("run-id", "edge-urls", "output-dir", "spawn-edges", "transport")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:10]
