import copy
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..common import data as D
from ..common.metrics import bootstrap_diff_ci
from ..fl.recorder import _jsonable
from .envs import FederatedEnv, LocalDistributedEnv, SiteData

DEFAULTS: Dict[str, Any] = dict(
    dataset="mnist", n_sites=5, partition="label_groups", classes_per_client=2, alpha=0.3, quantity_sigma=0.0,
    encrypt_updates=True, max_train=20000, missing_rate=0.02, outlier_rate=0.005, epochs=15, local_epochs=1, lr=0.05, batch_size=32,
    hidden=[128, 64], use_pca=True, mu=0.01, dp_noise=0.3, dp_clip=2.0, bandwidth_mbps=10.0, seed=0,
    envs=["ld_independent", "ld_sync", "ld_central", "fl_fedavg", "fl_fedavg_dp"],
)

ENV_LABELS = {
    "ld_independent": "지역분산 (독립 학습+결과 공유)",
    "ld_sync": "지역분산 (데이터 재분할+동기화)",
    "ld_central": "지역분산 (중앙 집중)",
    "fl_fedavg": "연합 (FedAvg+보안집계)",
    "fl_fedprox": "연합 (FedProx+보안집계)",
    "fl_fedavg_dp": "연합 (FedAvg+보안집계+DP)",
}


def build_sites(cfg: Dict[str, Any]):
    Xtr, ytr, Xte, yte = D.load_split(cfg["dataset"], seed=cfg["seed"], max_train=cfg.get("max_train"))
    Xtr = D.inject_corruption(Xtr, cfg["missing_rate"], cfg["outlier_rate"], seed=cfg["seed"])
    Xte = D.inject_corruption(Xte, cfg["missing_rate"], cfg["outlier_rate"], seed=cfg["seed"] + 1)
    n_classes = int(max(ytr.max(), yte.max())) + 1
    parts = D.partition(ytr, cfg["n_sites"], cfg["partition"], alpha=cfg["alpha"],
                        quantity_sigma=cfg["quantity_sigma"], seed=cfg["seed"],
                        classes_per_client=cfg.get("classes_per_client", 2))
    rng = np.random.RandomState(cfg["seed"])
    sites = []
    for p in parts:
        p = rng.permutation(p)
        nt = max(1, int(0.2 * len(p)))
        sites.append(SiteData(Xtr[p[nt:]], ytr[p[nt:]], Xtr[p[:nt]], ytr[p[:nt]]))
    return sites, Xte, yte, n_classes, D.describe_partition(parts, ytr, n_classes)


def make_env(key: str, sites, Xte, yte, n_classes, cfg):
    args = (ENV_LABELS.get(key, key), sites, Xte, yte, n_classes)
    if key.startswith("ld_"):
        return LocalDistributedEnv(*args, cfg=cfg, strategy=key[3:])
    algo = "fedprox" if "fedprox" in key else "fedavg"
    return FederatedEnv(*args, cfg=cfg, algorithm=algo, dp_noise=cfg["dp_noise"] if key.endswith("_dp") else 0.0,
                        secure_agg=True)


class ComparisonJob:
    def __init__(self, config: Dict[str, Any]):
        self.id = uuid.uuid4().hex[:8]
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg.update({k: v for k, v in config.items() if v is not None})
        self.status, self.error = "created", None
        self.progress: Dict[str, List[Dict[str, float]]] = {}
        self.report: Optional[Dict[str, Any]] = None
        self.created = time.time()
        self._lock = threading.Lock()

    def _on_progress(self, env: str, epoch: int, rec: Dict[str, float]):
        with self._lock:
            self.progress.setdefault(env, []).append(_jsonable(rec))

    def run(self) -> Dict[str, Any]:
        self.status = "running"
        try:
            self.report = run_comparison(self.cfg, progress=self._on_progress)
            self.status = "finished"
        except Exception:
            import traceback
            self.status, self.error = "error", traceback.format_exc()
            raise
        return self.report

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {"id": self.id, "status": self.status, "error": self.error, "config": _jsonable(self.cfg),
                    "progress": copy.deepcopy(self.progress), "report": self.report,
                    "labels": ENV_LABELS, "created": self.created}


def run_comparison(cfg: Dict[str, Any], progress: Optional[Callable] = None) -> Dict[str, Any]:
    c = copy.deepcopy(DEFAULTS)
    c.update(cfg)
    sites, Xte, yte, n_classes, dist = build_sites(c)
    return compare_sites(sites, Xte, yte, c, progress, dist)


def compare_sites(sites, Xte, yte, config=None, progress=None, partition=None):
    c = copy.deepcopy(DEFAULTS)
    c.update(config or {})
    if not sites or not len(yte):
        raise ValueError("At least one site and a nonempty global test set are required")
    n_classes = int(max([np.max(yte)] + [max(np.max(s.y), np.max(s.yte)) for s in sites])) + 1
    dist = partition if partition is not None else [{"id": i, "n": len(s.y)} for i, s in enumerate(sites)]
    c["n_sites"] = len(sites)
    if any(k not in ENV_LABELS for k in c["envs"]) or not c["envs"]:
        raise ValueError("envs must contain supported environment keys")
    results, preds = [], {}
    for key in c["envs"]:
        env = make_env(key, sites, Xte, yte, n_classes, c)
        env.fit(progress)
        ev = env.evaluate()
        preds[key] = ev.pop("pred")
        results.append({
            "key": key, "name": env.name, "kind": env.kind, "global": ev["global"], "local": ev["local"],
            "local_mean": {m: float(np.mean([l[m] for l in ev["local"]])) for m in ("accuracy", "precision", "recall", "f1")},
            "local_min_f1": float(min(l["f1"] for l in ev["local"])),
            "cost": dict(env.cost), "privacy": env.privacy, "history": env.history,
            "secure_agg_error": getattr(env, "secure_agg_error", None),
        })
    ref = next((k for k in c["envs"] if k == "fl_fedavg"), next((k for k in c["envs"] if k.startswith("fl_")), None))
    diffs = []
    if ref is not None:
        for key in c["envs"]:
            if key != ref:
                ci = bootstrap_diff_ci(yte, preds[ref], preds[key], n_classes, "f1", n_boot=200, seed=c["seed"])
                diffs.append({"ref": ref, "other": key, **ci})
    return _jsonable({"config": c, "results": results, "f1_diff_vs_ref": diffs, "partition": dist,
                      "n_test": int(len(yte)), "n_classes": n_classes})
