"""Federated strategies and the run-config -> strategy factory."""

from defai.privacy.dp import AdaptiveClipping, FixedClipping
from defai.strategies.base import ClientUpdate, FedAvg
from defai.strategies.fedopt import FedAdagrad, FedAdam, FedAvgM, FedYogi
from defai.strategies.heterogeneity import FedNova, FedProx, Moon, QFedAvg, Scaffold
from defai.strategies.robust import Bulyan, FedMedian, FedTrimmedAvg, Krum, MultiKrum

STRATEGIES = {
    cls.name: cls
    for cls in (
        FedAvg, FedProx, Moon, Scaffold, FedNova, QFedAvg,
        FedAvgM, FedAdagrad, FedAdam, FedYogi,
        FedMedian, FedTrimmedAvg, Krum, MultiKrum, Bulyan,
    )
}
FEDOPT = ("fedadagrad", "fedadam", "fedyogi")

__all__ = [
    "STRATEGIES", "build_strategy", "build_dp", "ClientUpdate",
    "FedAvg", "FedProx", "Moon", "Scaffold", "FedNova", "QFedAvg", "FedAvgM", "FedAdagrad", "FedAdam", "FedYogi",
    "FedMedian", "FedTrimmedAvg", "Krum", "MultiKrum", "Bulyan",
]


def build_dp(cfg):
    """Differential-privacy mechanism selected by ``dp`` in the run config, or None."""
    kind = cfg["dp"]
    if kind == "none":
        return None
    common = dict(
        side=cfg["dp-side"],
        delta=cfg["dp-delta"],
        max_epsilon=cfg["dp-max-epsilon"] if cfg["dp-max-epsilon"] > 0 else None,
        seed=cfg["seed"],
    )
    if kind == "fixed":
        return FixedClipping(cfg["dp-noise-multiplier"], cfg["dp-clip-norm"], **common)
    if kind == "adaptive":
        return AdaptiveClipping(
            cfg["dp-noise-multiplier"],
            cfg["dp-clip-norm"],
            target_quantile=cfg["dp-target-quantile"],
            clip_lr=cfg["dp-clip-lr"],
            clipped_count_stddev=cfg["dp-clipped-count-stddev"] if cfg["dp-clipped-count-stddev"] > 0 else None,
            **common,
        )
    raise ValueError(f"unknown dp mode: {kind!r} (choose from none, fixed, adaptive)")


def build_strategy(cfg):
    """Instantiate the strategy named by ``strategy`` in the run config."""
    name = cfg["strategy"]
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name!r} (choose from {', '.join(STRATEGIES)})")
    kwargs = dict(
        fraction_train=cfg["fraction-train"],
        fraction_evaluate=cfg["fraction-evaluate"],
        server_learning_rate=cfg["fedopt-learning-rate"] if name in FEDOPT else cfg["server-learning-rate"],
        lr_decay=cfg["lr-decay"],
        dp=build_dp(cfg),
        seed=cfg["seed"],
    )
    kwargs.update(
        {
            "fedprox": dict(proximal_mu=cfg["proximal-mu"]),
            "moon": dict(mu=cfg["moon-mu"], temperature=cfg["moon-temperature"]),
            "qfedavg": dict(q=cfg["qfedavg-q"], lipschitz=cfg["qfedavg-lipschitz"]),
            "fedavgm": dict(momentum=cfg["server-momentum"]),
            "fedadagrad": dict(beta_1=cfg["beta-1"], beta_2=cfg["beta-2"], tau=cfg["tau"]),
            "fedadam": dict(beta_1=cfg["beta-1"], beta_2=cfg["beta-2"], tau=cfg["tau"]),
            "fedyogi": dict(beta_1=cfg["beta-1"], beta_2=cfg["beta-2"], tau=cfg["tau"]),
            "fedtrimmedavg": dict(beta=cfg["trim-ratio"]),
            "krum": dict(num_byzantine=cfg["num-byzantine"]),
            "multikrum": dict(num_byzantine=cfg["num-byzantine"], num_select=cfg["multikrum-select"]),
            "bulyan": dict(num_byzantine=cfg["num-byzantine"]),
        }.get(name, {})
    )
    return STRATEGIES[name](**kwargs)
