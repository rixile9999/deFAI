"""Fail-fast validation of the run config (``[tool.flwr.app.config]`` plus ``--run-config`` overrides).

Without it a typo such as ``attack='signflip'`` silently runs a clean experiment and an
out-of-range value only fails deep inside a round.
"""

from logging import WARNING
import math
from numbers import Real

from flwr.supercore import log

from defai.attacks import ATTACKS
from defai.compression import METHODS
from defai.strategies import STRATEGIES
from defai.task import DATASETS

CHOICES = {
    "strategy": tuple(STRATEGIES),
    "dataset": tuple(DATASETS),
    "partitioner": ("iid", "dirichlet"),
    "attack": ATTACKS,
    "dp": ("none", "fixed", "adaptive"),
    "dp-side": ("server", "client"),
    "compression": METHODS,
}

# key -> (predicate, description)
RANGES = {
    "seed": (lambda v: v >= 0, ">= 0"),
    "num-server-rounds": (lambda v: v >= 1, ">= 1"),
    "fraction-train": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "fraction-evaluate": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "local-epochs": (lambda v: v >= 1, ">= 1"),
    "batch-size": (lambda v: v >= 1, ">= 1"),
    "learning-rate": (lambda v: v > 0, "> 0"),
    "lr-decay": (lambda v: 0 < v <= 1, "in (0, 1]"),
    "dirichlet-alpha": (lambda v: v > 0, "> 0"),
    "proximal-mu": (lambda v: v >= 0, ">= 0"),
    "moon-mu": (lambda v: v >= 0, ">= 0"),
    "qfedavg-q": (lambda v: v >= 0, ">= 0"),
    "qfedavg-lipschitz": (lambda v: v >= 0, ">= 0"),
    "moon-temperature": (lambda v: v > 0, "> 0"),
    "server-learning-rate": (lambda v: v > 0, "> 0"),
    "server-momentum": (lambda v: 0 <= v < 1, "in [0, 1)"),
    "fedopt-learning-rate": (lambda v: v > 0, "> 0"),
    "beta-1": (lambda v: 0 <= v < 1, "in [0, 1)"),
    "beta-2": (lambda v: 0 <= v < 1, "in [0, 1)"),
    "tau": (lambda v: v > 0, "> 0"),
    "trim-ratio": (lambda v: 0 <= v < 0.5, "in [0, 0.5)"),
    "num-byzantine": (lambda v: v >= 0, ">= 0"),
    "multikrum-select": (lambda v: v >= 1, ">= 1"),
    "num-malicious": (lambda v: v >= 0, ">= 0"),
    "dropout-rate": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "personalize-epochs": (lambda v: v >= 0, ">= 0"),
    "compression-ratio": (lambda v: 0 < v <= 1, "in (0, 1]"),
    "compression-bits": (lambda v: 2 <= v <= 16, "in [2, 16]"),
    "dp-noise-multiplier": (lambda v: v >= 0, ">= 0"),
    "dp-clip-norm": (lambda v: v > 0, "> 0"),
    "dp-delta": (lambda v: 0 < v < 1, "in (0, 1)"),
    "dp-max-epsilon": (lambda v: v >= 0, ">= 0 (0 = no limit)"),
    "dp-target-quantile": (lambda v: 0 <= v <= 1, "in [0, 1]"),
    "dp-clip-lr": (lambda v: v > 0, "> 0"),
    "dp-clipped-count-stddev": (lambda v: v >= 0, ">= 0 (0 = default)"),
}


def validate_config(cfg) -> None:
    """Raise ``ValueError`` listing every problem found in ``cfg``."""
    errors = []
    for key, allowed in CHOICES.items():
        if key in cfg and cfg[key] not in allowed:
            errors.append(f"{key}={cfg[key]!r}: expected one of {', '.join(map(str, allowed))}")
    for key, (ok, expected) in RANGES.items():
        if key in cfg:
            value = cfg[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not ok(value):
                errors.append(f"{key}={value!r}: expected a finite number {expected}")
    for key in ("seed", "num-server-rounds", "local-epochs", "batch-size", "num-byzantine", "multikrum-select", "num-malicious", "personalize-epochs", "compression-bits"):
        if key in cfg and (isinstance(cfg[key], bool) or not isinstance(cfg[key], int)):
            errors.append(f"{key} must be an integer")
    for key in ("save-model", "error-feedback"):
        if key in cfg and not isinstance(cfg[key], bool):
            errors.append(f"{key} must be a boolean")

    name = cfg.get("strategy")
    strategy = STRATEGIES.get(name) if isinstance(name, str) else None
    if cfg.get("dp", "none") != "none" and strategy is not None and not strategy.supports_dp:
        errors.append(f"dp={cfg['dp']!r} cannot be combined with strategy={cfg['strategy']!r} (needs a mean-based strategy)")
    if errors:
        raise ValueError("invalid run config:\n  - " + "\n  - ".join(errors))

    if cfg.get("attack", "none") == "none" and cfg.get("num-malicious", 0) > 0:
        log(WARNING, "num-malicious=%d has no effect while attack='none'", cfg["num-malicious"])
    if cfg.get("attack", "none") != "none" and cfg.get("num-malicious", 0) == 0:
        log(WARNING, "attack=%r has no effect while num-malicious=0", cfg["attack"])
