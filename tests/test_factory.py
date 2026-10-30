import tomllib
from pathlib import Path

import pytest

from defai.privacy.dp import AdaptiveClipping, FixedClipping
from defai.strategies import STRATEGIES, build_dp, build_strategy

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def default_config(**overrides):
    cfg = tomllib.loads(PYPROJECT.read_text())["tool"]["flwr"]["app"]["config"]
    return {**cfg, **overrides}


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_builds_from_the_default_run_config(name):
    strategy = build_strategy(default_config(strategy=name))
    assert type(strategy) is STRATEGIES[name] and strategy.name == name
    assert strategy.dp is None


def test_unknown_names_are_rejected_with_the_valid_choices():
    with pytest.raises(ValueError, match="fedavg"):
        build_strategy(default_config(strategy="nope"))
    with pytest.raises(ValueError):
        build_dp(default_config(dp="nope"))


def test_dp_modes_follow_the_config():
    assert build_dp(default_config()) is None
    fixed = build_dp(default_config(dp="fixed", **{"dp-side": "client", "dp-clip-norm": 2.0, "dp-noise-multiplier": 0.7}))
    assert isinstance(fixed, FixedClipping) and (fixed.side, fixed.clip_norm, fixed.noise_multiplier) == ("client", 2.0, 0.7)
    adaptive = build_dp(default_config(dp="adaptive", **{"dp-target-quantile": 0.8, "dp-clipped-count-stddev": 3.0}))
    assert isinstance(adaptive, AdaptiveClipping) and adaptive.target_quantile == 0.8 and adaptive.clipped_count_stddev == 3.0
    assert build_dp(default_config(dp="adaptive")).clipped_count_stddev is None  # 0 means "clients / 20"
    assert build_dp(default_config(dp="fixed", **{"dp-max-epsilon": 4.0})).max_epsilon == 4.0
    assert build_dp(default_config(dp="fixed")).max_epsilon is None


def test_hyperparameters_reach_the_strategies():
    cfg = default_config()
    assert build_strategy({**cfg, "strategy": "fedprox", "proximal-mu": 0.3}).proximal_mu == 0.3
    assert build_strategy({**cfg, "strategy": "fedavgm", "server-momentum": 0.5}).momentum == 0.5
    adam = build_strategy({**cfg, "strategy": "fedadam", "fedopt-learning-rate": 0.07, "beta-2": 0.9})
    assert (adam.server_learning_rate, adam.beta_2) == (0.07, 0.9)
    assert build_strategy({**cfg, "strategy": "fedavg", "server-learning-rate": 0.5}).server_learning_rate == 0.5
    mk = build_strategy({**cfg, "strategy": "multikrum", "num-byzantine": 2, "multikrum-select": 4})
    assert (mk.num_byzantine, mk.num_select) == (2, 4)
    assert build_strategy({**cfg, "strategy": "krum"}).num_select == 1
    qfedavg = build_strategy({**cfg, "strategy": "qfedavg", "qfedavg-q": 2.5, "qfedavg-lipschitz": 7.0})
    assert (qfedavg.q, qfedavg.lipschitz) == (2.5, 7.0)
    assert build_strategy({**cfg, "strategy": "qfedavg"}).lipschitz == 0.0  # default: 1 / lr
    assert build_strategy({**cfg, "strategy": "fedtrimmedavg", "trim-ratio": 0.1}).beta == 0.1


def test_dp_is_rejected_for_strategies_that_cannot_support_it():
    for name in ("scaffold", "fednova", "qfedavg", "fedmedian", "fedtrimmedavg", "krum", "multikrum", "bulyan"):
        with pytest.raises(ValueError, match="differential privacy"):
            build_strategy(default_config(strategy=name, dp="fixed"))
    build_strategy(default_config(strategy="fedadam", dp="adaptive"))  # allowed


def test_lr_decay_reaches_every_strategy():
    cfg = default_config()
    for name in STRATEGIES:
        assert build_strategy({**cfg, "strategy": name, "lr-decay": 0.8}).lr_decay == 0.8
