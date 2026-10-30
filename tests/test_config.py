import tomllib
from pathlib import Path

import pytest

from defai.config import validate_config

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def cfg(**overrides):
    base = tomllib.loads(PYPROJECT.read_text())["tool"]["flwr"]["app"]["config"]
    return {**base, **overrides}


def test_default_config_is_valid():
    validate_config(cfg())


@pytest.mark.parametrize(
    "overrides,key",
    [
        ({"attack": "signflip"}, "attack"),            # typo that used to run a clean experiment silently
        ({"strategy": "fedavgg"}, "strategy"),
        ({"dataset": "mnits"}, "dataset"),
        ({"partitioner": "noniid"}, "partitioner"),
        ({"dp": "yes"}, "dp"),
        ({"dp-side": "both"}, "dp-side"),
        ({"fraction-train": 1.5}, "fraction-train"),
        ({"server-momentum": 1.0}, "server-momentum"),
        ({"trim-ratio": 0.5}, "trim-ratio"),
        ({"dp-delta": 0.0}, "dp-delta"),
        ({"dp-clip-norm": -1.0}, "dp-clip-norm"),
        ({"num-malicious": -1}, "num-malicious"),
        ({"dropout-rate": 1.5}, "dropout-rate"),
        ({"lr-decay": 0.0}, "lr-decay"),
        ({"qfedavg-q": -1.0}, "qfedavg-q"),
        ({"qfedavg-lipschitz": -1.0}, "qfedavg-lipschitz"),
        ({"compression": "gzip"}, "compression"),
        ({"personalize-epochs": -1}, "personalize-epochs"),
        ({"compression-ratio": 0.0}, "compression-ratio"),
        ({"compression-bits": 1}, "compression-bits"),
        ({"learning-rate": 0.0}, "learning-rate"),
    ],
)
def test_invalid_values_are_named_in_the_error(overrides, key):
    with pytest.raises(ValueError, match=key):
        validate_config(cfg(**overrides))


def test_all_problems_are_reported_together():
    with pytest.raises(ValueError) as err:
        validate_config(cfg(attack="x", **{"dp-delta": 2.0}))
    assert "attack" in str(err.value) and "dp-delta" in str(err.value)


@pytest.mark.parametrize("strategy", ["scaffold", "fednova", "qfedavg", "fedmedian", "fedtrimmedavg", "krum", "multikrum", "bulyan"])
def test_dp_with_a_non_mean_strategy_is_rejected_up_front(strategy):
    with pytest.raises(ValueError, match="mean-based"):
        validate_config(cfg(strategy=strategy, dp="fixed"))
    validate_config(cfg(strategy=strategy, dp="none"))


def test_ineffective_attack_settings_only_warn(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flwr"):
        validate_config(cfg(attack="none", **{"num-malicious": 2}))
        validate_config(cfg(attack="sign-flip", **{"num-malicious": 0}))
    text = caplog.text
    assert "num-malicious=2 has no effect" in text and "attack='sign-flip' has no effect" in text
