"""Constructors reject nonsensical hyper-parameters, and every strategy can describe itself."""

import pytest

from defai.privacy.accountant import RdpAccountant
from defai.privacy.dp import AdaptiveClipping, FixedClipping
from defai.strategies import (
    STRATEGIES,
    Bulyan,
    FedAdagrad,
    FedAdam,
    FedAvgM,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    Krum,
    Moon,
    MultiKrum,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FedAvgM(momentum=1.0),
        lambda: FedAvgM(momentum=-0.1),
        lambda: FedAdam(beta_1=1.0),
        lambda: FedYogi(beta_2=-0.5),
        lambda: FedAdagrad(tau=0.0),
        lambda: FedProx(proximal_mu=-0.1),
        lambda: Moon(mu=-1.0),
        lambda: Moon(temperature=0.0),
        lambda: FedTrimmedAvg(beta=0.5),
        lambda: FedTrimmedAvg(beta=-0.1),
        lambda: MultiKrum(num_byzantine=-1),
        lambda: MultiKrum(num_select=0),
        lambda: Krum(num_byzantine=-2),
        lambda: Bulyan(num_byzantine=-1),
        lambda: FixedClipping(-1.0, 1.0),
        lambda: FixedClipping(1.0, 0.0),
        lambda: FixedClipping(1.0, 1.0, side="both"),
        lambda: FixedClipping(1.0, 1.0, delta=0.0),
        lambda: FixedClipping(1.0, 1.0, delta=1.0),
        lambda: AdaptiveClipping(1.0, 1.0, target_quantile=1.5),
        lambda: AdaptiveClipping(1.0, 1.0, clip_lr=0.0),
        lambda: AdaptiveClipping(1.0, 1.0, clipped_count_stddev=-1.0),
        lambda: RdpAccountant(orders=[]),
        lambda: RdpAccountant(orders=[1.0, 2.0]),  # order 1 is not allowed
    ],
    ids=lambda f: "",
)
def test_invalid_hyperparameters_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_the_accountant_rejects_an_invalid_delta():
    acc = RdpAccountant()
    acc.compose(1.0)
    for delta in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            acc.get_epsilon(delta)


def test_aggregating_nothing_is_an_error_not_a_zero_update():
    with pytest.raises(ValueError, match="no updates"):
        FixedClipping(1.0, 1.0).aggregate([], num_sampled=4)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_describes_itself(name, caplog):
    import logging

    kwargs = {"krum": dict(num_byzantine=1), "multikrum": dict(num_byzantine=1), "bulyan": dict(num_byzantine=1)}.get(name, {})
    strategy = STRATEGIES[name](**kwargs)
    with caplog.at_level(logging.INFO, logger="flwr"):
        strategy.summary()
    assert type(strategy).__name__ in caplog.text and "Differential privacy: off" in caplog.text


def test_summary_mentions_the_dp_mechanism_when_one_is_set(caplog):
    import logging

    strategy = STRATEGIES["fedavg"](dp=AdaptiveClipping(1.0, 0.5, side="client"))
    with caplog.at_level(logging.INFO, logger="flwr"):
        strategy.summary()
    assert "AdaptiveClipping(side=client, z=1.0, C=0.5" in caplog.text


def test_a_fresh_accountant_spends_nothing_and_budget_checks_follow_the_noise_level():
    acc = RdpAccountant()
    assert acc.get_epsilon(1e-5) == 0.0
    assert not acc.would_exceed(1e6, 1e-5, max_epsilon=1.0)  # enormous noise: one release costs almost nothing
    assert acc.would_exceed(0.01, 1e-5, max_epsilon=1.0)     # almost no noise: one release blows the budget
