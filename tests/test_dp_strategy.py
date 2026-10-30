"""Differential privacy wired into the strategies, driven through Strategy.start()."""

import numpy as np
import pytest
from flwr.app import ConfigRecord, MessageType

from defai.privacy.accountant import RdpAccountant
from defai.privacy.dp import KEY_CLIP_NORM, AdaptiveClipping, FixedClipping
from defai.records import to_record
from defai.strategies import FedAdam, FedAvg, FedAvgM, Moon
from conftest import FakeGrid, QuadraticClients, run_strategy

N = 10
DIM = 4
COMMON = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)
TARGETS = {n: np.full(DIM, float(n)) for n in range(1, N + 1)}


def train_messages(grid):
    return [m for m in grid.sent if m.metadata.message_type == MessageType.TRAIN]


def test_dp_aggregation_uses_uniform_weights_not_example_counts():
    skewed = {n: (1000 if n == 1 else 1) for n in TARGETS}
    clients = QuadraticClients(TARGETS, lr=0.5, steps=1, num_examples=skewed)
    dp = FixedClipping(0.0, clip_norm=1e6)
    w_dp, _, _ = run_strategy(FedAvg(dp=dp, **COMMON), clients, N, rounds=1, dim=DIM)
    w_plain, _, _ = run_strategy(FedAvg(**COMMON), QuadraticClients(TARGETS, lr=0.5, steps=1, num_examples=skewed), N, 1, DIM)
    uniform = np.mean([0.5 * TARGETS[n] for n in TARGETS], axis=0)
    np.testing.assert_allclose(w_dp, uniform, atol=1e-5)
    assert not np.allclose(w_plain, uniform, atol=1e-2)  # without DP, client 1 dominates


def test_noise_is_calibrated_to_clients_per_round():
    dim = 200_000
    targets = {n: np.zeros(dim) for n in range(1, N + 1)}  # zero updates: the release is pure noise
    clients = QuadraticClients(targets, lr=0.5, steps=1)
    dp = FixedClipping(2.0, clip_norm=3.0, seed=0)
    strategy = FedAvg(dp=dp, **{**COMMON, "fraction_train": 0.5})  # 5 clients per round
    w, result, _ = run_strategy(strategy, clients, N, rounds=1, dim=dim)
    assert w.std() == pytest.approx(2.0 * 3.0 / 5, rel=0.01)
    assert result.train_metrics_clientapp[1]["dp-noise-std"] == pytest.approx(2.0 * 3.0 / 5)


def test_epsilon_in_metrics_matches_an_independent_accountant():
    clients = QuadraticClients(TARGETS)
    dp = FixedClipping(1.1, 1.0, delta=1e-5, seed=0)
    _, result, _ = run_strategy(FedAvg(dp=dp, **COMMON), clients, N, rounds=6, dim=DIM)
    ref = RdpAccountant()
    for rnd in range(1, 7):
        ref.compose(1.1)
        assert result.train_metrics_clientapp[rnd]["dp-epsilon"] == pytest.approx(ref.get_epsilon(1e-5))
    assert dp.accountant.num_releases == 6


def test_privacy_budget_stops_training_but_not_evaluation():
    clients = QuadraticClients(TARGETS)
    probe = RdpAccountant()
    probe.compose(1.0, 3)
    limit = probe.get_epsilon(1e-5) + 1e-9  # exactly three rounds fit
    dp = FixedClipping(1.0, 1.0, max_epsilon=limit, seed=0)
    w, result, grid = run_strategy(FedAvg(dp=dp, **COMMON), clients, N, rounds=8, dim=DIM)
    assert len(train_messages(grid)) == 3 * N  # rounds 4..8 sent no train messages
    assert dp.epsilon <= limit
    evals = [m for m in grid.sent if m.metadata.message_type == MessageType.EVALUATE]
    assert len(evals) == 8 * N  # evaluation continues, including after the last trained round
    assert sorted(result.train_metrics_clientapp) == [1, 2, 3]


def test_noise_precedes_the_server_optimiser():
    """The optimiser must only ever see the already-noised update (it is post-processing)."""
    seen = {}

    class Spy(FedAvgM):
        def update_global(self, server_round, delta):
            seen[server_round] = delta
            return super().update_global(server_round, delta)

    dp = FixedClipping(1.0, 1.0, seed=3)
    released = {}
    original = dp.aggregate

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        released[len(released) + 1] = out[0]
        return out

    dp.aggregate = spy
    run_strategy(Spy(momentum=0.9, dp=dp, **COMMON), QuadraticClients(TARGETS), N, rounds=3, dim=DIM)
    assert sorted(seen) == [1, 2, 3]
    for rnd in seen:
        assert all(np.array_equal(seen[rnd][k], released[rnd][k]) for k in seen[rnd])


@pytest.mark.parametrize("cls,kwargs", [(FedAvgM, dict(momentum=0.9)), (FedAdam, {}), (Moon, {})])
def test_other_mean_based_strategies_accept_dp(cls, kwargs):
    dp = FixedClipping(0.5, 1.0, seed=0)
    w, result, _ = run_strategy(cls(dp=dp, **kwargs, **COMMON), QuadraticClients(TARGETS), N, rounds=2, dim=DIM)
    assert np.all(np.isfinite(w)) and "dp-epsilon" in result.train_metrics_clientapp[1]


def test_client_side_mode_asks_clients_to_clip():
    dp = FixedClipping(0.0, clip_norm=2.5, side="client")
    _, _, grid = run_strategy(FedAvg(dp=dp, **COMMON), QuadraticClients(TARGETS), N, rounds=1, dim=DIM)
    config = train_messages(grid)[0].content["config"]
    assert config[KEY_CLIP_NORM] == 2.5 and config["dp-adaptive"] is False
    # evaluation messages carry no clipping instruction
    assert KEY_CLIP_NORM not in [m for m in grid.sent if m.metadata.message_type == MessageType.EVALUATE][0].content["config"]


def test_server_side_adaptive_clipping_tracks_the_median_update_norm():
    clients = QuadraticClients(TARGETS, lr=0.5, steps=1)
    dp = AdaptiveClipping(0.0, clip_norm=0.1, target_quantile=0.5, clip_lr=0.3, seed=0)
    w, _, _ = run_strategy(FedAvg(dp=dp, **COMMON), clients, N, rounds=60, dim=DIM)
    # Update norms shrink as the model converges, so compare with the *final* norms:
    # client n sends 0.5 * (t_n - w), whose norm is |n - 5.5| once w has reached the mean target.
    norms = [0.5 * np.linalg.norm(TARGETS[n] - w) for n in TARGETS]
    assert dp.clip_norm == pytest.approx(np.median(norms), rel=0.15)
    assert dp.clip_norm == pytest.approx(2.5, rel=0.15)


def test_missing_replies_keep_the_fixed_denominator():
    """If only 4 of 10 sampled clients answer, the mean still divides by 10 (sensitivity C / 10)."""
    clients = QuadraticClients(TARGETS, lr=0.5, steps=1)
    strategy = FedAvg(dp=FixedClipping(0.0, clip_norm=1e6), **COMMON)
    grid = FakeGrid(N, clients)
    messages = list(strategy.configure_train(1, to_record({"w": np.zeros(DIM, np.float32)}), ConfigRecord({"lr": 0.5}), grid))
    assert len(messages) == N
    answered = messages[:4]
    arrays, _ = strategy.aggregate_train(1, [clients(m) for m in answered])
    expected = np.sum([0.5 * TARGETS[m.metadata.dst_node_id] for m in answered], axis=0) / N
    np.testing.assert_allclose(arrays["w"].numpy(), expected, atol=1e-5)


def test_adaptive_clipping_with_default_settings_works_for_a_small_federation():
    """Regression: z=1 with 5 clients per round used to crash in round 1 (needs z < m / 10)."""
    dp = AdaptiveClipping(1.0, clip_norm=1.0, seed=0)
    strategy = FedAvg(dp=dp, **{**COMMON, "fraction_train": 0.5})
    _, result, _ = run_strategy(strategy, QuadraticClients(TARGETS), N, rounds=2, dim=DIM)
    assert "dp-epsilon" in result.train_metrics_clientapp[2]


def test_infeasible_adaptive_settings_fail_before_any_client_trains():
    dp = AdaptiveClipping(1.0, clip_norm=1.0, clipped_count_stddev=0.25)
    strategy = FedAvg(dp=dp, **COMMON)
    clients = QuadraticClients(TARGETS)
    grid = FakeGrid(N, clients)
    with pytest.raises(ValueError, match="dp-clipped-count-stddev"):
        strategy.configure_train(1, to_record({"w": np.zeros(DIM, np.float32)}), ConfigRecord({"lr": 0.1}), grid)
    assert grid.sent == []


class _BitClients(QuadraticClients):
    """Honest clients report bit 1 (unclipped); node 1 reports ``bad_bit`` (None = omits it)."""

    def __init__(self, targets, bad_bit, **kw):
        super().__init__(targets, **kw)
        self.bad_bit = bad_bit

    def __call__(self, msg):
        from flwr.app import Message, MetricRecord, RecordDict

        reply = super().__call__(msg)
        if "arrays" not in reply.content:
            return reply
        metrics = dict(reply.content["metrics"])
        if msg.metadata.dst_node_id == 1:
            if self.bad_bit is not None:
                metrics["dp-unclipped-bit"] = self.bad_bit
        else:
            metrics["dp-unclipped-bit"] = 1
        content = dict(reply.content)
        content["metrics"] = MetricRecord(metrics)
        return Message(content=RecordDict(content), reply_to=msg)


@pytest.mark.parametrize("bad_bit", [10**6, -10**6, 2, None])
def test_a_client_cannot_steer_the_clip_norm_with_a_malformed_bit(bad_bit, caplog):
    import logging

    dp = AdaptiveClipping(0.0, clip_norm=1.0, side="client", seed=0)
    strategy = FedAvg(dp=dp, **COMMON)
    with caplog.at_level(logging.WARNING, logger="flwr"):
        w, _, _ = run_strategy(strategy, _BitClients(TARGETS, bad_bit), N, 3, DIM)
    assert "unclipped-bit" in caplog.text
    assert 0.1 < dp.clip_norm < 10 and np.all(np.isfinite(w))  # only well-formed replies steer C
    assert dp.accountant.num_releases == 3


def test_the_exact_clipped_count_of_client_reports_is_not_published():
    dp = AdaptiveClipping(0.5, clip_norm=1.0, side="client", seed=0)
    strategy = FedAvg(dp=dp, **COMMON)
    _, result, _ = run_strategy(strategy, _BitClients(TARGETS, bad_bit=1), N, 2, DIM)
    for metrics in result.train_metrics_clientapp.values():
        assert "dp-unclipped-bit" not in metrics and "dp-clipped-fraction" not in metrics
        assert "dp-noisy-unclipped-fraction" in metrics
