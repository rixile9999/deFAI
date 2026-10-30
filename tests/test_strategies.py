import numpy as np
import pytest
from flwr.app import Message, MetricRecord, RecordDict

from conftest import FakeGrid
from defai.privacy.dp import FixedClipping
from defai.records import to_record
from defai.strategies import (
    Bulyan,
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedNova,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    Krum,
    Moon,
    MultiKrum,
    QFedAvg,
    Scaffold,
)
from conftest import QuadraticClients, run_strategy

N = 8
DIM = 4
RNG = np.random.default_rng(0)
TARGETS = {n: RNG.normal(0, 1.0, DIM) for n in range(1, N + 1)}
OPT = np.mean(list(TARGETS.values()), axis=0)  # optimum of the average objective
COMMON = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)


def distance(w):
    return float(np.linalg.norm(w - OPT))


def run(strategy, rounds=40, **client_kwargs):
    clients = QuadraticClients(TARGETS, **client_kwargs)
    return run_strategy(strategy, clients, N, rounds, DIM)


# ---------------------------------------------------------------------------- FedAvg family
def test_fedavg_one_round_is_example_weighted_mean_of_client_models():
    num_examples = {n: n for n in TARGETS}  # unequal weights 1..8
    clients = QuadraticClients(TARGETS, num_examples=num_examples, lr=0.5, steps=1)
    w, result, _ = run_strategy(FedAvg(**COMMON), clients, N, rounds=1, dim=DIM)
    models = {n: 0.5 * TARGETS[n] for n in TARGETS}  # one step of lr 0.5 from w=0
    weights = np.array([num_examples[n] for n in TARGETS], dtype=float)
    expected = sum(w_ * models[n] for w_, n in zip(weights / weights.sum(), TARGETS))
    np.testing.assert_allclose(w, expected, atol=1e-6)
    assert result.train_metrics_clientapp[1]["train_loss"] == pytest.approx(
        sum(w_ * 0.5 * np.sum(TARGETS[n] ** 2) for w_, n in zip(weights / weights.sum(), TARGETS))
    )


def test_fedavg_converges_to_the_optimum():
    w, _, _ = run(FedAvg(**COMMON))
    assert distance(w) < 1e-4


def test_server_learning_rate_scales_the_update():
    w1, _, _ = run(FedAvg(server_learning_rate=1.0, **COMMON), rounds=1, lr=0.5, steps=1)
    w2, _, _ = run(FedAvg(server_learning_rate=0.5, **COMMON), rounds=1, lr=0.5, steps=1)
    np.testing.assert_allclose(w2, 0.5 * w1, atol=1e-6)


def test_fedprox_sends_mu_and_converges_to_the_same_optimum():
    strategy = FedProx(proximal_mu=0.5, **COMMON)
    w, _, grid = run(strategy, rounds=80)
    assert grid.sent[0].content["config"]["proximal-mu"] == 0.5
    assert distance(w) < 1e-3


def test_moon_sends_its_hyperparameters():
    _, _, grid = run(Moon(mu=2.0, temperature=0.3, **COMMON), rounds=1)
    cfg = grid.sent[0].content["config"]
    assert (cfg["moon-mu"], cfg["moon-temperature"]) == (2.0, 0.3)


def test_scaffold_converges_and_sends_the_control_variate():
    w, _, grid = run(Scaffold(**COMMON), rounds=60)
    assert "c" in grid.sent[0].content  # control variate travels with the model
    assert distance(w) < 1e-4


def test_scaffold_server_control_variate_is_the_average_client_control_variate():
    clients = QuadraticClients(TARGETS)
    strategy = Scaffold(**COMMON)
    run_strategy(strategy, clients, N, rounds=5, dim=DIM)
    # full participation: c = mean_i c_i exactly (c_i starts at 0, c += |S|/N * mean(dc_i))
    np.testing.assert_allclose(strategy._c["w"], np.mean(list(clients.control.values()), axis=0), atol=1e-5)


def test_scaffold_removes_the_client_sampling_noise_floor_that_fedavg_keeps():
    """Clients fully fit their own target (large lr * steps) and only half participate each round.

    FedAvg then jumps to the mean of whichever clients were sampled and never settles;
    SCAFFOLD's control variates cancel the drift, so it converges to the true optimum.
    """
    kwargs = dict(lr=0.3, steps=30)
    part = dict(fraction_train=0.5, fraction_evaluate=0.5)
    fedavg_err = np.mean([distance(run(FedAvg(seed=s, **part), rounds=150, **kwargs)[0]) for s in range(3)])
    scaffold_err = np.mean([distance(run(Scaffold(seed=s, **part), rounds=150, **kwargs)[0]) for s in range(3)])
    assert fedavg_err > 0.1
    assert scaffold_err < 0.2 * fedavg_err


def test_scaffold_rejects_dp():
    with pytest.raises(ValueError):
        Scaffold(dp=FixedClipping(1.0, 1.0))


# ------------------------------------------------------------------------ server optimisers
def test_fedavgm_matches_torch_sgd_momentum_convention():
    clients = QuadraticClients(TARGETS, lr=0.5, steps=1)
    strategy = FedAvgM(momentum=0.9, server_learning_rate=1.0, **COMMON)
    # reference: replay FedAvg deltas by hand with torch.optim.SGD's momentum convention
    x = np.zeros(DIM)
    v = None
    for _ in range(3):
        # client models after one lr=0.5 step from x
        models = [x + 0.5 * (TARGETS[n] - x) for n in TARGETS]
        delta = np.mean(models, axis=0) - x
        grad = -delta
        v = grad if v is None else 0.9 * v + grad
        x = x - 1.0 * v
    w, _, _ = run_strategy(strategy, clients, N, rounds=3, dim=DIM)
    np.testing.assert_allclose(w, x, atol=1e-5)


FEDOPT_HPARAMS = dict(beta_1=0.9, beta_2=0.9, tau=0.05, server_learning_rate=0.2)


def reference_fedopt(kind, rounds):
    """Reddi et al. (2021), Algorithm 2, replayed with plain numpy on the quadratic clients.

    Written independently of the library code: v starts at tau^2, no bias correction.
    """
    b1, b2, tau, eta = (FEDOPT_HPARAMS[k] for k in ("beta_1", "beta_2", "tau", "server_learning_rate"))
    x, m, v = np.zeros(DIM), np.zeros(DIM), np.full(DIM, tau**2)
    for _ in range(rounds):
        models = [x + 0.5 * (TARGETS[n] - x) for n in TARGETS]  # one local step, lr 0.5
        delta = np.mean(models, axis=0) - x
        m = b1 * m + (1 - b1) * delta
        if kind == "adagrad":
            v = v + delta**2
        elif kind == "adam":
            v = b2 * v + (1 - b2) * delta**2
        else:
            v = v - (1 - b2) * delta**2 * np.sign(v - delta**2)
        x = x + eta * m / (np.sqrt(v) + tau)
    return x


@pytest.mark.parametrize("cls,kind", [(FedAdagrad, "adagrad"), (FedAdam, "adam"), (FedYogi, "yogi")])
def test_fedopt_matches_the_paper_over_several_rounds(cls, kind):
    clients = QuadraticClients(TARGETS, lr=0.5, steps=1)
    w, _, _ = run_strategy(cls(**FEDOPT_HPARAMS, **COMMON), clients, N, rounds=6, dim=DIM)
    expected = reference_fedopt(kind, 6)
    np.testing.assert_allclose(w, expected, rtol=1e-4, atol=1e-6)


def test_the_fedopt_references_are_distinguishable():
    """Guard: the replay above only proves something if the three rules give different results."""
    ref = {k: reference_fedopt(k, 6) for k in ("adagrad", "adam", "yogi")}
    for a, b in [("adagrad", "adam"), ("adam", "yogi"), ("adagrad", "yogi")]:
        assert np.linalg.norm(ref[a] - ref[b]) > 1e-3, (a, b)


@pytest.mark.parametrize(
    "strategy",
    [
        FedAvgM(momentum=0.5, **COMMON),
        FedAdam(server_learning_rate=0.05, **COMMON),
        FedYogi(server_learning_rate=0.05, **COMMON),
        FedAdagrad(server_learning_rate=0.1, **COMMON),
    ],
    ids=lambda s: type(s).__name__,
)
def test_server_optimisers_make_progress_towards_the_optimum(strategy):
    start = distance(np.zeros(DIM))
    w, _, _ = run(strategy, rounds=100)
    assert distance(w) < 0.2 * start


# ------------------------------------------------------------------------ robust aggregation
@pytest.mark.parametrize(
    "strategy",
    [
        FedMedian(**COMMON),
        FedTrimmedAvg(beta=0.25, **COMMON),
        Krum(num_byzantine=2, **COMMON),
        MultiKrum(num_byzantine=2, num_select=4, **COMMON),
    ],
    ids=lambda s: type(s).__name__,
)
def test_robust_rules_survive_byzantine_clients_that_break_fedavg(strategy):
    bad = (1, 2)
    honest_opt = np.mean([TARGETS[n] for n in TARGETS if n not in bad], axis=0)
    w, _, _ = run(strategy, rounds=40, bad=bad)
    fedavg_w, _, _ = run(FedAvg(**COMMON), rounds=40, bad=bad)
    assert np.linalg.norm(fedavg_w - honest_opt) > 100  # 2/8 clients sending 1e3 ruin FedAvg
    # median-like rules are biased by the spread of honest targets, but stay in range
    assert np.linalg.norm(w - honest_opt) < 1.5


def test_krum_selects_exactly_one_honest_clients_model():
    bad = (1, 2)
    w, _, _ = run(Krum(num_byzantine=2, **COMMON), rounds=1, bad=bad)
    # one round of 5 GD steps (lr 0.1) from w=0 gives honest client n the model t_n * (1 - 0.9**5)
    honest = [TARGETS[n] * (1 - 0.9**5) for n in TARGETS if n not in bad]
    assert any(np.allclose(w, h, atol=1e-5) for h in honest)


@pytest.mark.parametrize("cls", [FedMedian, FedTrimmedAvg, Krum, MultiKrum])
def test_robust_rules_reject_dp(cls):
    with pytest.raises(ValueError):
        cls(dp=FixedClipping(1.0, 1.0))


def test_trimmed_avg_warns_when_it_trims_nothing(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flwr"):
        run(FedTrimmedAvg(beta=0.05, **COMMON), rounds=1)  # int(0.05 * 8) == 0
    assert "trims nothing" in caplog.text


def test_bulyan_survives_byzantine_clients_and_selects_theta_updates():
    n = 11  # 4f + 3 with f = 2
    rng = np.random.default_rng(1)
    targets = {i: rng.normal(0, 0.3, DIM) for i in range(1, n + 1)}
    bad = (1, 2)
    honest_mean = np.mean([targets[i] for i in targets if i not in bad], axis=0)
    common = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)

    def final(strategy):
        w, result, _ = run_strategy(strategy, QuadraticClients(targets, bad=bad), n, 30, DIM)
        return w, result

    w_avg, _ = final(FedAvg(**common))
    w_bulyan, result = final(Bulyan(num_byzantine=2, **common))
    assert np.linalg.norm(w_avg - honest_mean) > 100
    assert np.linalg.norm(w_bulyan - honest_mean) < 0.5
    assert np.all(np.isfinite(w_bulyan))


def test_bulyan_warns_and_degrades_with_too_few_replies(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flwr"):
        w, _, _ = run(Bulyan(num_byzantine=2, **COMMON), rounds=2)  # 8 replies < 4*2+3
    assert "robustness guarantee does not hold" in caplog.text and np.all(np.isfinite(w))


def test_bulyan_rejects_dp():
    with pytest.raises(ValueError):
        Bulyan(dp=FixedClipping(1.0, 1.0))


# ------------------------------------------------------------------------- reproducibility
def partitions_sampled(seed, grid_seed, rounds=4, n=10):
    """Registration positions (== partition ids in the simulation) picked in each round."""
    from flwr.app import ConfigRecord as Cfg
    from conftest import RegistrationOrderGrid
    from defai.records import to_record as rec

    targets = {i: np.zeros(DIM) for i in range(n)}
    grid = RegistrationOrderGrid(n, lambda m: None, grid_seed)
    position = {node: i for i, node in enumerate(grid.nodes)}
    strategy = FedAvg(fraction_train=0.5, fraction_evaluate=0.5, seed=seed)
    picked = []
    for rnd in range(1, rounds + 1):
        messages = strategy.configure_train(rnd, rec({"w": np.zeros(DIM, np.float32)}), Cfg({"lr": 0.1}), grid)
        picked.append(sorted(position[m.metadata.dst_node_id] for m in messages))
    assert targets  # keep the helper's signature honest
    return picked


def test_client_sampling_picks_the_same_partitions_whatever_the_random_node_ids():
    first = partitions_sampled(seed=3, grid_seed=1)
    assert first == partitions_sampled(seed=3, grid_seed=2)  # different random node ids, same partitions
    assert first != partitions_sampled(seed=4, grid_seed=1)  # a different seed samples differently
    assert all(len(p) == 5 for p in first)


def test_aggregation_order_does_not_depend_on_reply_arrival_order():
    from flwr.app import ConfigRecord as Cfg
    from conftest import RegistrationOrderGrid
    from defai.records import to_record as rec

    seen = []

    class Spy(FedAvg):
        def aggregate_updates(self, updates):
            seen.append([u.node_id for u in updates])
            return super().aggregate_updates(updates)

    n = 8
    grid = RegistrationOrderGrid(n, lambda m: None, 5)
    targets = dict(zip(grid.nodes, [np.full(DIM, float(i)) for i in range(n)]))
    clients = QuadraticClients(targets, lr=0.5, steps=1, num_examples={node: 10 for node in grid.nodes})
    strategy = Spy(fraction_train=1.0, fraction_evaluate=1.0, seed=0)
    messages = list(strategy.configure_train(1, rec({"w": np.zeros(DIM, np.float32)}), Cfg({"lr": 0.5}), grid))
    replies = [clients(m) for m in messages]
    rng = np.random.default_rng(0)
    for _ in range(3):
        rng.shuffle(replies)
        strategy.aggregate_train(1, list(replies))
    assert seen[0] == grid.nodes  # registration order, not id order and not arrival order
    assert all(order == seen[0] for order in seen)


# --------------------------------------------------------------------- malformed replies
class _Corrupt(QuadraticClients):
    """Node 1 sends the given value in place of its model (and node 2 an infinite weight if asked)."""

    def __init__(self, targets, value=np.nan, weight=None, **kw):
        super().__init__(targets, **kw)
        self.value, self.weight = value, weight

    def __call__(self, msg):
        reply = super().__call__(msg)
        node = msg.metadata.dst_node_id
        if "arrays" not in reply.content:
            return reply
        content = dict(reply.content)
        if node == 1 and self.weight is None:
            content["arrays"] = to_record({"w": np.full(DIM, self.value, dtype=np.float32)})
        if node == 1 and self.weight is not None:
            content["metrics"] = MetricRecord({**dict(reply.content["metrics"]), "num-examples": self.weight})
        return Message(content=RecordDict(content), reply_to=msg)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("make", [
    lambda: FedAvg(**COMMON),
    lambda: FedAvg(dp=FixedClipping(1.0, 1.0, seed=0), **COMMON),
    lambda: FedMedian(**COMMON),
], ids=["fedavg", "fedavg+dp", "fedmedian"])
def test_replies_with_non_finite_parameters_are_dropped(make, value, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flwr"):
        w, _, _ = run_strategy(make(), _Corrupt(TARGETS, value=value), N, 3, DIM)
    assert np.all(np.isfinite(w))
    assert "non-finite values" in caplog.text


@pytest.mark.parametrize("weight", [float("inf"), float("nan"), 0, -3])
def test_replies_with_unusable_weights_are_dropped(weight, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flwr"):
        w, _, _ = run_strategy(FedAvg(**COMMON), _Corrupt(TARGETS, weight=weight), N, 2, DIM)
    assert np.all(np.isfinite(w)) and "positive finite number" in caplog.text


def test_integer_parameters_do_not_drift_by_one_when_all_clients_agree():
    """Regression: float64 round-off (257.99999...) used to be truncated to 257 in the global model."""
    from flwr.app import ConfigRecord

    rng = np.random.default_rng(0)
    strategy = FedAvg(**COMMON)
    start = {"w": np.zeros(DIM, np.float32), "steps": np.array([979], dtype=np.int64)}
    grid = FakeGrid(6, lambda m: None)
    for trial in range(60):  # different sample orders / weights hit different round-off
        messages = list(strategy.configure_train(1, to_record(start), ConfigRecord({"lr": 0.1}), grid))
        weights = rng.integers(1, 500, size=len(messages))
        replies = [
            Message(
                content=RecordDict({
                    "arrays": to_record({"w": np.ones(DIM, np.float32), "steps": np.array([258], dtype=np.int64)}),
                    "metrics": MetricRecord({"num-examples": int(w)}),
                }),
                reply_to=m,
            )
            for m, w in zip(messages, weights)
        ]
        arrays, _ = strategy.aggregate_train(1, replies)
        steps = arrays["steps"].numpy()
        assert steps.dtype == np.int64 and steps[0] == 258, (trial, steps)


# ------------------------------------------------------------------------------------ FedNova
def two_clients(steps, lr=0.01):
    """Client 1 wants +1, client 2 wants -1 (true optimum 0); they take ``steps`` local steps."""
    targets = {1: np.array([1.0]), 2: np.array([-1.0])}
    return QuadraticClients(targets, lr=lr, steps=steps)


def fixed_point(strategy, steps, rounds=300):
    w, _, _ = run_strategy(strategy, two_clients(steps), 2, rounds, dim=1)
    return float(w[0])


def test_fednova_equals_fedavg_when_every_client_takes_the_same_number_of_steps():
    common = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)
    nova, _, _ = run(FedNova(**common), rounds=5)
    avg, _, _ = run(FedAvg(**common), rounds=5)
    np.testing.assert_allclose(nova, avg, atol=1e-6)


def test_fednova_matches_the_formula_for_unequal_steps_and_weights():
    steps = {n: 2 * n for n in TARGETS}  # 2, 4, ..., 16
    sizes = {n: 10 * n for n in TARGETS}
    clients = QuadraticClients(TARGETS, lr=0.05, steps=steps, num_examples=sizes)
    w, _, _ = run_strategy(FedNova(**COMMON), clients, N, rounds=1, dim=DIM)
    p = np.array([sizes[n] for n in TARGETS], float)
    p /= p.sum()
    tau = np.array([steps[n] for n in TARGETS], float)
    deltas = np.stack([TARGETS[n] * (1 - 0.95 ** steps[n]) for n in TARGETS])  # local GD from w=0 on 0.5*||w-t||^2
    expected = (p @ tau) * np.einsum("i,id->d", p / tau, deltas)
    np.testing.assert_allclose(w, expected, rtol=1e-4, atol=1e-6)


def test_fednova_removes_the_bias_that_unequal_local_steps_put_into_fedavg():
    common = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)
    steps = {1: 1, 2: 30}  # the client that wants -1 does 30 times more work
    biased = fixed_point(FedAvg(**common), steps)
    corrected = fixed_point(FedNova(**common), steps)
    assert biased < -0.8      # FedAvg settles next to the busy client's optimum instead of the true one (0)
    assert abs(corrected) < 0.15


def test_fednova_rejects_dp_and_needs_the_step_count():
    with pytest.raises(ValueError):
        FedNova(dp=FixedClipping(1.0, 1.0))
    assert FedNova.required_metrics == ("local-steps",)


# ----------------------------------------------------------------------------------- q-FedAvg
def client_losses(targets, w):
    return np.array([0.5 * np.sum((w - t) ** 2) for t in targets.values()])


def test_qfedavg_with_q_zero_is_the_uniform_average():
    common = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)
    q0, _, _ = run(QFedAvg(q=0.0, **common), rounds=6)
    avg, _, _ = run(FedAvg(**common), rounds=6)  # QuadraticClients has equal data sizes
    np.testing.assert_allclose(q0, avg, atol=1e-6)


def test_qfedavg_one_round_matches_the_paper_formula():
    q, lr, steps = 1.0, 0.1, 5
    clients = QuadraticClients(TARGETS, lr=lr, steps=steps)
    w, _, _ = run_strategy(QFedAvg(q=q, **COMMON), clients, N, rounds=1, dim=DIM)
    # from w = 0: client n's loss is F_n = 0.5*||t_n||^2 and its local model moves by delta_n = t_n * (1 - (1-lr)^steps)
    L = 1 / lr
    F = np.array([0.5 * np.sum(TARGETS[n] ** 2) for n in TARGETS]) + 1e-10
    delta = np.stack([TARGETS[n] * (1 - (1 - lr) ** steps) for n in TARGETS])
    h = np.sum(q * F ** (q - 1) * L**2 * np.sum(delta**2, axis=1) + L * F**q)
    expected = L * np.einsum("i,id->d", F**q, delta) / h
    np.testing.assert_allclose(w, expected, rtol=1e-4)


def qfedavg_expected(q, lipschitz, lr, steps):
    """Paper formula for one round from w = 0: (update, step scale)."""
    F = np.array([0.5 * np.sum(TARGETS[n] ** 2) for n in TARGETS]) + 1e-10
    delta = np.stack([TARGETS[n] * (1 - (1 - lr) ** steps) for n in TARGETS])
    h = q * F ** (q - 1) * lipschitz**2 * np.sum(delta**2, axis=1) + lipschitz * F**q
    return lipschitz * np.einsum("i,id->d", F**q, delta) / h.sum(), np.sum(lipschitz * F**q) / h.sum()


def test_qfedavg_lipschitz_constant_replaces_one_over_lr_and_moves_the_step_size():
    q, lr, steps = 1.0, 0.1, 5
    results = {}
    for lipschitz in (0.0, 10.0, 2.0):  # 0 means the default 1 / lr = 10
        clients = QuadraticClients(TARGETS, lr=lr, steps=steps)
        w, _, _ = run_strategy(QFedAvg(q=q, lipschitz=lipschitz, **COMMON), clients, N, rounds=1, dim=DIM)
        expected, _ = qfedavg_expected(q, lipschitz or 1 / lr, lr, steps)
        np.testing.assert_allclose(w, expected, rtol=1e-4)
        results[lipschitz] = w
    np.testing.assert_allclose(results[0.0], results[10.0])
    # a smaller L divides by less, so the step is larger
    assert np.linalg.norm(results[2.0]) > 2 * np.linalg.norm(results[10.0])


def test_qfedavg_reports_how_much_its_normalisation_shrinks_the_step():
    q, lr, steps = 1.0, 0.1, 5
    for q_value in (q, 0.0):
        clients = QuadraticClients(TARGETS, lr=lr, steps=steps)
        _, result, _ = run_strategy(QFedAvg(q=q_value, **COMMON), clients, N, rounds=1, dim=DIM)
        scale = result.train_metrics_clientapp[1]["qfedavg-step-scale"]
        assert scale == pytest.approx(qfedavg_expected(q_value, 1 / lr, lr, steps)[1], rel=1e-6)
    assert scale == pytest.approx(1.0)  # q = 0: nothing shrinks the uniform average
    _, result, _ = run_strategy(FedAvg(**COMMON), QuadraticClients(TARGETS, lr=lr, steps=steps), N, rounds=1, dim=DIM)
    assert "qfedavg-step-scale" not in result.train_metrics_clientapp[1]  # only q-FedAvg reports it


def test_extra_round_metrics_of_a_strategy_are_reported_for_that_round_only():
    class Once(FedAvg):
        calls = 0

        def aggregate_updates(self, updates):
            self.calls += 1
            if self.calls == 1:
                self._round_metrics["only-in-round-one"] = 3.0
            return super().aggregate_updates(updates)

    _, result, _ = run(Once(**COMMON), rounds=2)
    assert result.train_metrics_clientapp[1]["only-in-round-one"] == 3.0
    assert "only-in-round-one" not in result.train_metrics_clientapp[2]


def test_qfedavg_trades_mean_loss_for_a_better_worst_client():
    targets = {1: np.array([-1.0]), 2: np.array([-1.0]), 3: np.array([-1.0]), 4: np.array([4.0])}  # one outlier client
    common = dict(fraction_train=1.0, fraction_evaluate=1.0, seed=0)

    def final_losses(q):
        w, _, _ = run_strategy(QFedAvg(q=q, **common), QuadraticClients(targets, lr=0.1, steps=5), 4, 300, dim=1)
        return client_losses(targets, w)

    fair, plain = final_losses(2.0), final_losses(0.0)
    assert fair.max() < 0.6 * plain.max()   # the outlier is served much better ...
    assert fair.mean() > plain.mean()       # ... at the price of the average (that is the trade-off)


def test_qfedavg_asks_for_the_initial_loss_with_the_decayed_learning_rate():
    strategy = QFedAvg(lr_decay=0.5, **COMMON)
    _, _, grid = run(strategy, rounds=3)
    train = [m for m in grid.sent if m.metadata.message_type == "train"]
    assert all(m.content["config"]["report-initial-loss"] is True for m in train)
    assert strategy._client_lr == pytest.approx(0.1 * 0.5**2)  # round 3 -> L = 1 / (lr * decay^2)


def test_qfedavg_validation_and_dp_exclusion():
    with pytest.raises(ValueError):
        QFedAvg(q=-0.1)
    with pytest.raises(ValueError, match="lipschitz"):
        QFedAvg(lipschitz=-1.0)
    with pytest.raises(ValueError):
        QFedAvg(dp=FixedClipping(1.0, 1.0))
    assert QFedAvg.required_metrics == ("initial-loss",)
