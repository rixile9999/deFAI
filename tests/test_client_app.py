"""client_app.train/evaluate driven directly with a real Context, real model and synthetic data."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict

from defai import client_app
from defai.aggregation import l2_norm
from defai.records import (
    KEY_C,
    KEY_C_DELTA,
    STATE_MOON_PREVIOUS,
    STATE_SCAFFOLD_C,
    to_arrays,
    to_record,
)
from defai.task import build_model, load_client_data

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
NUM_PARTITIONS = 4
LR = 0.05


def config(**overrides):
    cfg = tomllib.loads(PYPROJECT.read_text())["tool"]["flwr"]["app"]["config"]
    return {**cfg, "dataset": "synthetic", "batch-size": 64, **overrides}


def global_model_record():
    torch.manual_seed(123)
    return ArrayRecord(build_model("synthetic").state_dict())


GLOBAL = global_model_record()


def train_message(round_config=None, **records):
    content = {"arrays": GLOBAL, "config": ConfigRecord({"lr": LR, "server-round": 1, **(round_config or {})}), **records}
    return Message(content=RecordDict(content), message_type=MessageType.TRAIN, dst_node_id=1)


def context(cfg, partition_id=0, state=None):
    return Context(
        run_id=1,
        node_id=partition_id + 1,
        node_config={"partition-id": partition_id, "num-partitions": NUM_PARTITIONS},
        state=state if state is not None else RecordDict(),
        run_config=cfg,
    )


def run_train(cfg, round_config=None, partition_id=0, ctx=None, **records):
    torch.manual_seed(0)  # DataLoader shuffling
    ctx = ctx or context(cfg, partition_id)
    reply = client_app.train(train_message(round_config, **records), ctx)
    return reply, ctx


def update_of(reply):
    glob = to_arrays(GLOBAL)
    local = to_arrays(reply.content["arrays"])
    return {k: local[k] - glob[k] for k in glob}


def test_reply_structure_and_metrics():
    reply, _ = run_train(config())
    assert list(reply.content["arrays"].keys()) == list(GLOBAL.keys())
    metrics = reply.content["metrics"]
    assert metrics["num-examples"] > 0 and np.isfinite(metrics["train_loss"])
    assert set(metrics) == {"train_loss", "num-examples", "local-steps"}  # nothing algorithm-specific leaks in
    assert l2_norm(update_of(reply)) > 0


def test_local_training_is_reproducible_whatever_the_global_torch_state():
    """Data order is seeded from (seed, partition, round), so the worker's RNG state is irrelevant."""
    def update(torch_seed, **kw):
        torch.manual_seed(torch_seed)
        ctx = context(config(**kw.pop("cfg", {})), kw.pop("partition", 0))
        msg = train_message({"server-round": kw.pop("round", 1)})
        return update_of(client_app.train(msg, ctx))

    a, b = update(0), update(12345)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    other_round, other_partition, other_seed = update(0, round=2), update(0, partition=1), update(0, cfg={"seed": 7})
    for other in (other_round, other_partition, other_seed):
        assert any(not np.allclose(a[k], other[k]) for k in a)


def test_fedprox_term_shrinks_the_local_update():
    free = l2_norm(update_of(run_train(config())[0]))
    prox = l2_norm(update_of(run_train(config(), {"proximal-mu": 2.0})[0]))
    assert prox < free


# --------------------------------------------------------------------------- SCAFFOLD
def zero_control_variate():
    return to_record({k: np.zeros_like(v) for k, v in to_arrays(GLOBAL).items()})


def test_scaffold_first_round_returns_control_variate_from_the_formula_and_stores_it():
    cfg = config()
    reply, ctx = run_train(cfg, **{KEY_C: zero_control_variate()})
    trainloader, _ = load_client_data(0, NUM_PARTITIONS, cfg)
    steps = len(trainloader) * cfg["local-epochs"]
    # c = c_i = 0  ->  c_i^+ = (x - y) / (K * lr), and dc = c_i^+
    expected = {k: -v / (steps * LR) for k, v in update_of(reply).items()}
    delta_c = to_arrays(reply.content[KEY_C_DELTA])
    stored = to_arrays(ctx.state[STATE_SCAFFOLD_C])
    assert list(delta_c) == [n for n, _ in build_model("synthetic").named_parameters()]
    for k in delta_c:
        np.testing.assert_allclose(delta_c[k], expected[k], rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(stored[k], delta_c[k])
    assert reply.content["metrics"]["scaffold-c-norm"] == pytest.approx(l2_norm(stored))


def test_scaffold_state_persists_and_changes_the_next_round():
    cfg = config()
    first, ctx = run_train(cfg, **{KEY_C: zero_control_variate()})
    after_first = to_arrays(ctx.state[STATE_SCAFFOLD_C])
    again, _ = run_train(cfg, ctx=ctx, **{KEY_C: zero_control_variate()})  # same context -> c_i is remembered
    fresh, _ = run_train(cfg, **{KEY_C: zero_control_variate()})           # new client, no state
    first_u, again_u, fresh_u = (update_of(r) for r in (first, again, fresh))
    for k in first_u:  # a client without state reproduces round 1 exactly
        np.testing.assert_array_equal(first_u[k], fresh_u[k])
    assert any(not np.allclose(first_u[k], again_u[k]) for k in first_u)  # the remembered c_i corrected the gradients
    after_second = to_arrays(ctx.state[STATE_SCAFFOLD_C])
    assert any(not np.allclose(after_first[k], after_second[k]) for k in after_first)  # and was updated again


# ------------------------------------------------------------------------------- MOON
MOON = {"moon-mu": 5.0, "moon-temperature": 0.5}


def test_moon_first_participation_equals_plain_training_then_state_is_kept():
    cfg = config()
    plain = update_of(run_train(cfg)[0])
    first, ctx = run_train(cfg, MOON)
    for k, v in update_of(first).items():
        np.testing.assert_allclose(v, plain[k], atol=2e-4)
    assert STATE_MOON_PREVIOUS in ctx.state
    saved = to_arrays(ctx.state[STATE_MOON_PREVIOUS])
    for k, v in to_arrays(first.content["arrays"]).items():
        np.testing.assert_allclose(saved[k], v, atol=1e-6)  # the previous local model is this round's result

    second, _ = run_train(cfg, MOON, ctx=ctx)  # now z_prev != z_global
    assert any(not np.allclose(v, plain[k], atol=1e-3) for k, v in update_of(second).items())


# ----------------------------------------------------------------------------- attacks
def attacked(attack, partition_id=0, **extra):
    cfg = config(attack=attack, **{"num-malicious": 1, **extra})
    return run_train(cfg, partition_id=partition_id)[0]


def test_sign_flip_negates_the_honest_update():
    honest = update_of(run_train(config())[0])
    flipped = update_of(attacked("sign-flip", **{"attack-scale": 2.0}))
    for k in honest:
        np.testing.assert_allclose(flipped[k], -2.0 * honest[k], atol=1e-6)


def test_gaussian_attack_sends_noise_of_the_requested_scale():
    update = update_of(attacked("gaussian", **{"attack-scale": 0.5}))
    flat = np.concatenate([v.ravel() for v in update.values()])
    assert flat.std() == pytest.approx(0.5, rel=0.02) and abs(flat.mean()) < 0.01


def test_label_flip_changes_what_is_learned():
    honest = update_of(run_train(config())[0])
    flipped = update_of(attacked("label-flip"))
    assert any(not np.allclose(honest[k], flipped[k], atol=1e-4) for k in honest)


@pytest.mark.parametrize("attack", ["label-flip", "sign-flip", "gaussian"])
def test_only_the_first_num_malicious_partitions_misbehave(attack):
    honest = update_of(run_train(config(), partition_id=1)[0])
    same = update_of(attacked(attack, partition_id=1))  # partition 1 >= num-malicious (1)
    for k in honest:
        np.testing.assert_array_equal(honest[k], same[k])


def test_num_malicious_zero_disables_the_attack():
    honest = update_of(run_train(config())[0])
    same = update_of(run_train(config(attack="sign-flip", **{"num-malicious": 0}))[0])
    for k in honest:
        np.testing.assert_array_equal(honest[k], same[k])


# ------------------------------------------------------------------------ client-side DP
def test_client_side_clipping_bounds_the_update_and_reports_the_bit():
    clipped, _ = run_train(config(), {"dp-clip-norm": 0.01, "dp-adaptive": True})
    assert l2_norm(update_of(clipped)) == pytest.approx(0.01, rel=1e-3)
    assert clipped.content["metrics"]["dp-unclipped-bit"] == 0

    loose, _ = run_train(config(), {"dp-clip-norm": 1e6, "dp-adaptive": True})
    unclipped = update_of(run_train(config())[0])
    assert loose.content["metrics"]["dp-unclipped-bit"] == 1
    for k, v in update_of(loose).items():
        np.testing.assert_allclose(v, unclipped[k], atol=1e-6)


def test_fixed_clipping_does_not_report_a_bit():
    reply, _ = run_train(config(), {"dp-clip-norm": 0.01, "dp-adaptive": False})
    assert "dp-unclipped-bit" not in reply.content["metrics"]


def test_attack_is_applied_before_clipping_so_dp_bounds_it():
    cfg = config(attack="sign-flip", **{"num-malicious": 1, "attack-scale": 100.0})
    reply, _ = run_train(cfg, {"dp-clip-norm": 0.5, "dp-adaptive": False})
    assert l2_norm(update_of(reply)) <= 0.5 * (1 + 1e-4)


# -------------------------------------------------------------------------------- evaluate
def test_evaluate_returns_loss_and_accuracy():
    msg = Message(
        content=RecordDict({"arrays": GLOBAL, "config": ConfigRecord({"server-round": 1})}),
        message_type=MessageType.EVALUATE,
        dst_node_id=1,
    )
    reply = client_app.evaluate(msg, context(config()))
    metrics = reply.content["metrics"]
    assert 0.0 <= metrics["eval_acc"] <= 1.0 and metrics["eval_loss"] > 0 and metrics["num-examples"] > 0


# ------------------------------------------------------------------------------- client dropout
def test_a_dropped_out_client_answers_with_an_error_and_changes_no_state():
    from flwr.common.constant import ErrorCode

    cfg = config(**{"dropout-rate": 1.0})
    reply, ctx = run_train(cfg, **{KEY_C: zero_control_variate()})
    assert reply.has_error() and reply.error.code == ErrorCode.NODE_UNAVAILABLE
    assert "dropout" in reply.error.reason
    assert STATE_SCAFFOLD_C not in ctx.state  # it never trained, so its control variate is untouched


def test_without_dropout_the_client_always_answers():
    reply, _ = run_train(config(**{"dropout-rate": 0.0}))
    assert not reply.has_error()


def test_dropout_follows_the_rate_and_is_fixed_by_seed_partition_and_round():
    cfg = config(**{"dropout-rate": 0.3})
    decisions = [client_app.dropped_out(cfg, p, r) for p in range(40) for r in range(1, 51)]
    assert sum(decisions) / len(decisions) == pytest.approx(0.3, abs=0.04)
    again = [client_app.dropped_out(cfg, p, r) for p in range(40) for r in range(1, 51)]
    assert decisions == again  # reproducible
    other = [client_app.dropped_out({**cfg, "seed": cfg["seed"] + 1}, p, r) for p in range(40) for r in range(1, 51)]
    assert decisions != other  # but not the same for another seed
    assert not any(client_app.dropped_out({**cfg, "dropout-rate": 0.0}, p, r) for p in range(40) for r in range(1, 51))
    assert all(client_app.dropped_out({**cfg, "dropout-rate": 1.0}, p, r) for p in range(5) for r in range(1, 6))


def test_the_reported_local_steps_are_the_batches_times_the_epochs_actually_run():
    cfg = config(**{"local-epochs": 3})
    trainloader, _ = load_client_data(0, NUM_PARTITIONS, cfg)
    reply, _ = run_train(cfg)
    assert reply.content["metrics"]["local-steps"] == 3 * len(trainloader)
    assert reply.content["metrics"]["local-steps"] > run_train(config())[0].content["metrics"]["local-steps"]


# ------------------------------------------------------------------------------- initial loss
def test_the_initial_loss_is_the_received_models_loss_on_the_local_training_data():
    from defai.task import get_device
    from defai.task import test as test_fn

    cfg = config()
    reply, _ = run_train(cfg, {"report-initial-loss": True})
    model = build_model("synthetic")
    model.load_state_dict(GLOBAL.to_torch_state_dict())
    trainloader, _ = load_client_data(0, NUM_PARTITIONS, cfg)
    expected, _ = test_fn(model, torch.utils.data.DataLoader(trainloader.dataset, batch_size=cfg["batch-size"]), get_device())
    assert reply.content["metrics"]["initial-loss"] == pytest.approx(expected, rel=1e-5)
    # it is the loss *before* training: higher than the mean loss seen while training
    assert reply.content["metrics"]["initial-loss"] > reply.content["metrics"]["train_loss"]


def test_the_initial_loss_is_only_measured_when_asked_for():
    reply, _ = run_train(config())
    assert "initial-loss" not in reply.content["metrics"]


# ---------------------------------------------------------------------------------- compression
from defai.compression import topk  # noqa: E402
from defai.records import STATE_EF_RESIDUAL  # noqa: E402


def flat(update):
    return np.concatenate([v.ravel() for v in update.values()])


def test_without_compression_nothing_about_it_is_reported():
    metrics = run_train(config())[0].content["metrics"]
    assert "upload-bytes" not in metrics and "compression-error" not in metrics


def test_topk_sends_a_sparse_update_made_of_the_largest_entries_and_reports_the_real_cost():
    honest = update_of(run_train(config())[0])
    reply, ctx = run_train(config(compression="topk", **{"compression-ratio": 0.05}))
    sent = update_of(reply)
    n = flat(honest).size
    assert np.count_nonzero(flat(sent)) <= int(np.ceil(0.05 * n))
    expected, nbytes = topk(honest, 0.05)  # the same training, then top-k of its update
    for k in honest:
        np.testing.assert_allclose(sent[k], expected[k], atol=1e-6)
    metrics = reply.content["metrics"]
    assert metrics["upload-bytes"] == nbytes == 8 * int(np.ceil(0.05 * n))
    assert 0.0 < metrics["compression-error"] < 1.0
    assert STATE_EF_RESIDUAL not in ctx.state  # error feedback is opt-in


def test_quantization_sends_a_dense_update_that_is_close_and_cheaper():
    honest = update_of(run_train(config())[0])
    n = flat(honest).size
    errors, costs = {}, {}
    for bits in (4, 8):
        reply, _ = run_train(config(compression="quantize", **{"compression-bits": bits}))
        sent = update_of(reply)
        assert any(not np.array_equal(sent[k], honest[k]) for k in honest)  # lossy, and dense (not sparse)
        errors[bits] = reply.content["metrics"]["compression-error"]
        costs[bits] = reply.content["metrics"]["upload-bytes"]
    # update tensors are heavy-tailed (a few large entries set the per-tensor scale), so 8 bits are close
    # but not lossless; fewer bits cost proportionally less and are clearly worse
    assert errors[8] < 0.06 and errors[4] > 4 * errors[8]
    assert costs[8] == pytest.approx(0.25 * 4 * n, rel=0.01) and costs[4] == pytest.approx(0.125 * 4 * n, rel=0.01)


def test_error_feedback_keeps_a_residual_in_the_client_state_and_uses_it_next_round():
    cfg = config(compression="topk", **{"compression-ratio": 0.05, "error-feedback": True})
    first, ctx = run_train(cfg)
    residual = to_arrays(ctx.state[STATE_EF_RESIDUAL])
    honest = update_of(run_train(config())[0])
    # residual = (update) - (what was sent): the mass top-k dropped
    for k in honest:
        np.testing.assert_allclose(residual[k], honest[k] - update_of(first)[k], atol=1e-5)
    second, _ = run_train(cfg, ctx=ctx)                       # the same client, the remembered residual
    fresh, _ = run_train(cfg)                                 # a client without state
    assert any(not np.allclose(update_of(second)[k], update_of(fresh)[k]) for k in honest)


def test_compression_error_is_the_share_of_the_compressed_update_that_was_lost():
    # with error feedback the compressor sees update + carried-over residual, and the metric measures against that
    cfg = config(compression="topk", **{"compression-ratio": 0.05, "error-feedback": True})
    first, ctx = run_train(cfg)
    first_residual = to_arrays(ctx.state[STATE_EF_RESIDUAL])
    honest = update_of(run_train(config())[0])
    assert first.content["metrics"]["compression-error"] == pytest.approx(l2_norm(first_residual) / l2_norm(honest), rel=1e-4)
    second, _ = run_train(cfg, ctx=ctx)
    second_residual = to_arrays(ctx.state[STATE_EF_RESIDUAL])
    compressed = {k: honest[k] + first_residual[k] for k in honest}
    assert second.content["metrics"]["compression-error"] == pytest.approx(l2_norm(second_residual) / l2_norm(compressed), rel=1e-4)


def test_the_upload_cost_of_a_scaffold_client_includes_its_uncompressed_control_variate():
    plain, _ = run_train(config(compression="topk", **{"compression-ratio": 0.05}))
    scaffold, _ = run_train(config(compression="topk", **{"compression-ratio": 0.05}), **{KEY_C: zero_control_variate()})
    extra = scaffold.content["metrics"]["upload-bytes"] - plain.content["metrics"]["upload-bytes"]
    assert extra == sum(len(a.data) for a in scaffold.content[KEY_C_DELTA].values())


def test_a_dropped_client_does_not_touch_its_error_feedback_state():
    cfg = config(compression="topk", **{"compression-ratio": 0.05, "error-feedback": True, "dropout-rate": 1.0})
    reply, ctx = run_train(cfg)
    assert reply.has_error() and STATE_EF_RESIDUAL not in ctx.state


# ------------------------------------------------------------------------------ personalisation
def evaluate_message():
    return Message(
        content=RecordDict({"arrays": GLOBAL, "config": ConfigRecord({"server-round": 1})}),
        message_type=MessageType.EVALUATE,
        dst_node_id=1,
    )


def run_evaluate(cfg, partition_id=0):
    return client_app.evaluate(evaluate_message(), context(cfg, partition_id)).content["metrics"]


def test_personalized_metrics_appear_only_when_asked_for():
    assert "eval_acc_personalized" not in run_evaluate(config())
    metrics = run_evaluate(config(**{"personalize-epochs": 1}))
    assert {"eval_acc_personalized", "eval_loss_personalized"} <= set(metrics)


def test_fine_tuning_on_skewed_local_data_beats_the_global_model_and_leaves_it_unchanged():
    skewed = config(partitioner="dirichlet", **{"dirichlet-alpha": 0.05, "learning-rate": 0.05})
    plain = run_evaluate(skewed)
    tuned = run_evaluate({**skewed, "personalize-epochs": 2})
    # the untrained global model is near chance; two local epochs on a client's few classes help a lot
    assert tuned["eval_acc_personalized"] > tuned["eval_acc"] + 0.2
    assert tuned["eval_loss_personalized"] < tuned["eval_loss"]
    # evaluating the received model is unaffected by the personalisation step
    assert tuned["eval_acc"] == plain["eval_acc"] and tuned["eval_loss"] == pytest.approx(plain["eval_loss"])


def test_personalisation_is_reproducible():
    cfg = config(**{"personalize-epochs": 1})
    assert run_evaluate(cfg) == run_evaluate(cfg)
