"""ClientApp: runs on each SuperNode, trains/evaluates on its local partition.

The server decides which federated algorithm the client runs by what it puts into the
train message: ``proximal-mu`` (FedProx), a ``c`` control variate (SCAFFOLD),
``moon-mu`` (MOON), ``report-initial-loss`` (q-FedAvg) and ``dp-clip-norm`` (client-side
DP clipping). Algorithms that need memory between rounds keep it in ``context.state``.
The run config adds simulated misbehaviour (``attack``, ``dropout-rate``), update
compression (``compression``), applied last, just before the update is sent, and
personalisation metrics at evaluation time (``personalize-epochs``).
"""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from flwr.app import ArrayRecord, Context, Error, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.constant import ErrorCode

from defai.aggregation import l2_norm
from defai.attacks import is_malicious, poison_update
from defai.privacy.dp import KEY_ADAPTIVE, KEY_CLIP_NORM, KEY_UNCLIPPED_BIT, clip_to_norm
from defai.compression import compress, compress_with_feedback, relative_error
from defai.records import (
    KEY_C,
    KEY_C_DELTA,
    STATE_EF_RESIDUAL,
    STATE_MOON_PREVIOUS,
    STATE_SCAFFOLD_C,
    record_bytes,
    to_arrays,
    to_record,
)
from defai.task import NUM_CLASSES, MoonReference, build_model, get_device, load_client_data
from defai.task import scaffold_control_update
from defai.task import test as test_fn
from defai.task import train as train_fn

app = ClientApp()


def dropped_out(cfg, partition_id, server_round) -> bool:
    """Whether this client fails to answer this round (``dropout-rate``), fixed by (seed, partition, round)."""
    rate = cfg["dropout-rate"]
    return rate > 0 and np.random.default_rng([cfg["seed"], partition_id, server_round, 1]).random() < rate


def _frozen(model, device):
    model = copy.deepcopy(model).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _moon_reference(msg, context, model, device):
    """Global model (just received) vs. this client's previous local model."""
    config = msg.content["config"]
    global_model = _frozen(model, device)
    previous = global_model  # first participation: equal to the global model (no contrastive signal)
    if STATE_MOON_PREVIOUS in context.state:
        previous = build_model(context.run_config["dataset"])
        previous.load_state_dict(context.state[STATE_MOON_PREVIOUS].to_torch_state_dict())
        previous = _frozen(previous, device)
    return MoonReference(global_model, previous, config["moon-mu"], config["moon-temperature"])


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config
    config = msg.content["config"]
    partition_id = context.node_config["partition-id"]
    device = get_device()
    lr = config["lr"]
    if dropped_out(cfg, partition_id, config["server-round"]):
        # Decided before any work or state change: the client simply never answers this round
        return Message(Error(code=ErrorCode.NODE_UNAVAILABLE, reason="simulated client dropout"), reply_to=msg)
    # Reproducible local training (data order) for a given seed, whichever worker runs the client
    torch.manual_seed(int(np.random.SeedSequence([cfg["seed"], partition_id, config["server-round"]]).generate_state(1)[0]))

    model = build_model(cfg["dataset"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    global_arrays = to_arrays(msg.content["arrays"])
    param_names = [n for n, _ in model.named_parameters()]
    trainloader, _ = load_client_data(partition_id, context.node_config["num-partitions"], cfg)

    options, extra_metrics, extra_records = {}, {}, {}

    # SCAFFOLD: correct every gradient by (c - c_i)
    c_local = c_global = None
    if KEY_C in msg.content:
        c_global = to_arrays(msg.content[KEY_C])
        if STATE_SCAFFOLD_C in context.state:
            c_local = to_arrays(context.state[STATE_SCAFFOLD_C])
        else:
            c_local = {n: np.zeros_like(global_arrays[n]) for n in param_names}
        options["grad_correction"] = [torch.from_numpy(c_global[n] - c_local[n]) for n in param_names]

    # MOON: contrast against the global and previous local representation
    if "moon-mu" in config:
        options["moon"] = _moon_reference(msg, context, model, device)

    malicious = cfg["attack"] != "none" and is_malicious(partition_id, cfg["num-malicious"])
    if malicious and cfg["attack"] == "label-flip":
        options["flip_num_classes"] = NUM_CLASSES

    if config.get("report-initial-loss", False):
        # Loss of the *received* model on the local training data (q-FedAvg's F_k(w^t)), measured
        # before training; the loader is unshuffled so the training data order is not touched.
        loader = DataLoader(trainloader.dataset, batch_size=cfg["batch-size"])
        extra_metrics["initial-loss"] = max(test_fn(model, loader, device)[0], 1e-12)

    train_loss, steps = train_fn(
        model,
        trainloader,
        cfg["local-epochs"],
        lr,
        device,
        proximal_mu=config.get("proximal-mu", 0.0),  # set by the FedProx strategy
        **options,
    )
    local_arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    if c_local is not None:
        c_new = scaffold_control_update(
            c_local,
            {n: c_global[n] for n in param_names},
            {n: global_arrays[n] for n in param_names},
            {n: local_arrays[n] for n in param_names},
            steps,
            lr,
        )
        context.state[STATE_SCAFFOLD_C] = to_record(c_new)
        extra_records[KEY_C_DELTA] = to_record({n: c_new[n] - c_local[n] for n in param_names})
        extra_metrics["scaffold-c-norm"] = l2_norm(c_new)

    if "moon-mu" in config:
        context.state[STATE_MOON_PREVIOUS] = ArrayRecord(model.state_dict())

    if malicious and cfg["attack"] in ("sign-flip", "gaussian"):
        rng = np.random.default_rng([cfg["seed"], partition_id, config["server-round"]])
        local_arrays = poison_update(cfg["attack"], global_arrays, local_arrays, cfg["attack-scale"], rng)

    # Client-side DP: clip the update before it leaves the device
    if KEY_CLIP_NORM in config:
        update = {k: local_arrays[k] - global_arrays[k] for k in local_arrays}
        clipped, was_clipped = clip_to_norm(update, config[KEY_CLIP_NORM], tol=1e-6)
        local_arrays = {k: global_arrays[k] + clipped[k] for k in local_arrays}
        if config[KEY_ADAPTIVE]:
            extra_metrics[KEY_UNCLIPPED_BIT] = int(not was_clipped)

    # Compression is the last step before sending (the server still enforces any DP clipping bound)
    if cfg["compression"] != "none":
        update = {k: local_arrays[k] - global_arrays[k] for k in local_arrays}
        options_c = dict(
            ratio=cfg["compression-ratio"],
            bits=cfg["compression-bits"],
            rng=np.random.default_rng([cfg["seed"], partition_id, config["server-round"], 2]),
        )
        if cfg["error-feedback"]:
            residual = to_arrays(context.state[STATE_EF_RESIDUAL]) if STATE_EF_RESIDUAL in context.state else None
            decoded, nbytes, new_residual = compress_with_feedback(update, residual, cfg["compression"], **options_c)
            context.state[STATE_EF_RESIDUAL] = to_record(new_residual)
            update = {k: decoded[k] + new_residual[k] for k in decoded}  # what was compressed: update + carried-over residual
        else:
            decoded, nbytes = compress(update, cfg["compression"], **options_c)
        local_arrays = {k: (global_arrays[k] + decoded[k]).astype(global_arrays[k].dtype) for k in local_arrays}
        extra_metrics["compression-error"] = relative_error(update, decoded)
        extra_metrics["upload-bytes"] = nbytes + sum(record_bytes(r) for r in extra_records.values())

    content = RecordDict({
        "arrays": to_record(local_arrays),
        "metrics": MetricRecord({
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
            "local-steps": steps,  # optimiser steps taken; clients with more data take more (FedNova)
            **extra_metrics,
        }),
        **extra_records,
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    cfg = context.run_config
    partition_id = context.node_config["partition-id"]
    device = get_device()
    model = build_model(cfg["dataset"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainloader, valloader = load_client_data(partition_id, context.node_config["num-partitions"], cfg)
    loss, acc = test_fn(model, valloader, device)
    metrics = {"eval_loss": loss, "eval_acc": acc, "num-examples": len(valloader.dataset)}

    if cfg["personalize-epochs"] > 0:
        # Personalisation gap: how much a few local epochs on top of the *received* global model help
        # on this client's held-out data. The global model itself is evaluated above and left untouched.
        seed = [cfg["seed"], partition_id, msg.content["config"]["server-round"], 3]
        torch.manual_seed(int(np.random.SeedSequence(seed).generate_state(1)[0]))
        train_fn(model, trainloader, cfg["personalize-epochs"], cfg["learning-rate"], device)
        metrics["eval_loss_personalized"], metrics["eval_acc_personalized"] = test_fn(model, valloader, device)

    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
