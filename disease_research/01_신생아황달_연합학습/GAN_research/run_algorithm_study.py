"""Validation-only tuning then locked test evaluation of federated optimizers.

CPU image cache and one GPU client at a time accommodate other GPU workloads.
No test prediction is made until every configuration and seed has finished.
"""
import argparse
from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from runtime import DEVICE, device_name, require_device
from torch.nn import functional as F

from run_federated_study import ROOT, MultiModalNet, average_states, seed_all, patients_table, metrics
from run_skin_bias_study import patient_groups
from skin_bias import normalize, attenuate
from federated_algorithms import (bn_keys, train_mode, proximal_penalty,
                                  scaffold_correct, scaffold_control, personalized_state)

OUTPUT = ROOT / "results/algorithm_study"
CONTEXTS = [(data, bias) for data in ["real", "paired100"] for bias in [False, True]]
CONFIGS = [dict(id="fedavg_adam", family="fedavg_adam", optimizer="adam", lr=.0001),
           *[dict(id=f"fedprox_mu{mu:g}", family="fedprox", optimizer="adam", lr=.0001, mu=mu) for mu in [.01, .1, 1.]],
           *[dict(id=f"fedavg_sgd_lr{lr:g}", family="fedavg_sgd", optimizer="sgd", lr=lr) for lr in [.001, .01, .1]],
           *[dict(id=f"scaffold_lr{lr:g}", family="scaffold", optimizer="sgd", lr=lr) for lr in [.001, .01, .1]],
           dict(id="fedbn_adam", family="fedbn", optimizer="adam", lr=.0001),
           dict(id="fedavg_trainbn", family="fedavg_trainbn", optimizer="adam", lr=.0001)]
FAMILIES = list(dict.fromkeys(c["family"] for c in CONFIGS))


def cpu_state(model):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def cache(data):
    images = torch.tensor(data["images"], dtype=torch.float32).permute(0, 3, 1, 2)/255
    views = pd.Series(data["patient"]).value_counts()
    return {"images": normalize(images), "meta": torch.tensor(data["meta"]),
            "blood": torch.tensor(data["blood"]), "treatment": torch.tensor(data["treatment"]),
            "weight": torch.tensor([1/views[p] for p in data["patient"]], dtype=torch.float32)}


def local_update(model, tensors, indices, extra, extra_indices, ratio, bias, config,
                 anchor, server_control, client_control, seed):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    nuisance = torch.Generator(device=DEVICE).manual_seed(seed+900000)
    optimizer = (torch.optim.Adam if config["optimizer"] == "adam" else torch.optim.SGD)(model.parameters(), lr=config["lr"])
    if config["family"] == "scaffold":
        server_control = {k: v.to(DEVICE) for k, v in server_control.items()}
        client_control = {k: v.to(DEVICE) for k, v in client_control.items()}
    train_mode(model, config["family"] in ["fedbn", "fedavg_trainbn"])
    order = torch.as_tensor(indices)[torch.randperm(len(indices), generator=generator, device=DEVICE).cpu()]
    steps = 0
    # Only FedProx and SCAFFOLD need the round anchor on GPU.
    for selected in order.split(32):
        source = selected
        image = tensors["images"][selected]
        if ratio:
            choose = torch.randint(len(extra_indices), (len(selected),), generator=generator, device=DEVICE).cpu()
            synthetic = extra_indices[choose]
            source = torch.cat([source, extra["source"][synthetic]])
            image = torch.cat([image, extra["images"][synthetic]])
        image = image.to(DEVICE)
        meta, y, cls, weight = [tensors[k][source].to(DEVICE) for k in ["meta", "blood", "treatment", "weight"]]
        if torch.rand((), generator=generator, device=DEVICE) < .5:
            image = image.flip(-1)
        prediction, logits = model(image, meta)
        loss_each = F.smooth_l1_loss(prediction[:, 0]*30, y, reduction="none") + \
                    .5*F.binary_cross_entropy_with_logits(logits[:, 0], cls, reduction="none")
        if bias:
            gain = .65+.35*torch.rand(len(image), generator=nuisance, device=DEVICE)
            changed, changed_logits = model(attenuate(image, gain), meta)
            changed_loss = F.smooth_l1_loss(changed[:, 0]*30, y, reduction="none") + \
                           .5*F.binary_cross_entropy_with_logits(changed_logits[:, 0], cls, reduction="none")
            loss_each = .5*(loss_each+changed_loss) + \
                        .5*F.smooth_l1_loss(changed[:, 0]*30, prediction[:, 0]*30, reduction="none") + \
                        .5*(changed_logits[:, 0].sigmoid()-logits[:, 0].sigmoid()).square()
        loss = (loss_each*weight).sum()/weight.sum()
        if config["family"] == "fedprox":
            loss = loss+config["mu"]*proximal_penalty(model, anchor)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite local loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        # Clip the stochastic task gradient before SCAFFOLD's additive control.
        # This is disclosed as a clipped-gradient variant of the original method.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if config["family"] == "scaffold":
            scaffold_correct(model, server_control, client_control)
        optimizer.step(); steps += 1
    return cpu_state(model), steps


@torch.no_grad()
def evaluate(model, data, tensors, indices, gain=1.):
    model.eval(); predictions, probabilities = [], []
    for selected in np.array_split(indices, max(1, int(np.ceil(len(indices)/64)))):
        image = tensors["images"][selected].to(DEVICE)
        if gain != 1.:
            image = attenuate(image, gain)
        regression, logits = model(image, tensors["meta"][selected].to(DEVICE))
        predictions.extend((regression[:, 0]*30).cpu().tolist())
        probabilities.extend(logits[:, 0].sigmoid().cpu().tolist())
    return patients_table(data, indices, predictions, probabilities)


def evaluate_state(model, state, data, tensors, indices, gain=1.):
    if isinstance(state, list):
        tables = []
        for client in range(3):
            model.load_state_dict(state[client])
            tables.append(evaluate(model, data, tensors, indices[data["client"][indices] == client], gain))
        return pd.concat(tables).sort_values("patient").reset_index(drop=True)
    model.load_state_dict(state)
    return evaluate(model, data, tensors, indices, gain)


def train_one(model, initial, data, tensors, extra, training, synthetic_indices, condition, bias, config, seed, rounds):
    name = f"{condition}_bias{int(bias)}_{config['id']}_seed{seed}"
    result_path = OUTPUT / f"{name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    start = time.monotonic(); seed_all(seed); model.load_state_dict(initial)
    state = initial
    keys = bn_keys(model)
    local_bn = [{key: initial[key].clone() for key in keys} for _ in range(3)]
    controls = [OrderedDict((key, torch.zeros_like(p, device="cpu")) for key, p in model.named_parameters()) for _ in range(3)] if config["family"] == "scaffold" else None
    server_control = OrderedDict((key, torch.zeros_like(p, device="cpu")) for key, p in model.named_parameters()) if controls else None
    validation = np.flatnonzero(data["split"] == "val")
    weights = [len(np.unique(data["patient"][ix])) for ix in training]
    history = []; best_mae = float("inf"); best_state = None
    max_control_error = 0.
    for round_id in range(1, rounds+1):
        states = []
        anchor = OrderedDict((key, value.to(DEVICE)) for key, value in state.items() if key in dict(model.named_parameters())) if config["family"] in ["fedprox", "scaffold"] else None
        old_server = server_control
        for client in range(3):
            model.load_state_dict(personalized_state(state, local_bn[client]) if config["family"] == "fedbn" else state)
            local, steps = local_update(model, tensors, training[client], extra, synthetic_indices[client],
                                        condition == "paired100", bias, config, anchor,
                                        server_control, controls[client] if controls else None,
                                        seed+round_id*101+client)
            states.append(local)
            if config["family"] == "fedbn":
                local_bn[client] = {key: local[key] for key in keys}
            if controls:
                controls[client] = scaffold_control(controls[client], old_server, anchor, local, steps, config["lr"])
        if controls:
            server_control = average_states(controls, weights)
            max_control_error = max(max_control_error, max(float((server_control[k]-sum(w*c[k] for w, c in zip(weights, controls))/sum(weights)).abs().max()) for k in server_control))
        if config["family"] == "fedbn":
            shared = average_states([{k: v for k, v in s.items() if k not in keys} for s in states], weights)
            state = OrderedDict((key, initial[key] if key in keys else shared[key]) for key in initial)
            evaluation_state = [personalized_state(state, bn) for bn in local_bn]
        else:
            state = average_states(states, weights)
            evaluation_state = state
        frame = evaluate_state(model, evaluation_state, data, tensors, validation)
        score = metrics(frame)
        history.append({"round": round_id, **score})
        if score["mae_mgdl"] < best_mae:
            best_mae = score["mae_mgdl"]
            best_round = round_id
            best_state = [{k: v.clone() for k, v in s.items()} for s in evaluation_state] if isinstance(evaluation_state, list) else {k: v.clone() for k, v in evaluation_state.items()}
        if round_id % 5 == 0:
            print(f"{name} round={round_id}/{rounds} val={best_mae:.4f} elapsed={time.monotonic()-start:.1f}s", flush=True)
    torch.save(best_state, OUTPUT / f"{name}.pt")
    result = {"name": name, "condition": condition, "bias": bias, "config": config,
              "seed": seed, "best_round": best_round, "validation_mae": best_mae,
              "history": history, "seconds": time.monotonic()-start,
              "scaffold_mean_control_max_error": max_control_error,
              "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"FINISHED {name} val={best_mae:.5f}", flush=True)
    del best_state, states, anchor, controls, server_control
    torch.cuda.empty_cache()
    return result


def run(rounds=20, plan_only=False):
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    require_device()
    OUTPUT.mkdir(exist_ok=True)
    data = np.load(ROOT / "data/real_data.npz")
    training = [np.flatnonzero((data["split"] == "train") & (data["client"] == c)) for c in range(3)]
    patient_weights = [len(np.unique(data["patient"][ix])) for ix in training]
    protocol = {"stage": "research_only", "device": device_name(), "rounds": rounds,
                "seeds": [17, 29, 43], "contexts": CONTEXTS, "configurations": CONFIGS,
                "tuning": "seed17 best patient validation MAE per family/context; freeze configuration before training seeds29/43; test only after all training",
                "checkpoint": "minimum patient validation MAE separately per seed, fixed 20 rounds",
                "batch_size": 32, "local_epochs": 1, "all_clients_each_round": True,
                "client_weights": patient_weights, "server_learning_rate": 1.,
                "scaffold": "patient-weighted option II, global control is patient-weighted mean of client controls; plain SGD no momentum/weight decay, clipped task gradient then additive control, actual local step count",
                "fedbn": "all BN affine/running statistics/counters local; known-client personalized inference, no unseen-client test adaptation",
                "fedavg_trainbn": "matched training-mode BN control; running BN parameters averaged as standard model-state heuristic, not pooled population moments",
                "scope": "single GPU sequential 3-client simulation; random stratified client assignment, no newly constructed feature shift",
                "precision": "float32 high matmul precision, cudnn benchmark disabled, CPU image cache and streamed batches",
                "data_sha256": sha256((ROOT/'data/real_data.npz').read_bytes()).hexdigest()}
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == json.loads(json.dumps(protocol)), "Refuse changed resume protocol"
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    if plan_only:
        return
    tensors = cache(data)
    synthetic = np.load(ROOT / "data/gan_paired_synthetic.npz")
    source = synthetic["source_index"]
    assert (data["split"][source] == "train").all()
    np.testing.assert_array_equal(data["client"][source], synthetic["client"])
    extra = {"images": normalize(torch.tensor(synthetic["images"], dtype=torch.float32).permute(0, 3, 1, 2)/255),
             "source": torch.tensor(source)}
    training = [np.flatnonzero((data["split"] == "train") & (data["client"] == c)) for c in range(3)]
    synthetic_indices = [torch.tensor(np.flatnonzero(synthetic["client"] == c)) for c in range(3)]
    results = []; selection = []
    for seed in [17, 29, 43]:
        seed_all(seed); model = MultiModalNet().to(DEVICE); initial = cpu_state(model)
        for condition, bias in CONTEXTS:
            configs = CONFIGS if seed == 17 else list({s["config"]["id"]: s["config"] for s in selection if s["condition"] == condition and s["bias"] == bias}.values())
            for config in configs:
                results.append(train_one(model, initial, data, tensors, extra, training, synthetic_indices,
                                         condition, bias, config, seed, rounds))
            if seed == 17:
                for family in FAMILIES:
                    candidates = [r for r in results if r["condition"] == condition and r["bias"] == bias and r["config"]["family"] == family]
                    selected = min(candidates, key=lambda r: r["validation_mae"])
                    selection.append({"condition": condition, "bias": bias, "family": family,
                                      "config": selected["config"], "tuning_validation_mae": selected["validation_mae"]})
                local_selection = [s for s in selection if s["condition"] == condition and s["bias"] == bias]
                scaffold_lr = next(s["config"]["lr"] for s in local_selection if s["family"] == "scaffold")
                sgd_lr = next(s["config"]["lr"] for s in local_selection if s["family"] == "fedavg_sgd")
                if scaffold_lr != sgd_lr:
                    matched = next(r for r in results if r["condition"] == condition and r["bias"] == bias and r["seed"] == 17
                                   and r["config"]["family"] == "fedavg_sgd" and r["config"]["lr"] == scaffold_lr)
                    selection.append({"condition": condition, "bias": bias, "family": "fedavg_sgd_matched",
                                      "config": matched["config"], "tuning_validation_mae": matched["validation_mae"]})
        if seed == 17:
            (OUTPUT / "selection.json").write_text(json.dumps(selection, indent=2))
        del model, initial; torch.cuda.empty_cache()
    (OUTPUT / "training_summary.json").write_text(json.dumps(results, indent=2))
    # Lock primary method/context winners using validation alone, before opening test.
    validation_winners = []
    for condition, bias in CONTEXTS:
        candidates = []
        for selected in selection:
            if selected["condition"] != condition or selected["bias"] != bias or selected["family"] not in ["fedavg_adam", "fedprox", "scaffold", "fedbn"]:
                continue
            scores = [r["validation_mae"] for r in results if r["condition"] == condition and r["bias"] == bias and r["config"]["id"] == selected["config"]["id"]]
            assert len(scores) == 3
            candidates.append({"condition": condition, "bias": bias, "family": selected["family"],
                               "validation_mae_mean": float(np.mean(scores))})
        validation_winners.append(min(candidates, key=lambda c: c["validation_mae_mean"]))
    (OUTPUT / "validation_winners_locked.json").write_text(json.dumps({"per_context": validation_winners,
        "overall": min(validation_winners, key=lambda c: c["validation_mae_mean"]), "test_evaluated": False}, indent=2))
    # Only now unlock test inference; tuning code never inspects its outcomes.
    groups, _ = patient_groups(data)
    test = np.flatnonzero(data["split"] == "test")
    model = MultiModalNet(pretrained=False).to(DEVICE)
    for s in selection:
        for seed in [17, 29, 43]:
            name = f"{s['condition']}_bias{int(s['bias'])}_{s['config']['id']}_seed{seed}"
            state = torch.load(OUTPUT / f"{name}.pt", map_location="cpu", weights_only=True)
            for gain in [1., .55, .75, .9]:
                frame = evaluate_state(model, state, data, tensors, test, gain)
                frame = frame.merge(groups[["patient", "brightness", "group"]], on="patient", validate="one_to_one")
                frame.to_csv(OUTPUT / f"{name}_gain{gain:.2f}_test.csv", index=False)
    (OUTPUT / "complete.json").write_text(json.dumps({"trainings": len(results), "selected_models": len(selection)*3,
            "patient_test_predictions": len(selection)*3*4, "stage": "research_only"}, indent=2))
    print("ALL OPTIMIZER TRAINING AND LOCKED TEST EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(args.rounds, args.plan_only)
