"""Patient-disjoint actual-GPU comparison using three sequential FL clients.

Every round uses all real local images once; augmentation adds samples to the
same optimizer steps. Synthetic images never change real-patient client weights.
Test predictions are made only after selecting a checkpoint by validation MAE.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, mean_absolute_error, mean_squared_error)
import torch
from runtime import DEVICE, device_name, require_device
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "NeoJaundice"))
from src.model import MultiModalNet
from src.federated import average_states


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def patients_table(data, indices, blood_prediction, probabilities):
    frame = pd.DataFrame({"patient": data["patient"][indices], "client": data["client"][indices],
                          "blood": data["blood"][indices], "treatment": data["treatment"][indices],
                          "prediction": blood_prediction, "probability": probabilities})
    return frame.groupby("patient", sort=True).agg({"client": "first", "blood": "first", "treatment": "first",
                                                   "prediction": "mean", "probability": "mean"}).reset_index()


def metrics(table):
    label = table.treatment.to_numpy(); binary = table.probability.to_numpy() >= .5
    return {"mae_mgdl": float(mean_absolute_error(table.blood, table.prediction)),
            "rmse_mgdl": float(np.sqrt(mean_squared_error(table.blood, table.prediction))),
            "accuracy": float(accuracy_score(label, binary)),
            "balanced_accuracy": float(balanced_accuracy_score(label, binary)),
            "f1": float(f1_score(label, binary, zero_division=0)),
            "roc_auc": float(roc_auc_score(label, table.probability)) if len(np.unique(label)) == 2 else None,
            "n_patients": len(table)}


@torch.no_grad()
def evaluate(model, data, tensors, indices):
    model.eval()
    blood, probabilities = [], []
    for batch in np.array_split(indices, max(1, int(np.ceil(len(indices)/128)))):
        reg, cls = model(tensors["images"][batch], tensors["meta"][batch])
        blood.extend((reg[:, 0]*30).cpu().tolist())
        probabilities.extend(cls[:, 0].sigmoid().cpu().tolist())
    table = patients_table(data, indices, blood, probabilities)
    return metrics(table), table


def local_update(model, tensors, indices, extra, extra_indices, ratio, batch_size, epochs, seed):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    indices = torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
    model.train()
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), generator=generator, device=DEVICE)]
        for selected in order.split(batch_size):
            image, meta = tensors["images"][selected], tensors["meta"][selected]
            y, cls, weight = tensors["blood"][selected], tensors["treatment"][selected], tensors["weight"][selected]
            if ratio:
                count = max(1, int(round(len(selected)*ratio)))
                choose = torch.randint(len(extra_indices), (count,), generator=generator, device=DEVICE)
                synthetic = extra_indices[choose]
                source = extra["source"][synthetic]
                image = torch.cat([image, extra["images"][synthetic]])
                meta = torch.cat([meta, tensors["meta"][source]])
                y = torch.cat([y, tensors["blood"][source]])
                cls = torch.cat([cls, tensors["treatment"][source]])
                weight = torch.cat([weight, tensors["weight"][source]])
            # Geometry only: diagnostic colors are not randomly shifted.
            if torch.rand((), generator=generator, device=DEVICE) < .5:
                image = image.flip(-1)
            prediction, logits = model(image, meta)
            loss_each = F.smooth_l1_loss(prediction[:, 0]*30, y, reduction="none") + \
                        .5*F.binary_cross_entropy_with_logits(logits[:, 0], cls, reduction="none")
            loss = (loss_each*weight).sum()/weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite prediction loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


def make_tensors(data):
    images = torch.as_tensor(data["images"], dtype=torch.float32, device=DEVICE).permute(0, 3, 1, 2)/255
    mean = torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]
    views = pd.Series(data["patient"]).value_counts()
    return {"images": (images-mean)/std,
            "meta": torch.as_tensor(data["meta"], device=DEVICE),
            "blood": torch.as_tensor(data["blood"], device=DEVICE),
            "treatment": torch.as_tensor(data["treatment"], device=DEVICE),
            "weight": torch.tensor([1/views[p] for p in data["patient"]], dtype=torch.float32, device=DEVICE)}


def extras(data, synthetic):
    source = synthetic["source_index"]
    if not np.all(data["split"][source] == "train") or not np.array_equal(data["client"][source], synthetic["client"]):
        raise ValueError("Synthetic provenance crosses split/client boundaries")
    rng = np.random.default_rng(91)
    blood = data["blood"][source]
    # Reproduce the old procedural generator (BGR then RGB), cropped away from its corner chart.
    old = np.zeros_like(synthetic["images"], dtype=np.int16)
    old[:, :, :, 0] = 255; old[:, :, :, 1] = 200
    old[:, :, :, 2] = (200-(100*blood/20).astype(int))[:, None, None]
    old = np.clip(old+rng.integers(-20, 20, old.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    result = {}
    for name, array in [("gan", synthetic["images"]), ("procedural", old)]:
        image = torch.tensor(array, dtype=torch.float32, device=DEVICE).permute(0, 3, 1, 2)/255
        mean = torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]
        result[name] = {"images": (image-mean)/std, "source": torch.tensor(source, device=DEVICE)}
    return result


def run(rounds=20, local_epochs=1, seeds=(17, 29, 43), batch_size=32, paired=False, procedural_half=False):
    require_device()
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    data = np.load(ROOT / "data/real_data.npz")
    synthetic = np.load(ROOT / "data" / ("gan_paired_synthetic.npz" if paired else "gan_synthetic.npz"))
    tensors, extra = make_tensors(data), extras(data, synthetic)
    training = [np.flatnonzero((data["split"] == "train") & (data["client"] == client)) for client in range(3)]
    val = np.flatnonzero(data["split"] == "val"); test = np.flatnonzero(data["split"] == "test")
    weights = [len(np.unique(data["patient"][indices])) for indices in training]
    synthetic_indices = [torch.tensor(np.flatnonzero(synthetic["client"] == client), device=DEVICE) for client in range(3)]
    output = ROOT / "results" / ("procedural_half_prediction" if procedural_half else "paired_prediction" if paired else "prediction")
    output.mkdir(parents=True, exist_ok=True)
    study = {"stage": "research_only", "device": device_name(), "rounds": rounds,
             "local_epochs": local_epochs, "seeds": list(seeds), "batch_size": batch_size,
             "primary_metric": "patient-level MAE in mg/dL on untouched real test patients",
             "model": "ImageNet ResNet18 + 4 metadata features; frozen common CNN BatchNorm; metadata LayerNorm",
             "loss": "SmoothL1(mg/dL) + 0.5 * BCEWithLogits", "client_weights_real_patients": weights,
             "protocol": "single-device sequential client simulation; only model states averaged",
             "checkpoint_selection": "lowest patient validation MAE; test not evaluated during selection",
             "synthetic_source": "local paired residual GAN" if paired else "local noise-conditioned GAN and procedural control",
             "optimizer_budget": "fixed real-data passes and optimizer steps; synthetic samples added within the same updates",
             "steps_per_round": {"federated_total": sum(int(np.ceil(len(x)/batch_size)) for x in training)*local_epochs,
                                 "central": int(np.ceil(sum(map(len, training))/batch_size))*local_epochs},
             "runs": []}
    cases = [("central_real", "central", None, 0), ("isolated_real", "isolated", None, 0),
             ("federated_real", "federated", None, 0),
             ("federated_procedural_100", "federated", "procedural", 1),
             ("federated_gan_050", "federated", "gan", .5),
             ("federated_gan_100", "federated", "gan", 1)]
    if paired:
        cases = [("federated_paired_gan_050", "federated", "gan", .5),
                 ("federated_paired_gan_100", "federated", "gan", 1)]
    if procedural_half:
        cases = [("federated_procedural_050", "federated", "procedural", .5)]
    for seed in seeds:
        seed_all(seed)
        model = MultiModalNet().to(DEVICE)
        initial = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
        for name, regime, generation, ratio in cases:
            start = time.monotonic()
            seed_all(seed)
            model.load_state_dict(initial)
            best_mae, best_state, best_round = float("inf"), None, None
            history = []
            local_states = [OrderedDict((k, v.clone()) for k, v in initial.items()) for _ in range(3)] if regime == "isolated" else None
            local_best = [None]*3; local_best_mae = [float("inf")]*3
            local_best_round = [None]*3
            for round_id in range(1, rounds+1):
                if regime == "central":
                    local_update(model, tensors, np.concatenate(training), None, None, 0,
                                 batch_size, local_epochs, seed+round_id*101)
                else:
                    global_state = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
                    states = []
                    for client in range(3):
                        model.load_state_dict(local_states[client] if regime == "isolated" else global_state)
                        state = local_update(model, tensors, training[client], extra.get(generation),
                                             synthetic_indices[client], ratio, batch_size, local_epochs,
                                             seed+round_id*101+client)
                        states.append(state)
                        if regime == "isolated":
                            local_states[client] = state
                            local_val = val[data["client"][val] == client]
                            score, _ = evaluate(model, data, tensors, local_val)
                            if score["mae_mgdl"] < local_best_mae[client]:
                                local_best_mae[client] = score["mae_mgdl"]
                                local_best[client] = OrderedDict((k, v.clone()) for k, v in state.items())
                                local_best_round[client] = round_id
                    if regime != "isolated":
                        model.load_state_dict(average_states(states, weights))
                if regime != "isolated":
                    score, _ = evaluate(model, data, tensors, val)
                    history.append({"round": round_id, **score})
                    if score["mae_mgdl"] < best_mae:
                        best_mae, best_round = score["mae_mgdl"], round_id
                        best_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
                if round_id % 5 == 0:
                    current = min(local_best_mae) if regime == "isolated" else best_mae
                    print(f"seed={seed} case={name} round={round_id}/{rounds} val_MAE={current:.4f} elapsed={time.monotonic()-start:.1f}s", flush=True)
            if regime == "isolated":
                tables = []
                for client in range(3):
                    model.load_state_dict(local_best[client])
                    _, table = evaluate(model, data, tensors, test[data["client"][test] == client])
                    tables.append(table)
                table = pd.concat(tables).sort_values("patient")
                best_round = local_best_round
                torch.save([{k: v.cpu() for k, v in state.items()} for state in local_best], output / f"{name}_seed{seed}.pt")
            else:
                model.load_state_dict(best_state)
                _, table = evaluate(model, data, tensors, test)
                torch.save(best_state, output / f"{name}_seed{seed}.pt")
            table.to_csv(output / f"{name}_seed{seed}_test.csv", index=False)
            result = {"case": name, "seed": seed, "best_validation_round": best_round,
                      "metrics": metrics(table), "elapsed_s": time.monotonic()-start,
                      "per_client": {str(c): metrics(part) for c, part in table.groupby("client")}}
            study["runs"].append(result)
            (output / f"{name}_seed{seed}_validation.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            (output / "study.json").write_text(json.dumps(study, indent=2, allow_nan=False), encoding="utf-8")
            print(f"FINISHED {name} seed={seed}: {result['metrics']}", flush=True)
        del model, initial
        torch.cuda.empty_cache()
    rows = [{"case": r["case"], "seed": r["seed"], **r["metrics"]} for r in study["runs"]]
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    print("All prediction experiments complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--procedural-half", action="store_true")
    args = parser.parse_args()
    run(args.rounds, args.local_epochs, args.seeds, args.batch_size, args.paired, args.procedural_half)
