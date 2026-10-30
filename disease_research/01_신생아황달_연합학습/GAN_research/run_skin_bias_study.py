"""Prespecified brightness-proxy and counterfactual study on fixed patient split.

Original baselines are re-used, not re-selected. Three new methods, two data
conditions, three seeds, twenty rounds. No test-driven hyperparameter tuning.
"""
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from runtime import DEVICE, device_name, require_device
from torch.nn import functional as F

from run_federated_study import (ROOT, MultiModalNet, average_states, seed_all,
                                 make_tensors, extras, evaluate, metrics)
from skin_bias import attenuate, suppress_brightness

OUTPUT = ROOT / "results/skin_bias"
METHODS = ["baseline", "gain_aug", "gain_consistency", "chromaticity"]
GAINS = [.55, .75, .9]


def patient_groups(data):
    # Encoded RGB luminance proxy, not calibrated physical reflectance or melanin.
    brightness = (data["images"].astype(np.float64)/255 @ np.array([.2126, .7152, .0722])).mean((1, 2))
    frame = pd.DataFrame({"patient": data["patient"], "brightness": brightness, "split": data["split"]})
    frame = frame.groupby("patient", sort=True).agg({"brightness": "mean", "split": "first"}).reset_index()
    cutoffs = np.quantile(frame.loc[frame.split == "train", "brightness"], [1/3, 2/3])
    frame["group"] = np.searchsorted(cutoffs, frame.brightness, side="right")
    return frame, cutoffs


def local_update(model, tensors, indices, extra, extra_indices, ratio, method, seed):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    nuisance = torch.Generator(device=DEVICE).manual_seed(seed+900000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    indices = torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
    model.train()
    order = indices[torch.randperm(len(indices), generator=generator, device=DEVICE)]
    for selected in order.split(32):
        source = selected
        image = tensors["images"][selected]
        if ratio:
            choose = torch.randint(len(extra_indices), (len(selected),), generator=generator, device=DEVICE)
            synthetic = extra_indices[choose]
            source = torch.cat([source, extra["source"][synthetic]])
            image = torch.cat([image, extra["images"][synthetic]])
        meta, y, cls, weight = [tensors[k][source] for k in ["meta", "blood", "treatment", "weight"]]
        if torch.rand((), generator=generator, device=DEVICE) < .5:
            image = image.flip(-1)
        image = suppress_brightness(image, method)
        prediction, logits = model(image, meta)
        loss_each = F.smooth_l1_loss(prediction[:, 0]*30, y, reduction="none") + \
                    .5*F.binary_cross_entropy_with_logits(logits[:, 0], cls, reduction="none")
        if method in ["gain_aug", "gain_consistency"]:
            gain = .65+.35*torch.rand(len(image), generator=nuisance, device=DEVICE)
            changed_prediction, changed_logits = model(attenuate(image, gain), meta)
            changed_loss = F.smooth_l1_loss(changed_prediction[:, 0]*30, y, reduction="none") + \
                           .5*F.binary_cross_entropy_with_logits(changed_logits[:, 0], cls, reduction="none")
            loss_each = .5*(loss_each+changed_loss)
            if method == "gain_consistency":
                # Symmetric gradients, both branches supervised; coefficients frozen before testing.
                loss_each = loss_each + .5*F.smooth_l1_loss(changed_prediction[:, 0]*30,
                                                         prediction[:, 0]*30, reduction="none") + \
                            .5*(changed_logits[:, 0].sigmoid()-logits[:, 0].sigmoid()).square()
        loss = (loss_each*weight).sum()/weight.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


def inputs(tensors, method, gain=1.):
    result = dict(tensors)
    result["images"] = suppress_brightness(attenuate(tensors["images"], gain), method)
    return result


def run():
    require_device()
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    OUTPUT.mkdir(exist_ok=True)
    data = np.load(ROOT / "data/real_data.npz")
    groups, cutoffs = patient_groups(data)
    groups.to_csv(OUTPUT / "patient_brightness_groups.csv", index=False)
    protocol = {"stage": "research_only", "intrinsic_skin_labels_available": False,
                "device": device_name(), "methods": METHODS,
                "data_conditions": ["real", "paired100"], "seeds": [17, 29, 43], "rounds": 20,
                "train_gain_range": [.65, 1.], "test_gains": GAINS,
                "consistency_coefficient_regression": .5, "consistency_coefficient_probability": .5,
                "group_cutoffs_train_patient_terciles": cutoffs.tolist(),
                "group_definition": "mean photo encoded-RGB weighted brightness, averaged by patient; not intrinsic pigmentation",
                "selection": "minimum original-image patient validation MAE; no fairness/test selection",
                "baseline_reuse": "previous frozen original and paired100 checkpoints, same seeds and protocol",
                "compute": "same optimizer updates and real passes; aug and consistency use two forward passes",
                "limitation": "uniform encoded-RGB attenuation is not a biophysical melanin intervention",
                "runs": []}
    protocol_path = OUTPUT / "protocol.json"
    # Protocol is persisted before any new predictions.
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    tensors = make_tensors(data)
    synthetic = np.load(ROOT / "data/gan_paired_synthetic.npz")
    extra = extras(data, synthetic)["gan"]
    training = [np.flatnonzero((data["split"] == "train") & (data["client"] == c)) for c in range(3)]
    weights = [len(np.unique(data["patient"][ix])) for ix in training]
    synthetic_indices = [torch.tensor(np.flatnonzero(synthetic["client"] == c), device=DEVICE) for c in range(3)]
    val, test = [np.flatnonzero(data["split"] == s) for s in ["val", "test"]]
    for seed in protocol["seeds"]:
        seed_all(seed); model = MultiModalNet().to(DEVICE)
        initial = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
        for condition in protocol["data_conditions"]:
            ratio = int(condition == "paired100")
            for method in METHODS:
                name = f"{condition}_{method}_seed{seed}"
                started = time.monotonic(); seed_all(seed); model.load_state_dict(initial)
                history = []; best_mae, best_round, best_state = float("inf"), None, None
                if method == "baseline":
                    folder, old_case = ("prediction", "federated_real") if not ratio else ("paired_prediction", "federated_paired_gan_100")
                    checkpoint = ROOT / "results" / folder / f"{old_case}_seed{seed}.pt"
                    best_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                else:
                    for round_id in range(1, 21):
                        global_state = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
                        states = []
                        for c in range(3):
                            model.load_state_dict(global_state)
                            states.append(local_update(model, tensors, training[c], extra, synthetic_indices[c],
                                                       ratio, method, seed+round_id*101+c))
                        model.load_state_dict(average_states(states, weights))
                        score, _ = evaluate(model, data, inputs(tensors, method), val)
                        history.append({"round": round_id, **score})
                        if score["mae_mgdl"] < best_mae:
                            best_mae, best_round = score["mae_mgdl"], round_id
                            best_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
                        if round_id % 5 == 0:
                            print(f"{name} round={round_id} val={best_mae:.4f} seconds={time.monotonic()-started:.1f}", flush=True)
                    torch.save(best_state, OUTPUT / f"{name}.pt")
                model.load_state_dict(best_state)
                for gain in [1.] + GAINS:
                    _, frame = evaluate(model, data, inputs(tensors, method, gain), test)
                    frame = frame.merge(groups[["patient", "brightness", "group"]], on="patient", validate="one_to_one")
                    frame.to_csv(OUTPUT / f"{name}_gain{gain:.2f}_test.csv", index=False)
                (OUTPUT / f"{name}_validation.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                protocol["runs"].append({"condition": condition, "method": method, "seed": seed,
                                         "best_round": best_round, "seconds": time.monotonic()-started})
                protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
                print(f"FINISHED {name}", flush=True)
        del model, initial; torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
