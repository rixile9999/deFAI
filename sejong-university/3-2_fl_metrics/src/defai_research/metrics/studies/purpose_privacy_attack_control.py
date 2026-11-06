from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from ._paths import ROOT, source_path, relative_path
import numpy as np
from scipy.stats import t
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from . import _helpers as core

BASE = ROOT / "results/purpose_privacy/research_private"
OUT = BASE / "attack_sensitivity_control"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(seed, pool, x, y, name, plan):
    rng = np.random.default_rng(seed)
    member, nonmember = [], []
    for label in range(10):
        chosen = rng.choice(pool[y[pool] == label], 6, replace=False)
        member.extend(chosen[:3].tolist()); nonmember.extend(chosen[3:].tolist())
    member, nonmember = np.asarray(member), np.asarray(nonmember)
    assert not(set(member) & set(nonmember))
    model = core.initial_parameters(plan["public_initial_model_seed"])
    client = core.DigitsClient(0, x[member], y[member], plan["public_shuffle_seed"]).to_client()
    response = client.fit(FitIns(ndarrays_to_parameters(model), {"round": 1,
        "local_epochs": 100, "learning_rate": .14, "batch_size": 10, "l2": .0001}))
    model = parameters_to_ndarrays(response.parameters)
    ids = np.concatenate([member, nonmember])
    p = core.predict(model, x[ids]).astype(np.float64)
    loss = -np.log(np.clip(p[np.arange(60), y[ids]], 1e-30, 1.))
    score = -loss
    labels = np.repeat([1, 0], 30)
    record = {"run_id": name, "train_records": 30, "nonmember_probe_records": 30,
        "epochs": 100, "Flower_fit_calls": 1, "actual_optimizer_steps": response.metrics["optimizer_steps"],
        "training_accuracy": float(np.mean(p[:30].argmax(1) == y[member])),
        "nonmember_accuracy": float(np.mean(p[30:].argmax(1) == y[nonmember])),
        "training_macro_F1": float(f1_score(y[member], p[:30].argmax(1), labels=np.arange(10), average="macro", zero_division=0)),
        "nonmember_macro_F1": float(f1_score(y[nonmember], p[30:].argmax(1), labels=np.arange(10), average="macro", zero_division=0)),
        "member_loss_mean": float(loss[:30].mean()), "nonmember_loss_mean": float(loss[30:].mean()),
        "membership_AUC": float(roc_auc_score(labels, score))}
    np.savez_compressed(OUT / f"{name}_private_scores.npz", score=score, membership=labels, source_ids=ids)
    np.savez_compressed(OUT / f"{name}_private_model.npz", **{f"array_{i}": a for i, a in enumerate(model)})
    return record, labels, score


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError("Preserve the frozen positive-control study; do not tune/retry it")
    OUT.mkdir()
    main_plan = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    plan = {"stage": "frozen_before_any_control_training", "purpose": "Sensitivity positive control for the same negative-CE record-membership score; not a replacement or retuning of the main study.",
        "records_per_class_train": 3, "records_per_class_nonmember": 3,
        "local_epochs": 100, "batch_size": 10, "learning_rate": .14, "l2": .0001,
        "public_initial_model_seed": main_plan["public_initial_model_seed"],
        "public_shuffle_seed": main_plan["public_client_shuffle_seed"],
        "private_shadow_seed": main_plan["private_shadow_data_seed"]+11,
        "private_target_seeds": [seed+11 for seed in main_plan["private_target_data_seeds"]],
        "shadow_target_pool_split_seed": main_plan["public_shadow_target_pool_split_seed"],
        "attack": "Negative true-label CE; greater score predicts member. Threshold maximizes shadow TPR-FPR only, ties choose greatest threshold. No target optimization or direction flip.",
        "success_descriptive_gate": "Target mean AUC>=.75 and its ordinary5-seed95%t interval lower bound>.5. This is a diagnostic gate fixed before control training, not a DP guarantee.",
        "source_sha256": {"purpose_privacy_attack_control.py": hashfile(Path(__file__)),
                          "_helpers.py": hashfile(Path(core.__file__)), "learning.py":hashfile(source_path("learning.py")), "api.py":hashfile(source_path("api.py"))}}
    save(OUT / "protocol.json", plan)
    frozen_hash = hashfile(OUT / "protocol.json")
    ds = load_digits(); x = (ds.data/16).astype(np.float32); y = ds.target.astype(np.int64)
    shadow_pool, target_pool = train_test_split(np.arange(len(y)), train_size=.4, stratify=y,
        random_state=plan["shadow_target_pool_split_seed"])
    shadow, labels, score = train(plan["private_shadow_seed"], shadow_pool, x, y, "shadow", plan)
    fpr, tpr, thresholds = roc_curve(labels, score, drop_intermediate=False)
    threshold = float(thresholds[int(np.argmax(tpr-fpr))])
    if not math.isfinite(threshold): threshold = float(score.max()+1)
    save(OUT / "threshold_frozen_before_targets.json", {"threshold": threshold,
        "shadow_metrics": shadow, "target_scores_accessed": False})
    results = []
    for index, seed in enumerate(plan["private_target_seeds"]):
        row, labels, score = train(seed, target_pool, x, y, f"target{index}", plan)
        predicted = score >= threshold
        a, b = float(predicted[labels == 1].mean()), float(predicted[labels == 0].mean())
        row.update({"fixed_shadow_threshold": threshold, "TPR": a, "FPR": b,
            "fixed_threshold_advantage": a-b, "balanced_accuracy": .5*(a+1-b)})
        results.append(row)
    aggregate = {}
    for field in ("training_accuracy", "nonmember_accuracy", "member_loss_mean", "nonmember_loss_mean",
                  "membership_AUC", "fixed_threshold_advantage", "balanced_accuracy"):
        values = np.asarray([r[field] for r in results])
        sd = float(values.std(ddof=1)); half = float(t.ppf(.975, 4)*sd/np.sqrt(5))
        aggregate[field] = {"n": 5, "mean": float(values.mean()), "sd": sd,
                            "ci95": [float(values.mean()-half), float(values.mean()+half)]}
    assert hashfile(OUT / "protocol.json") == frozen_hash
    for filename, expected in plan["source_sha256"].items():
        assert hashfile(source_path(filename)) == expected
    positive = aggregate["membership_AUC"]["mean"] >= .75 and aggregate["membership_AUC"]["ci95"][0] > .5
    summary = {"status": "completed", "private_research_only": True,
        "scope": "Overfitting sensitivity control for one loss-based membership probe; no DP claim, no strong-attack claim for main models.",
        "protocol_sha256": frozen_hash, "shadow_runs": 1, "target_runs": 5,
        "Flower_fit_calls": 6, "total_local_epochs": 600,
        "threshold_developed_on_shadow_only": True, "sources_unchanged": True,
        "positive_control_gate_passed": bool(positive), "aggregates": aggregate, "target_runs_details": results,
        "interpretation": "A high positive-control AUC establishes that this score detects the constructed tiny-overfit setting. It does not show the low main no-DP AUC is proof of privacy, nor rule out stronger membership/client attacks.",
        "elapsed_seconds": time.perf_counter()-started}
    save(OUT / "summary.json", summary)
    print(json.dumps({"status": "completed", "positive_control_passed": bool(positive),
        "AUC": aggregate["membership_AUC"], "seconds": summary["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.parse_args()
    main()
