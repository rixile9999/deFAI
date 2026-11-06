from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import csv
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from collections import Counter
from ._paths import ROOT, source_path, relative_path
import numpy as np
from sklearn.datasets import load_digits

OUT = ROOT / "results/purpose_privacy"
PRIVATE = OUT / "research_private"
CHECKS = Counter()
MAX = {}
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[relative_path(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def arrays(path):
    raw = path.read_bytes()
    SOURCES[relative_path(path)] = hashlib.sha256(raw).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def close(a, b, family, tolerance=2e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error = float(np.max(np.abs(a-b)))
    MAX[family] = max(MAX.get(family, 0.), error)
    assert error <= tolerance, (family, error)
    CHECKS[family] += 1


def forward(model, features):
    weights = [model[f"array_{i}"] for i in range(4)]
    hidden = np.maximum(features @ weights[0]+weights[1], np.float32(0))
    logits = hidden @ weights[2]+weights[3]
    shifted = logits-logits.max(axis=1, keepdims=True)
    out = np.exp(shifted)
    return out/out.sum(axis=1, keepdims=True)


def auc_pairwise(labels, scores):
    members, others = scores[labels == 1], scores[labels == 0]
    difference = members[:, None]-others[None, :]
    return float(np.mean((difference > 0)+.5*(difference == 0)))


def main():
    protocol = read(PRIVATE / "protocol.json")
    summary = read(PRIVATE / "summary.json")
    theory = read(PRIVATE / "sensitivity_and_accounting_validation.json")
    thresholds = read(PRIVATE / "frozen_attack_thresholds.json")
    split = read(PRIVATE / "private_split_audit.json")
    assert set(split["shadow_pool_ids"]).isdisjoint(split["target_pool_ids"])
    CHECKS["shadow_target_pool_disjoint"] += 1
    ds = load_digits(); x = (ds.data/16).astype(np.float32); y = ds.target
    expected_shapes = [(64,32), (32,), (32,10), (10,)]
    threshold_by_sigma = {r["sigma"]: r["threshold"] for r in thresholds["thresholds"]}
    main_records = []
    for folder in sorted((PRIVATE / "runs").iterdir()):
        final = read(folder / "private_final.json")
        model = arrays(folder / "model_round020.npz")
        scores = arrays(folder / "private_membership_scores.npz")
        assert list(model) == [f"array_{i}" for i in range(4)]
        assert [model[f"array_{i}"].shape for i in range(4)] == expected_shapes
        ids, labels = scores["source_ids"], scores["membership"]
        assert np.sum(labels == 1) == np.sum(labels == 0) == 200
        assert set(ids[labels == 1]).isdisjoint(ids[labels == 0])
        assert np.array_equal(np.bincount(y[ids[labels == 1]], minlength=10), np.repeat(20,10))
        assert np.array_equal(np.bincount(y[ids[labels == 0]], minlength=10), np.repeat(20,10))
        p = forward(model, x[ids]).astype(np.float64)
        independent = np.log(np.clip(p[np.arange(len(ids)), y[ids]], 1e-30, 1))
        close(independent, scores["score"], "main_saved_model_forward_to_attack_scores", 2e-12)
        auc = auc_pairwise(labels, independent)
        if final["role"] == "target":
            main_records.append(final)
            cohort = read(PRIVATE / f"private_cohort{final['cohort_index']:02d}_ids.json")
            assert set(cohort["member"]).isdisjoint(cohort["nonmember"])
            assert set(cohort["member"]).issubset(split["target_pool_ids"])
            assert sorted(a for client in cohort["client_ids"] for a in client) == sorted(cohort["member"])
            assert set(ids[labels == 1]).issubset(cohort["member"])
            assert set(ids[labels == 0]).issubset(cohort["nonmember"])
            CHECKS["target_membership_source_disjointness"] += 1
        with (folder / "private_clip_rounds.csv").open(encoding="utf-8-sig") as stream:
            rounds = list(csv.DictReader(stream))
        assert len(rounds) == 20
        assert all(float(r["max_postclip_norm"]) <= .2+1e-12 for r in rounds)
        close(np.mean([float(r["clip_fraction"]) for r in rounds]), final["clip_fraction_mean"], "clipping_rate")
        CHECKS["round_clipping_bound"] += 20
        CHECKS["actual_Flower_fit_calls_from_saved_run"] += final["Flower_NumPyClient_fit_calls"]
        if final["sigma"]:
            with localcontext() as ctx:
                ctx.prec = 80
                rho = Decimal(20)/(2*Decimal(str(final["sigma"]))**2)
                epsilon = rho+2*(rho*(Decimal(1)/Decimal("0.00001")).ln()).sqrt()
            close(float(epsilon), final["epsilon"], "high_precision_accountant")
        else:
            assert final["epsilon"] is None and final["PPI_original_1_minus_epsilon"] is None
            CHECKS["no_noise_has_no_DP_claim"] += 1
    with (PRIVATE / "final_metrics.csv").open(encoding="utf-8-sig") as stream:
        final_rows = list(csv.DictReader(stream))
    for row in final_rows:
        scores = arrays(PRIVATE / "runs" / row["run_id"] / "private_membership_scores.npz")
        labels, score = scores["membership"], scores["score"]
        close(auc_pairwise(labels, score), float(row["membership_AUC"]), "pairwise_main_AUC")
        threshold = threshold_by_sigma[float(row["sigma"])]
        predicted = score >= threshold
        tpr, fpr = float(predicted[labels == 1].mean()), float(predicted[labels == 0].mean())
        close([tpr, fpr, tpr-fpr, .5*(tpr+1-fpr)],
              [float(row[key]) for key in ("membership_TPR", "membership_FPR", "membership_fixed_threshold_advantage", "membership_balanced_accuracy")], "fixed_shadow_threshold_attack")
        cohort = read(PRIVATE / f"private_cohort{int(row['cohort_index']):02d}_ids.json")
        model = arrays(PRIVATE / "runs" / row["run_id"] / "model_round020.npz")
        holdout = np.asarray(cohort["nonmember"], int)
        pred = forward(model, x[holdout]).argmax(1)
        table = np.bincount(y[holdout]*10+pred, minlength=100).reshape(10,10)
        tp = np.diag(table).astype(float)
        denom = table.sum(0)+table.sum(1)
        f1 = np.divide(2*tp, denom, out=np.zeros(10), where=denom != 0).mean()
        close([float((pred == y[holdout]).mean()), f1], [float(row["test_accuracy"]), float(row["test_macro_f1"])], "target_utility_forward")
    assert len(main_records) == 20 and len(final_rows) == 20
    assert CHECKS["actual_Flower_fit_calls_from_saved_run"] == summary["Flower_fit_calls"] == 5760
    control = PRIVATE / "attack_sensitivity_control"
    positive = read(control / "summary.json")
    control_protocol = read(control / "protocol.json")
    control_threshold = read(control / "threshold_frozen_before_targets.json")["threshold"]
    control_values = []
    for row in positive["target_runs_details"]:
        scores = arrays(control / f"{row['run_id']}_private_scores.npz")
        model = arrays(control / f"{row['run_id']}_private_model.npz")
        ids, labels = scores["source_ids"], scores["membership"]
        assert np.sum(labels == 1) == np.sum(labels == 0) == 30
        assert np.array_equal(np.bincount(y[ids[labels == 1]], minlength=10), np.repeat(3,10))
        assert np.array_equal(np.bincount(y[ids[labels == 0]], minlength=10), np.repeat(3,10))
        p = forward(model, x[ids]).astype(np.float64)
        score = np.log(np.clip(p[np.arange(len(ids)), y[ids]], 1e-30, 1))
        close(score, scores["score"], "control_model_forward_scores")
        auc = auc_pairwise(labels, score)
        close(auc, row["membership_AUC"], "pairwise_control_AUC")
        predicted = score >= control_threshold
        adv = predicted[labels == 1].mean()-predicted[labels == 0].mean()
        close(adv, row["fixed_threshold_advantage"], "control_fixed_shadow_threshold")
        control_values.append(auc)
    close(np.mean(control_values), positive["aggregates"]["membership_AUC"]["mean"], "control_AUC_mean")
    assert positive["positive_control_gate_passed"]
    candidate = OUT / "release_candidate"
    expected = {"public_config.json"} | {f"model_round{i:03d}.npz" for i in range(1,21)}
    assert {p.name for p in candidate.iterdir()} == expected
    config = read(candidate / "public_config.json")
    assert config["rounds"] == 20 and config["sigma"] == 32
    close(config["epsilon"], theory["release_candidate_accounting"]["epsilon"], "candidate_budget_accountant")
    assert config["epsilon"] <= 1 and config["delta"] == 1e-5
    public_text = json.dumps(config)
    assert all(str(seed) not in public_text for seed in protocol["private_target_data_seeds"]+[protocol["private_shadow_data_seed"]])
    assert summary["fixed_protocol_sha256"] not in public_text
    for path in sorted(candidate.glob("*.npz")):
        private_path = PRIVATE / "runs/target_00_sigma32" / path.name
        assert path.read_bytes() == private_path.read_bytes()
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist()) == {f"array_{i}.npy" for i in range(4)}
            assert all(i.date_time == (1980,1,1,0,0,0) for i in archive.infolist())
        assert int(path.stat().st_mtime) == 946684800
        CHECKS["candidate_exact_model_and_inner_timestamp"] += 1
    with zipfile.ZipFile(OUT / "release_candidate.zip") as archive:
        assert set(archive.namelist()) == expected
        assert all(i.date_time == (2000,1,1,0,0,0) for i in archive.infolist())
        assert all(archive.read(p.name) == p.read_bytes() for p in candidate.iterdir())
    CHECKS["candidate_zip_allowlist_bytes_timestamps"] += 1
    for filename, expected_hash in protocol["source_sha256"].items():
        assert hashlib.sha256((source_path(filename)).read_bytes()).hexdigest() == expected_hash
    for filename, expected_hash in control_protocol["source_sha256"].items():
        assert hashlib.sha256((source_path(filename)).read_bytes()).hexdigest() == expected_hash
    assert hashlib.sha256((PRIVATE / "protocol.json").read_bytes()).hexdigest() == summary["fixed_protocol_sha256"]
    assert hashlib.sha256((control / "protocol.json").read_bytes()).hexdigest() == positive["protocol_sha256"]
    for key, expected_hash in SOURCES.items():
        assert hashlib.sha256((ROOT / key).read_bytes()).hexdigest() == expected_hash
    result = {"status": "passed", "private_research_only": True,
        "no_retraining_or_new_noise": True, "checks": dict(CHECKS), "max_absolute_errors": MAX,
        "independent_forward_main_models": 24, "independent_target_utility_models": 20,
        "independent_positive_control_models": 5, "public_models_exact_copies": 20,
        "sources_and_protocols_unchanged": True, "source_hashes": SOURCES,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitation": "Numerical correctness and release-boundary audit; not a finite-precision DP or adversarial-system security certificate."}
    (PRIVATE / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": dict(CHECKS), "no_retraining": True}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.parse_args()
    main()
