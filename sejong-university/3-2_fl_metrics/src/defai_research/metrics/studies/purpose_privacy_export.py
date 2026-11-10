from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

from ._paths import ROOT, source_path, relative_path
OUT = ROOT / "results/purpose_privacy"
PRIVATE = OUT / "research_private"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary = read(PRIVATE / "summary.json")
    protocol = read(PRIVATE / "protocol.json")
    positive = read(PRIVATE / "attack_sensitivity_control/summary.json")
    allowlist = read(PRIVATE / "release_allowlist_audit.json")
    sensitivity = read(PRIVATE / "sensitivity_and_accounting_validation.json")
    verification = read(PRIVATE / "independent_verification.json")
    assert summary["status"] == positive["status"] == "completed"
    assert allowlist["status"] == sensitivity["status"] == "passed"
    assert verification["status"] == "passed" and verification["no_retraining_or_new_noise"]
    assert hashlib.sha256((PRIVATE / "protocol.json").read_bytes()).hexdigest() == summary["fixed_protocol_sha256"]
    for filename, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((source_path(filename)).read_bytes()).hexdigest() == expected
    candidate = summary["prespecified_single_release_candidate_private_diagnostics"]
    rows = []
    for item in summary["summary_by_sigma"]:
        row = {"sigma": item["sigma"], "n": item["n_independent_target_cohorts"],
            "epsilon_20round_transcript": item["epsilon"], "delta": item["delta"],
            "original_PPI_1_minus_epsilon": item["PPI_original_1_minus_epsilon"]}
        for name in ("test_accuracy", "test_macro_f1", "clip_fraction_mean", "membership_AUC", "membership_fixed_threshold_advantage", "membership_balanced_accuracy"):
            row[name+"_mean"] = item[name]["mean"]
            row[name+"_sd"] = item[name]["sd"]
        rows.append(row)
    report = {
        "metric_ids": [10], "metric_name": "프라이버시 보존 지표의 목적 직접 검증",
        "scope": "PRIVATE research diagnostics. This JSON, its figures and the research report are NOT part of the DP-protected release candidate.",
        "original_formula": "PPI=1-mean(epsilon_i), retained as a non-probability contrast; not used to rank privacy.",
        "replacement_definition": "(epsilon,delta), replace-one-fixed-client adjacency, full output-transcript scope, accountant and public-artifact allowlist; smaller epsilon at fixed delta/adjacency gives a stronger stated bound.",
        "purpose_verdict": "지정한 공개 출력의 이론 보호예산·노출 경계와 보호-효용 교환관계를 직접 검증함. 실용 DP 모델 효용은 미달이며 구현 인증/하드웨어 보장은 별도임.",
        "actual_Flower_training": {"target_runs": 20, "shadow_runs": 4, "rounds_each": 20,
            "client_slots": 12, "Flower_NumPyClient_fit_calls": summary["Flower_fit_calls"],
            "local_epochs": 2, "clip_norm_C": .2, "sensitivity_2C_over_K": 2*.2/12,
            "sigma_grid": [0, 2, 8, 32], "public_preprocessing": "Digits pixel/16", "fixed_final_round": True},
        "summary_by_sigma": summary["summary_by_sigma"], "table_rows": rows,
        "prespecified_release_candidate": {"choice_before_training": "cohort0/sigma32; fixed20-round transcript, no utility/attack-dependent choice",
            "epsilon": candidate["epsilon"], "delta": candidate["delta"], "epsilon_budget": 1.,
            "epsilon_budget_met": candidate["epsilon"] <= 1., "rho": candidate["rho"],
            "noisy_models": 20, "public_constant_config_files": 1,
            "allowlist_passed": True, "timestamp_normalization_checked": True,
            "raw_updates_graph_hash_filter_private_diagnostics_released": False,
            "private_utility_accuracy": candidate["test_accuracy"],
            "private_utility_macro_F1": candidate["test_macro_f1"],
            "private_membership_AUC": candidate["membership_AUC"],
            "private_fixed_threshold_advantage": candidate["membership_fixed_threshold_advantage"],
            "deployment_decision": "보호예산은 충족하나 효용이 낮아 실용 모델로 채택하지 않음. 외부 배포/업로드 없음.",
            "protected_candidate_folder": "results/purpose_privacy/release_candidate",
            "protected_candidate_zip": "results/purpose_privacy/release_candidate.zip"},
        "membership_attack": {"score": "negative true-label cross-entropy; larger score predicts member",
            "threshold_development": "Shadow-record pool disjoint from all target records; threshold fixed separately per sigma before target training; no target optimization.",
            "target_members_nonmembers_per_seed": [200, 200], "class_matched": True,
            "advantage": "TPR-FPR at the shadow-frozen threshold, not target-maximized advantage",
            "no_noise_AUC_mean": rows[0]["membership_AUC_mean"],
            "interpretation": "Main no-DP AUC is already near0.5. Low main attack AUC/advantage does not prove privacy or exhaust stronger attacks. This is a record-membership probe, not client-participation inference."},
        "attack_sensitivity_positive_control": {"protocol_fixed_before_training": True,
            "records_per_class_train_nonmember": [3, 3], "local_epochs": 100, "batch_size": 10,
            "learning_rate": .14, "Flower_fit_calls": 6, "target_seeds": 5, "shadow_seeds": 1,
            "threshold_shadow_only": True, "positive_control_gate_passed": positive["positive_control_gate_passed"],
            "aggregates": positive["aggregates"],
            "interpretation": "동일한 손실 공격이 구성한 과적합 대조를 검출함을 확인한 것이며, 주 실험의 강한 공격 저항을 입증하지 않음."},
        "hypothetical_composition_stress_not_released": sensitivity["hypothetical_not_released"],
        "independent_verification": {key: verification[key] for key in (
            "status", "no_retraining_or_new_noise", "checks", "max_absolute_errors",
            "independent_forward_main_models", "independent_target_utility_models",
            "independent_positive_control_models", "public_models_exact_copies", "sources_and_protocols_unchanged", "limitation")},
        "critical_scope_limits": [
            "Only21 allowlisted files in the prespecified candidate are protected-output candidates. No-noise models, other sigma/seed models, shadow models and all research diagnostics remain private.",
            "Existing non-DP graph/hash/filter mechanisms do not acquire DP from this separate mechanism. No graph/filter/hash is included in the public candidate.",
            "The accountant is an ideal Gaussian full20-round bound under a logical trusted central process. No certified finite-precision DP service, hardware enclave or malicious-server/side-channel defense is claimed.",
            "Private clipping/attack/utility logs and this report are excluded from the protected transcript; their future publication would need separate analysis.",
            "The strong formal budget is accompanied by severe empirical utility loss. Attack failure and protection-utility tradeoffs are reported, not tuned away.",
        ],
        "source_reference_note": {"corrected_membership_reference": "https://arxiv.org/abs/1709.01604",
            "correct_primary_membership_reference": "https://arxiv.org/abs/1709.01604",
            "author_pdf": "https://www.cs.cmu.edu/~mfredrik/papers/YeomCSF18.pdf",
            "reason": "This public source copy corrects a bibliographic URL from the original local study; no training or accounting change."},
        "primary_sources": ["https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf",
            "https://arxiv.org/abs/1605.02065", "https://arxiv.org/abs/1709.01604",
            "https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py",
            "https://arxiv.org/abs/2107.10138"],
        "sources": [{"path": relative_path(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [PRIVATE/"protocol.json", PRIVATE/"summary.json", PRIVATE/"release_allowlist_audit.json",
                         PRIVATE/"sensitivity_and_accounting_validation.json", PRIVATE/"attack_sensitivity_control/protocol.json",
                         PRIVATE/"attack_sensitivity_control/summary.json", PRIVATE/"independent_verification.json", Path(__file__)]],
    }
    (OUT / "report_export.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (OUT / "report_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"report_export": str(OUT / "report_export.json"), "candidate_epsilon": candidate["epsilon"],
                      "positive_control_AUC": positive["aggregates"]["membership_AUC"]["mean"]}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.parse_args()
    main()
