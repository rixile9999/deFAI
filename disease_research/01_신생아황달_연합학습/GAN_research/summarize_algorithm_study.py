"""Report all validation candidates and locked federated-optimizer comparisons."""
from hashlib import sha256
from html import escape
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from run_algorithm_study import ROOT, OUTPUT, CONTEXTS
from summarize_skin_bias import error_metrics, table

LABELS = {"fedavg_adam": "연합 평균·Adam", "fedprox": "FedProx·Adam", "scaffold": "SCAFFOLD·경사하강",
          "fedbn": "FedBN·기관별 정규화", "fedavg_sgd": "연합 평균·경사하강(검증 선택)",
          "fedavg_sgd_matched": "연합 평균·SCAFFOLD 동일 학습률", "fedavg_trainbn": "연합 평균·배치 통계 학습"}
PRIMARY = ["fedavg_adam", "fedprox", "scaffold", "fedbn"]


def context_label(condition, bias):
    return ("원본" if condition == "real" else "촬영쌍 생성 100%")+("＋명도 억제" if bias else "")


def main():
    complete = json.loads((OUTPUT / "complete.json").read_text())
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    selection = json.loads((OUTPUT / "selection.json").read_text())
    runs = json.loads((OUTPUT / "training_summary.json").read_text())
    assert len(runs) == complete["trainings"]
    assert sha256((ROOT / "data/real_data.npz").read_bytes()).hexdigest() == protocol["data_sha256"]
    summaries, subgroups, arrays = [], [], {}
    rows = []; order = []
    for selected in selection:
        condition, bias, family = selected["condition"], selected["bias"], selected["family"]
        frames, shifts, histories = [], [], []
        for seed in [17, 29, 43]:
            name = f"{condition}_bias{int(bias)}_{selected['config']['id']}_seed{seed}"
            frame = pd.read_csv(OUTPUT / f"{name}_gain1.00_test.csv", dtype={"patient": str}).sort_values("patient")
            assert len(frame) == 112 and frame.patient.is_unique
            frames.append(frame)
            shifted = [pd.read_csv(OUTPUT / f"{name}_gain{gain:.2f}_test.csv", dtype={"patient": str}).sort_values("patient") for gain in [.55, .75, .9]]
            for f in shifted:
                np.testing.assert_array_equal(f.patient, frame.patient)
                np.testing.assert_array_equal(f.blood, frame.blood)
                np.testing.assert_array_equal(f.treatment, frame.treatment)
            shifts.append(shifted)
            history = next(r for r in runs if r["name"] == name)
            histories.append(history)
            rows.append({"condition": condition, "bias": bias, "family": family, "seed": seed,
                         "config": selected["config"]["id"], "validation_mae": history["validation_mae"],
                         "best_round": history["best_round"], "test_mae_mgdl": float(abs(frame.prediction-frame.blood).mean())})
        if len(order):
            np.testing.assert_array_equal(frame.patient, order)
        order = frame.patient.to_numpy()
        y = frame.blood.to_numpy(); g = frame.group.to_numpy(); treatment = frame.treatment.to_numpy()
        predictions = np.stack([f.prediction.to_numpy() for f in frames])
        probabilities = np.stack([f.probability.to_numpy() for f in frames])
        shifted_predictions = np.stack([[f.prediction.to_numpy() for f in changed] for changed in shifts])
        errors = abs(predictions-y).mean(0)
        sensitivity = abs(shifted_predictions-predictions[:, None]).mean((0, 1))
        stats = error_metrics(errors, g)
        key = (condition, bias, family)
        arrays[key] = {"errors": errors, "sensitivity": sensitivity,
                       "classification_error": ((probabilities >= .5) != treatment).mean(0)}
        summary = {"condition": condition, "bias": bias, "family": family, "label": LABELS[family],
                   "selected_config": selected["config"], "validation_mae_mean": float(np.mean([h["validation_mae"] for h in histories])),
                   "mae_mgdl": float(stats[0]), "mae_seed_sd": float(abs(predictions-y).mean(1).std(ddof=1)),
                   "worst_group_mae_mgdl": float(stats[1]), "group_gap_mgdl": float(stats[2]),
                   "stress_prediction_shift_mgdl": float(sensitivity.mean()),
                   "stress_mae_mgdl": float(abs(shifted_predictions-y).mean()),
                   "accuracy": float(((probabilities >= .5) == treatment).mean()),
                   "f1": float(np.mean([f1_score(treatment, p >= .5, zero_division=0) for p in probabilities])),
                   "roc_auc": float(np.mean([roc_auc_score(treatment, p) for p in probabilities])),
                   "mean_seconds": float(np.mean([h["seconds"] for h in histories]))}
        summaries.append(summary)
        for group in range(3):
            chosen = g == group; positive = chosen & (treatment == 1); negative = chosen & (treatment == 0)
            subgroups.append({"condition": condition, "bias": bias, "family": family, "group": group,
                             "n_patients": int(chosen.sum()), "mae_mgdl": float(errors[chosen].mean()),
                             "signed_error_mgdl": float((predictions[:, chosen]-y[chosen]).mean()),
                             "false_negative_rate": float((probabilities[:, positive] < .5).mean()) if positive.any() else None,
                             "false_positive_rate": float((probabilities[:, negative] >= .5).mean()) if negative.any() else None})
    rng = np.random.default_rng(20261017)
    draws = np.concatenate([rng.choice(np.flatnonzero(g == k), (5000, int((g == k).sum()))) for k in range(3)], axis=1)
    boot_g = g[draws[0]]
    contrasts = []
    for summary in summaries:
        condition, bias, family = [summary[k] for k in ["condition", "bias", "family"]]
        if family == "fedavg_adam":
            continue
        bases = ["fedavg_adam"]
        if family == "scaffold":
            match = next(s for s in selection if s["condition"] == condition and s["bias"] == bias and s["family"] in ["fedavg_sgd", "fedavg_sgd_matched"] and s["config"]["lr"] == summary["selected_config"]["lr"])
            bases.append(match["family"])
        if family == "fedbn":
            bases.append("fedavg_trainbn")
        current = arrays[(condition, bias, family)]
        for base in bases:
            reference = arrays[(condition, bias, base)]
            point = np.r_[error_metrics(current["errors"], g)-error_metrics(reference["errors"], g),
                          (current["sensitivity"]-reference["sensitivity"]).mean(),
                          (reference["classification_error"]-current["classification_error"]).mean()]
            bootstrap = np.column_stack([
                error_metrics(current["errors"][draws], boot_g)-error_metrics(reference["errors"][draws], boot_g),
                (current["sensitivity"]-reference["sensitivity"])[draws].mean(1),
                (reference["classification_error"]-current["classification_error"])[draws].mean(1)])
            contrasts.append({"condition": condition, "bias": bias, "family": family, "baseline": base,
                "differences": {metric: {"difference": float(point[j]), "ci95": np.quantile(bootstrap[:, j], [.025, .975]).tolist()}
                                for j, metric in enumerate(["mae_mgdl", "worst_group_mae_mgdl", "group_gap_mgdl", "stress_prediction_shift_mgdl", "accuracy_fraction"])}})
    winners = []
    for condition, bias in CONTEXTS:
        candidates = [s for s in summaries if s["condition"] == condition and s["bias"] == bias and s["family"] in PRIMARY]
        winners.append(min(candidates, key=lambda s: s["validation_mae_mean"]))
    overall_winner = min(winners, key=lambda s: s["validation_mae_mean"])
    locked = json.loads((OUTPUT / "validation_winners_locked.json").read_text())
    assert all(overall_winner[k] == locked["overall"][k] for k in ["condition", "bias", "family"])
    for winner, locked_winner in zip(winners, locked["per_context"]):
        assert all(winner[k] == locked_winner[k] for k in ["condition", "bias", "family"])
    tuning = [{"condition": r["condition"], "bias": r["bias"], "config": r["config"]["id"],
               "best_validation_mae": r["validation_mae"], "best_round": r["best_round"]} for r in runs if r["seed"] == 17]
    pd.DataFrame(summaries).to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(subgroups).to_csv(OUTPUT / "subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUTPUT / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tuning).to_csv(OUTPUT / "all_tuning_candidates.csv", index=False, encoding="utf-8-sig")
    report = {"stage": "research_only", "protocol": protocol, "complete": complete, "summary": summaries,
              "subgroups": subgroups, "contrasts": contrasts, "validation_selected_per_context": winners,
              "validation_selected_overall": overall_winner, "tuning": tuning,
              "uncertainty": "5000 paired patient bootstrap stratified by observed photo brightness; three seed errors averaged before resampling; fixed split, no multiplicity correction",
              "scaffold_control_invariant_max_error": max(r["scaffold_mean_control_max_error"] for r in runs)}
    if (OUTPUT / "parallel_execution.json").exists():
        report["execution"] = json.loads((OUTPUT / "parallel_execution.json").read_text())
    previous = json.loads((ROOT / "results/skin_bias/summary.json").read_text(encoding="utf-8"))
    baseline_recalculation = []
    for condition, bias in CONTEXTS:
        old = next(s for s in previous["summary"] if s["condition"] == condition and s["method"] == ("gain_consistency" if bias else "baseline"))
        current = next(s for s in summaries if s["condition"] == condition and s["bias"] == bias and s["family"] == "fedavg_adam")
        baseline_recalculation.append({"condition": condition, "bias": bias, "previous_mae": old["mae_mgdl"], "recalculated_mae": current["mae_mgdl"]})
    report["baseline_recalculation"] = baseline_recalculation
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (condition, bias) in zip(axes.ravel(), CONTEXTS):
        chosen = [next(s for s in summaries if s["condition"] == condition and s["bias"] == bias and s["family"] == f) for f in PRIMARY]
        ax.bar(PRIMARY, [s["mae_mgdl"] for s in chosen], yerr=[s["mae_seed_sd"] for s in chosen], capsize=4)
        ax.set_ylabel("Patient-level MAE (mg/dL)"); ax.set_title(f"{condition}; brightness consistency={bias}")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUTPUT / "algorithm_comparison.png", dpi=160); plt.close(fig)
    scores = table(["자료·명도 억제", "방법", "검증 오차", "시험 오차 ± 초기값 표준편차", "치료 정확도", "집단 격차", "명도 민감도"], [
        [context_label(s["condition"], s["bias"]), s["label"], f"{s['validation_mae_mean']:.4f}",
         f"{s['mae_mgdl']:.4f} ± {s['mae_seed_sd']:.4f}", f"{100*s['accuracy']:.2f}%", f"{s['group_gap_mgdl']:.4f}", f"{s['stress_prediction_shift_mgdl']:.4f}"] for s in summaries])
    candidates = table(["조건", "모든 설정(초기값 17)", "최소 검증 평균절대오차", "선택 라운드"], [
        [context_label(t["condition"], t["bias"]), t["config"], f"{t['best_validation_mae']:.4f}", t["best_round"]] for t in tuning])
    baseline_table = table(["조건", "앞 절의 기존 평균 오차", "동일 알고리즘의 이번 재학습 오차"], [
        [context_label(s["condition"], s["bias"]), f"{s['previous_mae']:.4f}", f"{s['recalculated_mae']:.4f}"] for s in baseline_recalculation])
    winner_table = table(["조건", "검증으로 선택한 방법", "평균 검증 오차", "시험 오차", "치료 정확도"], [
        [context_label(s["condition"], s["bias"]), s["label"], f"{s['validation_mae_mean']:.4f}", f"{s['mae_mgdl']:.4f}", f"{100*s['accuracy']:.2f}%"] for s in winners])
    comparison_table = table(["조건", "방법 − 대조군", "시험 오차 차이", "95% 재표집 구간", "집단 격차 차이"], [
        [context_label(c["condition"], c["bias"]), LABELS[c["family"]]+" − "+LABELS[c["baseline"]],
         f"{c['differences']['mae_mgdl']['difference']:+.4f}",
         "["+", ".join(f"{v:+.4f}" for v in c["differences"]["mae_mgdl"]["ci95"])+"]",
         f"{c['differences']['group_gap_mgdl']['difference']:+.4f}"] for c in contrasts if c["family"] in ["fedprox", "scaffold", "fedbn"]])
    explanation = []
    for s in winners:
        c = next((c for c in contrasts if c["condition"] == s["condition"] and c["bias"] == s["bias"] and c["family"] == s["family"] and c["baseline"] == "fedavg_adam"), None)
        text = "연합 평균 자체가 검증 기준 최적 후보로 남았다." if c is None else (
            f"같은 조건의 연합 평균·Adam 대비 시험 오차 차이 {c['differences']['mae_mgdl']['difference']:+.4f} mg/dL; "
            + ("95% 구간 전체가 음수지만 고정 분할의 탐색 결과다." if c["differences"]["mae_mgdl"]["ci95"][1] < 0 else "95% 구간이 0을 포함하거나 양수이므로 시험 오차 개선이 입증됐다고 표현하지 않는다."))
        explanation.append(f"<li>{context_label(s['condition'], s['bias'])}: {s['label']}. {text}</li>")
    chosen_context = [s for s in summaries if s["condition"] == overall_winner["condition"] and s["bias"] == overall_winner["bias"]]
    by_family = {s["family"]: s for s in chosen_context}
    interpretation = []
    for family, base in [("fedprox", "fedavg_adam"), ("scaffold", "fedavg_sgd"), ("fedbn", "fedavg_adam")]:
        if family == "scaffold" and by_family[base]["selected_config"]["lr"] != by_family[family]["selected_config"]["lr"]:
            base = "fedavg_sgd_matched"
        c = next(c for c in contrasts if c["condition"] == overall_winner["condition"] and c["bias"] == overall_winner["bias"] and c["family"] == family and c["baseline"] == base)
        delta = c["differences"]["mae_mgdl"]
        conclusion = "오차 감소 구간이 관찰됐다(고정 분할·탐색 결과)." if delta["ci95"][1] < 0 else "오차 증가 구간이 관찰됐다(고정 분할·탐색 결과)." if delta["ci95"][0] > 0 else "구간에 0이 포함돼 명확한 오차 감소를 확인하지 못했다."
        interpretation.append(f"<li>{LABELS[family]}: 시험 오차 {by_family[family]['mae_mgdl']:.4f}, "
            f"{LABELS[base]} {by_family[base]['mae_mgdl']:.4f} mg/dL 대비 차이 {delta['difference']:+.4f}, "
            f"95% 구간 [{delta['ci95'][0]:+.4f}, {delta['ci95'][1]:+.4f}]. {conclusion}</li>")
    all_validation_best = min(summaries, key=lambda s: s["validation_mae_mean"])
    report["all_families_validation_minimum_descriptive"] = all_validation_best
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    section = f'''<h2 id="federated-optimization">9. FedProx·SCAFFOLD·FedBN 적용과 검증 기준 최적화</h2>
<p>2026-10-17 추가 · RTX 4080 · <b>연구 단계·실전 미적용</b>.
연합 평균(FedAvg, Federated Averaging)을 대조군으로 세 방법을 직접 구현·학습했다.
세 가상 기관의 기존 환자 분할, 학습 519명·검증 111명·시험 112명, 실제 학습 환자 수 {protocol['client_weights']} 집계 가중치를 유지했다.
새로운 기관 이질성을 인위적으로 만들지 않고 현재의 치료 라벨 층화 기관 배정을 그대로 평가했다.</p>
<p>착수 시 기관별 학습 환자가 같다는 가정을 검사했으나 실제 수는 174·173·172명으로 확인돼,
SCAFFOLD 실행 전 검사가 중단시켰다. 이후 서버 보정 상태도 실제 환자 가중 목적함수에 맞게 수정했다.
기존 연합 평균·FedProx·경사하강 대조는 이미 실제 환자 수로 집계하고 있었으므로 완료된 해당 실행은 유지했다.
초기 잘못된 가정의 계획과 중단 로그도 보존했다. 자료 분할이나 라벨을 바꾼 수정은 아니다.</p>
<h3>무엇을 바꾸었고 무엇을 통제했나</h3>
<ul><li><b>FedProx(Federated Proximal)</b>: 지역 목적함수에 μ/2 × Σ(지역 파라미터−해당 라운드 공통 파라미터)² 추가.
파라미터 개수로 나누지 않는다. 기존 Adam 학습률 0.0001을 유지하고 μ=0.01/0.1/1.0을 비교했다.
원논문의 모든 실험 재현이 아니라 근접 벌점을 같은 영상 예측기에 적용한 것이다.</li>
<li><b>SCAFFOLD(Stochastic Controlled Averaging for Federated Learning)</b>: 지역 경사에 c−cᵢ를 더하고,
통신 후 cᵢ ← cᵢ−c+(라운드 시작 모델−지역 최종 모델)/(실제 지역 갱신 수×학습률)로 갱신하는 두 번째 방식을 구현했다.
모든 기관이 참여하며 서버 c는 실제 환자 수 {protocol['client_weights']}로 가중한 기관 cᵢ의 평균이다.
균등 기관 목적함수를 그대로 쓰지 않고 실제 환자 가중 목적함수로 확장했다.
Adam에 이 식을 그대로 섞지 않고 운동량 없는 확률적 경사하강(SGD, Stochastic Gradient Descent)을 사용했다.
학습률 0.001/0.01/0.1을 비교하고 같은 학습률의 연합 평균 대조도 확보했다.
종전 안정화와 맞추기 위해 손실 경사를 길이 1로 제한한 후 보정항을 더하는 변형이며, 원논문의 무제한 경사 이론을 검증한 결과가 아니다.</li>
<li><b>FedBN(Federated Batch Normalization)</b>: 모든 배치 정규화 계수·실행 평균·분산·정수 계수기를 기관별로 보유하고,
그 외 파라미터만 실제 환자 수로 집계했다. 종전 모델의 배치 통계 고정 기능을 이 실험의 지역 학습에서만 해제했다.
메타정보의 LayerNorm은 공유한다. 시험 때는 해당 환자가 배정된 기관의 모델을 사용하고 시험 자료로 통계를 갱신하지 않았다.
이는 알려진 기관에 대한 개인화 평가이며 새 병원에서 바로 작동한다는 근거가 아니다.</li>
<li><b>배치 통계 학습 대조</b>: FedBN과 같은 지역 배치 정규화 학습을 하되 모델 상태를 전역 평균.
이때 실행 평균·분산의 집계는 일반적인 모델 상태 평균 휴리스틱이며 통합 환자 모집단의 평균·분산 추정량이 아니다.
정수 버퍼는 기관 간 일치 여부를 검사하며 임의 실수 평균을 하지 않는다.</li></ul>
<p>자료 조건 원본/촬영쌍 생성 100% × 명도 증강·일관성 손실 미적용/적용의 네 조합을 모두 비교했다.
명도 억제는 앞 절의 계수·범위를 그대로 사용했다. 생성기를 다시 학습하거나 다른 색상 변환을 추가하지 않았다.
모든 실행은 20회 통신·라운드마다 기관별 실제 영상 1회 통과·배치 32, 같은 초기값에서 시작한다.
생성 영상은 같은 갱신에 추가하고 실제 환자 수를 늘려 세지 않는다.
FedBN을 포함한 모든 최적화기는 라운드마다 재생성하며 Adam의 운동량을 다음 라운드로 넘기지 않는다.
SCAFFOLD의 보정 상태와 FedBN의 지역 정규화 상태는 라운드 사이 유지한다.</p>
<h3>시험 자료를 열기 전에 설정을 선택했다</h3>
<p>초기값 17에서 네 조건별 모든 12설정을 검증 평균절대오차로 비교했다.
처음 계획은 9설정이었으나 작은 학습률의 검증 수렴이 느려 시험 평가 전에 학습률 0.1·근접 강도 1.0을 추가했다.
각 방법의 설정을 고정한 뒤 초기값 29·43을 반복했다. SCAFFOLD와 최적으로 선택된 경사하강 대조의 학습률이 다르면
SCAFFOLD와 정확히 같은 학습률의 대조도 반복했다. 모든 학습이 끝난 뒤에만 시험 예측을 계산했다.
총 학습 {complete['trainings']}개, 선택 모델 {complete['selected_models']}개.
각 실행의 체크포인트는 검증 오차 최소 라운드로 선택했다.
설정 조정에 초기값 17만 썼으므로 초기값에 따른 설정 선택 불안정성이 남고, 동일 검증 집합 재사용에 의한 낙관 편향도 가능하다.</p>
<p>Adam 계열의 학습률 0.0001과 배치 정규화의 기본 이동 계수 0.1은 고정했다.
FedProx·SCAFFOLD는 추가 설정 탐색을 했으므로 모든 방법의 탐색 계산 예산을 같게 맞춘 비교는 아니다.
전체 학습 횟수와 탈락 설정을 함께 공개하며 검증 최적 후보를 보편적인 최적 알고리즘이라고 부르지 않는다.</p>
{candidates}<p>이 표는 탈락한 설정을 포함한 검증 결과 전체다. 시험 결과를 근거로 숨긴 설정은 없다.
약어 Adam은 적응적 모멘트 추정 최적화기이고 SGD는 확률적 경사하강법이다. 이후 오차 단위는 모두 mg/dL이다.</p>
{baseline_table}<p>이전 계산은 GPU 전체 영상 캐시·배치 128 평가·합성곱 자동 탐색을 사용했다.
이번에는 CPU 영상 캐시·배치 64 평가·합성곱 자동 탐색 비활성화로 모든 대조군도 재학습했다.
수학적 학습 목적·초기값·자료 통과 수는 같아도 부동소수점 연산 경로에 따라 학습 궤적과 선택 체크포인트가 달라질 수 있다.
따라서 앞 절 결과는 그대로 보존하며, 추가 알고리즘의 효과는 이번 동일 실행 조건의 대조군과 비교한다.</p>
<h3>선택 설정의 전체 시험 결과</h3>{scores}
<p>시험 오차는 같은 환자의 영상 예측을 먼저 평균한 뒤 환자별 절대오차를 계산하고, 세 초기값의 오차를 평균했다.
± 값은 초기값별 전체 평균절대오차의 표준편차이며 신뢰구간이 아니다.
집단 격차는 사진 명도 세 집단 중 최대·최소 평균절대오차 차이다. 본래 피부색 집단이 아니다.
명도 민감도는 같은 사진을 0.55/0.75/0.90배로 어둡게 했을 때 원본 대비 예측 변화 절댓값 평균이다.
치료 확률 기준은 0.5로 고정했다. 민감도·오차는 작을수록, 정확도는 클수록 좋다.
치료 분류 성능은 보유 Treatment 라벨과의 일치도이며 진료 기준의 임상적 검증이 아니다.
원본 사진만으로 학습한 방법도 같은 시험 명도 변환을 평가했다.</p>
<img src="algorithm_study/algorithm_comparison.png" alt="네 자료·명도 억제 조합에서 연합 최적화 방법별 환자 오차 비교">
<h3>검증으로 선택한 최적 후보와 비교 불확실성</h3>{winner_table}<ul>{''.join(explanation)}</ul>
<p>조건별 후보와 전체 후보는 연합 평균·FedProx·SCAFFOLD·FedBN 네 방법의 초기값 세 개 평균 검증 오차로 선택했다.
시험 성능으로 순서를 바꾸지 않았다. 전체 검증 기준 최적 후보는 <b>{context_label(overall_winner['condition'], overall_winner['bias'])}
· {overall_winner['label']}</b>이며 검증 오차 {overall_winner['validation_mae_mean']:.4f}, 시험 오차 {overall_winner['mae_mgdl']:.4f} mg/dL,
치료 정확도 {100*overall_winner['accuracy']:.2f}%다. 이 선택은 빌리루빈 평균절대오차 기준이며 치료 분류·피부색 공정성의 동시 최적화를 뜻하지 않는다.</p>
<p><b>검증 선택과 시험 개선을 구분해야 한다.</b> 이 후보는 시험에서 같은 조건의 연합 평균·Adam
{by_family['fedavg_adam']['mae_mgdl']:.4f} mg/dL보다 오차가 컸다.
치료 정확도도 {100*by_family['fedavg_adam']['accuracy']:.2f}% → {100*overall_winner['accuracy']:.2f}%
({100*(overall_winner['accuracy']-by_family['fedavg_adam']['accuracy']):+.2f}%p)였고,
명도 변화 민감도는 {by_family['fedavg_adam']['stress_prediction_shift_mgdl']:.4f} →
{overall_winner['stress_prediction_shift_mgdl']:.4f} mg/dL였다.
검증에서 선택됐다는 사실을 예측·편향 억제 개선의 입증으로 바꾸지 않는다.</p>
<p>참고로 정규화 학습 대조까지 포함한 표 전체의 검증 최저값은
{context_label(all_validation_best['condition'], all_validation_best['bias'])} · {all_validation_best['label']}
{all_validation_best['validation_mae_mean']:.4f} mg/dL다. 이는 요청한 네 주방법을 대상으로 시험 전에 고정한 선택과 구분한 기술적 비교다.</p>
<h3>검증 선택 조건에서 방법별 기여 해석</h3><ul>{''.join(interpretation)}</ul>
<p>SCAFFOLD는 같은 학습률 경사하강 대조와 비교해 보정항의 기여를 보지만, 기존 Adam 연합 평균보다 낫다는 뜻은 아니다.
FedBN의 배치 통계 학습 대조와의 차이도 아래에 제시한다. 유리한 대조군 하나만 골라 전체 개선을 주장하지 않는다.</p>
{comparison_table}<p>차이는 해당 방법−같은 자료·명도 조건의 대조군이다. 음수는 오차 또는 격차 감소다.
세 초기값의 환자별 오차를 먼저 평균한 뒤 사진 명도 집단별 표본 수를 유지하며 112명을 5,000번 짝지어 재표집했다.
시험 분할에 조건부이고 설정 선택·새 병원·다른 환자 분할의 불확실성을 포함하지 않는다. 다중 비교 보정을 하지 않은 탐색 결과다.
SCAFFOLD는 같은 학습률의 경사하강 대조와, FedBN은 배치 통계 학습 대조와도 비교해 구성 요소의 기여를 구분한다.
명도 격차·명도 민감도·정확도 차이의 구간 및 집단별 과소예측·치료 누락·오탐률은 summary.json과 subgroup_metrics.csv에도 모두 남겼다.</p>
{table(['한 것', '안 한 것'], [['세 알고리즘 구현·수식 테스트·네 자료/명도 조건·세 초기값', '실제 병원 간 배포·추가 통신 성능 시험·새 병원 개인화'], ['검증 전용 설정 선택·탈락 설정 공개·짝지은 환자 재표집', '중첩 검증·다른 환자/기관 분할·새로운 강한 기관 이질성'], ['학습률 일치 경사하강 대조·배치 통계 학습 대조', '세 알고리즘의 혼합·부분 기관 참여·다른 지역 학습 횟수'], ['명도 집단 오차·변환 민감도·치료 성능 동시 기록', '본래 피부색 측정에 따른 공정성 및 임상적 유효성 검증']])}
<p>이 데이터·이 방식·이 설정 범위 한정의 결과이며 새로운 방법이 항상 기존 방법보다 낫다고 해석하지 않는다.
한 GPU의 순차 모의실험이고 현재 가상 기관 배정은 실제 병원 간 장비·피부색 분포 차이를 대표하지 않는다.
각 연합학습 안에서 기관은 순차 실행하되 GPU 여유가 생긴 후 서로 독립적인 네 조건 실험은 동시 계산했다.
다른 GPU 작업과 공유된 실행 시간이 포함될 수 있어 기록된 실행 시간을 엄밀한 속도 벤치마크로 사용하지 않는다.</p>
<p>원논문·구현 근거: <a href="https://arxiv.org/abs/1812.06127">FedProx</a>,
<a href="https://proceedings.mlr.press/v119/karimireddy20a.html">SCAFFOLD</a>,
<a href="https://github.com/med-air/FedBN">FedBN 저자 구현</a>.
산출물은 algorithm_study/protocol.json, selection.json, all_tuning_candidates.csv, per_seed_metrics.csv,
summary.csv, summary.json, subgroup_metrics.csv, 개별 검증 이력·가중치·시험 예측이다.</p>'''
    (OUTPUT / "report_section.html").write_text(section, encoding="utf-8")
    print(pd.DataFrame(summaries).drop(columns="selected_config").to_string(index=False))
    print("VALIDATION SELECTED OVERALL", overall_winner)


if __name__ == "__main__":
    main()
