"""Patient-paired uncertainty; disclose every method and subgroup."""
from html import escape
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from run_skin_bias_study import ROOT, OUTPUT, METHODS, GAINS
from skin_bias import normalize, denormalize, suppress_brightness

LABELS = {"baseline": "기존 입력", "gain_aug": "명도 변화 증강", "gain_consistency": "명도 증강 + 예측 일관성", "chromaticity": "색상 비율 입력"}


def table(headers, rows):
    def cells(row, tag):
        return "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in row)
    return "<table><tr>"+cells(headers, "th")+"</tr>"+"".join("<tr>"+cells(row, "td")+"</tr>" for row in rows)+"</table>"


def error_metrics(errors, groups):
    values = np.stack([errors[..., groups == g].mean(-1) for g in range(3)], axis=-1)
    return np.stack([errors.mean(-1), values.max(-1), values.max(-1)-values.min(-1)], axis=-1)


def main():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    assert len(protocol["runs"]) == 24, "Require all eight conditions and three seeds"
    groups = pd.read_csv(OUTPUT / "patient_brightness_groups.csv", dtype={"patient": str})
    data = np.load(ROOT / "data/real_data.npz")
    label_table = pd.DataFrame({"patient": data["patient"], "blood": data["blood"]}).groupby("patient").first().reset_index()
    train_table = groups[groups.split == "train"].merge(label_table, on="patient", validate="one_to_one")
    correlation = float(spearmanr(train_table.brightness, train_table.blood).statistic)
    fig, axes = plt.subplots(3, 5, figsize=(9, 6))
    for k in range(3):
        patient = train_table[train_table.group == k].sort_values("brightness").iloc[len(train_table[train_table.group == k])//2].patient
        image = data["images"][np.flatnonzero(data["patient"] == patient)[0]].astype(np.float32)/255
        transformed = [image*gain for gain in [1., .9, .75, .55]]
        tensor = torch.tensor(image).permute(2, 0, 1)[None]
        transformed.append(denormalize(suppress_brightness(normalize(tensor), "chromaticity"))[0].permute(1, 2, 0).numpy())
        for j, item in enumerate(transformed):
            axes[k, j].imshow(item); axes[k, j].set_xticks([]); axes[k, j].set_yticks([])
            if k == 0:
                axes[k, j].set_title(["Original", "Gain 0.90", "Gain 0.75", "Gain 0.55", "RGB ratios"][j])
            if j == 0:
                axes[k, j].set_ylabel(f"Photo group {k}")
    fig.tight_layout(); fig.savefig(OUTPUT / "transform_examples.png", dpi=160); plt.close(fig)
    test_groups = groups[groups.split == "test"].sort_values("patient")
    g = test_groups.group.to_numpy()
    rng = np.random.default_rng(20261016)
    # Stratified paired bootstrap: preserve observed subgroup sizes.
    resamples = np.concatenate([rng.choice(np.flatnonzero(g == k), (5000, int((g == k).sum()))) for k in range(3)], axis=1)
    boot_groups = g[resamples[0]]
    summaries, subgroup_rows, contrasts = [], [], []
    arrays = {}; baseline_rechecks = []
    for condition in ["real", "paired100"]:
        for method in METHODS:
            originals, changed = [], []
            for seed in [17, 29, 43]:
                prefix = f"{condition}_{method}_seed{seed}"
                frame = pd.read_csv(OUTPUT / f"{prefix}_gain1.00_test.csv", dtype={"patient": str}).sort_values("patient")
                assert np.array_equal(frame.patient, test_groups.patient)
                if method == "baseline":
                    folder, case = ("prediction", "federated_real") if condition == "real" else ("paired_prediction", "federated_paired_gan_100")
                    previous = pd.read_csv(ROOT / "results" / folder / f"{case}_seed{seed}_test.csv", dtype={"patient": str}).sort_values("patient")
                    assert np.array_equal(frame.patient, previous.patient)
                    # Separate GPU executions need not reproduce low precision
                    # kernel arithmetic bitwise. Explicitly report the measured
                    # drift; do not silently claim identical stored predictions.
                    maximum_delta = float(np.max(abs(frame.prediction.to_numpy()-previous.prediction.to_numpy())))
                    assert maximum_delta < .005, "Baseline rerun drift exceeds 0.005 mg/dL numerical guard"
                    baseline_rechecks.append({"condition": condition, "seed": seed,
                                              "max_prediction_delta_mgdl": maximum_delta})
                originals.append(frame)
                shifted = [pd.read_csv(OUTPUT / f"{prefix}_gain{gain:.2f}_test.csv", dtype={"patient": str}).sort_values("patient") for gain in GAINS]
                for f in shifted:
                    assert np.array_equal(f.patient, frame.patient)
                    np.testing.assert_array_equal(f.blood, frame.blood)
                    np.testing.assert_array_equal(f.treatment, frame.treatment)
                changed.append(shifted)
            y = originals[0].blood.to_numpy()
            predictions = np.stack([f.prediction.to_numpy() for f in originals])
            probabilities = np.stack([f.probability.to_numpy() for f in originals])
            shifted_predictions = np.stack([[f.prediction.to_numpy() for f in ss] for ss in changed])
            errors = abs(predictions-y).mean(0)
            sensitivity = abs(shifted_predictions-predictions[:, None, :]).mean((0, 1))
            stats = error_metrics(errors, g)
            arrays[(condition, method)] = (errors, sensitivity)
            # Case-mix standardization: only shared bins with >=3 patients in each group.
            # This changes the evaluated population; raw full-cohort metrics remain primary.
            bins = np.digitize(y, [8., 12., 16.])
            support = [b for b in range(4) if all(np.sum((g == k) & (bins == b)) >= 3 for k in range(3))]
            common = np.isin(bins, support)
            std_maes = []
            if support:
                for k in range(3):
                    std_maes.append(sum(np.mean(errors[(g == k) & (bins == b)])*np.sum(bins == b)/common.sum() for b in support))
            summary = {"condition": condition, "method": method, "mae_mgdl": float(stats[0]),
                       "mae_seed_sd": float(np.std(abs(predictions-y).mean(1), ddof=1)),
                       "worst_group_mae_mgdl": float(stats[1]), "group_mae_gap_mgdl": float(stats[2]),
                       "stress_prediction_shift_mgdl": float(sensitivity.mean()),
                       "stress_mae_mgdl": float(abs(shifted_predictions-y).mean()),
                       "standardized_gap_mgdl": float(np.ptp(std_maes)) if std_maes else None,
                       "standardized_support_bins": support, "standardized_coverage_n": int(common.sum()),
                       "accuracy": float(((probabilities >= .5) == originals[0].treatment.to_numpy()).mean())}
            summaries.append(summary)
            for k in range(3):
                chosen = g == k
                positives = chosen & (originals[0].treatment.to_numpy() == 1)
                negatives = chosen & (originals[0].treatment.to_numpy() == 0)
                subgroup_rows.append({"condition": condition, "method": method, "group": k,
                    "n_patients": int(chosen.sum()), "n_treatment_positive": int(positives.sum()),
                    "blood_mean_mgdl": float(y[chosen].mean()), "mae_mgdl": float(errors[chosen].mean()),
                    "signed_error_mgdl": float((predictions[:, chosen]-y[chosen]).mean()),
                    "false_negative_rate": float((probabilities[:, positives] < .5).mean()) if positives.any() else None,
                    "false_positive_rate": float((probabilities[:, negatives] >= .5).mean()) if negatives.any() else None})
            if method != "baseline":
                base_errors, base_sensitivity = arrays[(condition, "baseline")]
                delta = error_metrics(errors[resamples], boot_groups)-error_metrics(base_errors[resamples], boot_groups)
                sens_delta = (sensitivity-base_sensitivity)[resamples].mean(1)
                point = np.r_[stats-error_metrics(base_errors, g), (sensitivity-base_sensitivity).mean()]
                samples = np.column_stack([delta, sens_delta])
                for j, metric in enumerate(["mae_mgdl", "worst_group_mae_mgdl", "group_mae_gap_mgdl", "stress_prediction_shift_mgdl"]):
                    contrasts.append({"condition": condition, "method": method, "metric": metric,
                                      "difference_vs_baseline": float(point[j]),
                                      "paired_stratified_patient_ci95": np.quantile(samples[:, j], [.025, .975]).tolist()})
    ablations = []
    for condition in ["real", "paired100"]:
        aug_error, aug_sensitivity = arrays[(condition, "gain_aug")]
        error, sensitivity = arrays[(condition, "gain_consistency")]
        delta = error_metrics(error[resamples], boot_groups)-error_metrics(aug_error[resamples], boot_groups)
        samples = np.column_stack([delta, (sensitivity-aug_sensitivity)[resamples].mean(1)])
        point = np.r_[error_metrics(error, g)-error_metrics(aug_error, g), (sensitivity-aug_sensitivity).mean()]
        for j, metric in enumerate(["mae_mgdl", "worst_group_mae_mgdl", "group_mae_gap_mgdl", "stress_prediction_shift_mgdl"]):
            ablations.append({"condition": condition, "contrast": "gain_consistency minus gain_aug", "metric": metric,
                              "difference": float(point[j]), "ci95": np.quantile(samples[:, j], [.025, .975]).tolist()})
    pd.DataFrame(summaries).to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(subgroup_rows).to_csv(OUTPUT / "subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    report = {"protocol": protocol, "summary": summaries, "subgroups": subgroup_rows, "contrasts": contrasts,
              "train_brightness_blood_spearman": correlation, "consistency_ablation": ablations,
              "baseline_numerical_rechecks": baseline_rechecks,
              "uncertainty": "5000 patient paired bootstrap stratified on fixed photo-brightness group; errors averaged over three seeds first; no multiplicity correction; fixed split only"}
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition, marker in [("real", "o"), ("paired100", "s")]:
        subset = [s for s in summaries if s["condition"] == condition]
        for ax, metric in zip(axes, ["group_mae_gap_mgdl", "stress_prediction_shift_mgdl"]):
            ax.plot(METHODS, [s[metric] for s in subset], marker=marker, label=condition)
            ax.tick_params(axis="x", rotation=20); ax.set_ylabel("mg/dL"); ax.legend()
    axes[0].set_title("Observed brightness-group MAE disparity")
    axes[1].set_title("Prediction sensitivity to synthetic attenuation")
    fig.tight_layout(); fig.savefig(OUTPUT / "bias_comparison.png", dpi=160); plt.close(fig)
    score_table = table(["자료", "방법", "전체 오차", "최대 집단 오차", "집단 오차 격차", "변환 예측 변화", "변환 후 오차", "치료 정확도"], [
        ["원본" if s["condition"] == "real" else "촬영쌍 생성 100%", LABELS[s["method"]], *[f"{s[k]:.4f}" for k in
          ["mae_mgdl", "worst_group_mae_mgdl", "group_mae_gap_mgdl", "stress_prediction_shift_mgdl", "stress_mae_mgdl"]], f"{100*s['accuracy']:.2f}%"] for s in summaries])
    gap_contrasts = [c for c in contrasts if c["metric"] in ["group_mae_gap_mgdl", "stress_prediction_shift_mgdl"]]
    contrast_table = table(["자료", "방법", "지표", "기존 입력 대비 차이(mg/dL)", "95% 재표집 구간"], [
        [c["condition"], LABELS[c["method"]], "집단 오차 격차" if c["metric"] == "group_mae_gap_mgdl" else "변환 예측 변화",
         f"{c['difference_vs_baseline']:+.4f}", "["+", ".join(f"{x:+.4f}" for x in c["paired_stratified_patient_ci95"])+"]"] for c in gap_contrasts])
    ablation_table = table(["자료", "일관성 추가 − 증강만", "차이(mg/dL)", "95% 재표집 구간"], [
        [c["condition"], c["metric"], f"{c['difference']:+.4f}", "["+", ".join(f"{x:+.4f}" for x in c["ci95"])+"]"] for c in ablations])
    subgroup_table = table(["자료·방법", "사진 명도 집단", "환자 수/치료 양성 수", "평균 검사값", "평균절대오차", "부호 있는 오차", "치료 양성 누락률/음성 오탐률"], [
        [r["condition"]+" · "+LABELS[r["method"]], ["낮음", "중간", "높음"][r["group"]],
         f"{r['n_patients']}/{r['n_treatment_positive']}", f"{r['blood_mean_mgdl']:.3f}", f"{r['mae_mgdl']:.3f}",
         f"{r['signed_error_mgdl']:+.3f}",
         (f"{100*r['false_negative_rate']:.2f}%" if r["false_negative_rate"] is not None else "계산 불가")+" / "+
         (f"{100*r['false_positive_rate']:.2f}%" if r["false_positive_rate"] is not None else "계산 불가")] for r in subgroup_rows])
    standard_table = table(["자료", "방법", "공통 검사값 구간의 표준화 오차 격차(mg/dL)", "포함 환자 수/전체 112명", "사용 구간 번호"], [
        [s["condition"], LABELS[s["method"]], f"{s['standardized_gap_mgdl']:.4f}" if s["standardized_gap_mgdl"] is not None else "공통 지지 부족",
         s["standardized_coverage_n"], str(s["standardized_support_bins"])] for s in summaries])
    observations = []
    for condition in ["real", "paired100"]:
        baseline = next(s for s in summaries if s["condition"] == condition and s["method"] == "baseline")
        for method in METHODS[1:]:
            row = next(s for s in summaries if s["condition"] == condition and s["method"] == method)
            gap = next(c for c in contrasts if c["condition"] == condition and c["method"] == method and c["metric"] == "group_mae_gap_mgdl")
            ci = gap["paired_stratified_patient_ci95"]
            evidence = "재표집 구간 전체가 음수인 격차 감소" if ci[1] < 0 else "재표집 구간 전체가 양수인 격차 증가" if ci[0] > 0 else "격차 차이 구간에 0이 포함되어 뚜렷한 감소 확인 불가"
            observations.append(f"<li>{'원본' if condition == 'real' else '촬영쌍 생성 100%'} · {LABELS[method]}: "
                f"명도 집단 격차 {baseline['group_mae_gap_mgdl']:.4f} → {row['group_mae_gap_mgdl']:.4f}, "
                f"전체 오차 {baseline['mae_mgdl']:.4f} → {row['mae_mgdl']:.4f} mg/dL, "
                f"치료 정확도 {100*baseline['accuracy']:.2f}% → {100*row['accuracy']:.2f}% "
                f"({100*(row['accuracy']-baseline['accuracy']):+.2f}%p). {evidence}.</li>")
    # No post-test method selection: present all eight in prespecified order.
    section = f'''<h2 id="skin-bias">8. 피부색 관련 편향 억제 연구: 명도 성분에 한정한 검증</h2>
<div class="note"><b>연구 단계·실전 미적용.</b> 본래 피부색·멜라닌 수치·황달 이전 사진이 없다.
아래 집단은 황달·카메라·조명이 섞인 <b>사진 명도 집단</b>이며 생물학적 피부색 집단이나 인종 집단이 아니다.
이 실험으로 본래 피부색 편향 제거 또는 임상 공정성 달성을 주장할 수 없다.</div>
<p>피부 명도에 기대는 예측을 줄이면서 황달의 상대적 색 신호를 보존하기 위해 세 방법을 구현했다.
① 명도 변화 증강: 원본과 모든 RGB 채널을 같은 배수(0.65~1.00)로 줄인 영상의 지도 손실을 각각 0.5배로 합산.
② 예측 일관성: 같은 증강 손실에 원본·변환 영상의 빌리루빈 예측 차이 SmoothL1의 0.5배 및 치료 확률 차이 제곱의 0.5배 추가.
③ 색상 비율 입력: 픽셀마다 RGB를 R+G+B로 나누어 공통 명도 배수에 수학적으로 불변인 입력 사용.
후자는 픽셀 밝기 질감까지 없애므로 실제 오차 악화를 함께 점검했다. ①과 ② 비교는 일관성 항의 추가 기여를 보여준다.
실제·생성 영상 모두 동일 변환을 사용했고 라벨·메타정보는 유지했다. 생성기를 다시 학습한 것은 아니며 예측기의 학습·입력을 바꿨다.</p>
<p>중요한 가정: 채널 공통 배수는 RGB 비율을 보존하지만 실제 멜라닌의 파장별 흡수를 재현하지 않는다.
인코딩된 RGB에서 수행하므로 물리적 선형 광량 변환도 아니다. 따라서 이름도 생리적 멜라닌 증강이 아닌 제한적 명도 스트레스다.
빌리루빈의 진단 신호가 완전히 보존된다는 임상 검증도 하지 않았다.</p>
<p>원본/촬영쌍 생성 100% × 기존 입력/세 방법 × 초기값 17·29·43 = 24개 모델 평가.
기존 6개 모델은 저장된 체크포인트를 재사용하고 새 18개 모델은 RTX 4080에서 각각 20회·3기관 연합학습했다.
기존 모델의 재평가는 이전 저장 예측과 완전히 같은 비트값은 아니며 최대 차이는
{max(r['max_prediction_delta_mgdl'] for r in baseline_rechecks):.6f} mg/dL였다.
원본 입력과 정규화 왕복 입력의 별도 재평가가 같은 값을 내는 점도 점검했다(baseline_numerical_audit.json).
원래 결과표는 보존하고 이 절의 대조군·변환 비교는 동일 추가 평가 실행의 예측을 사용했다.
학습 환자·갱신 수·기관 가중치는 종전과 같으며, 증강 두 방법은 같은 갱신에서 두 번 순전파하므로 계산량이 늘었다.
손실 계수·방법·변환 범위는 시험 예측 전에 protocol.json으로 고정했다.
선택은 원본 검증 사진의 환자 평균절대오차만 사용했고 시험 집단 격차로 모델을 고르지 않았다.</p>
<p>집단 기준: 사진 중앙 피부 영역의 0~1 RGB에서 0.2126R+0.7152G+0.0722B를 평균하고 환자별로 다시 평균.
학습 519명의 삼분위 경계 {protocol['group_cutoffs_train_patient_terciles']}를 검증·시험에도 그대로 적용했다.
노란 성분이 들어가는 피부 유형 각도나 인종 추정은 사용하지 않았다.</p>
<p>이 집단화는 연구용 중앙 평가에만 쓰고 예측기 학습이나 기관별 모델 집계 가중치에는 사용하지 않았다.
실제 병원 간 개인정보 보호를 유지한 집단 통계 산출 절차를 구현·검증한 결과는 아니다.</p>
<p>학습 환자에서 사진 명도와 빌리루빈의 순위 상관계수는 {correlation:+.3f}이다.
이 값은 연관성만 나타내며 피부색·황달·촬영 조건을 분리하지 않는다. 아래는 집단별 학습 환자 명도 중앙에 가까운
예시 한 명씩의 원본·명도 변환·색상 비율 입력이다. 시험 환자를 골라 예시로 쓰지 않았다.</p>
<img src="skin_bias/transform_examples.png" alt="학습 명도 집단별 원본과 채널 비율 보존 명도 스트레스 예시">
<ul>{''.join(observations)}</ul>
{score_table}<p>치료 정확도를 제외한 오차·변화 수치 단위는 mg/dL이며 낮을수록 작다. 치료 정확도는 %이며 높을수록 좋다.
전체 오차는 환자별 절대오차의 평균,
최대 집단 오차는 세 집단 중 가장 큰 평균절대오차, 집단 격차는 최대−최소 집단 평균절대오차다.
변환 예측 변화는 같은 환자 원본 대비 0.55·0.75·0.90배 시험 영상의 예측 차이 절댓값 평균이다.
0.55배는 학습 범위 밖 외삽 스트레스이고 나머지는 학습 범위 안이다.
색상 비율 방법의 변환 민감도가 거의 0인 것은 수학적 구성의 결과이며 실제 피부색 공정성의 증거가 아니다.
오차 지표는 세 초기값의 환자별 절대오차를 먼저 평균해 산출했다. 전체 오차는 초기값별 전체 오차 평균과 같지만,
집단 최대값·격차는 초기값별 최대값·격차 평균과 다를 수 있다. 예측을 먼저 평균한 앙상블 점수는 아니다.</p>
<img src="skin_bias/bias_comparison.png" alt="사진 명도 집단 오차 격차와 합성 명도 변화 민감도 비교">
{contrast_table}<p>음수는 기존 입력 대비 해당 격차 또는 민감도 감소다. 같은 환자를 짝지어 세 초기값의 환자 오차를 평균하고,
명도 집단별 표본 수를 고정해 5,000번 재표집한 구간이다. 고정 분할에 조건부이며 초기값 불확실성·외부 병원 변이는 포함하지 않는다.
다중 비교 보정이 없어 탐색적 결과다. 0을 포함한 차이는 이 실험에서 명확한 개선으로 판정하지 않는다.</p>
<h3>집단별 전체 공개와 검사값 구성 차이</h3>{subgroup_table}
<p>오차·검사값 단위는 mg/dL, 부호 있는 오차는 예측−검사값이므로 음수는 과소예측이다.
치료 양성 누락률은 실제 치료 라벨 1인 환자 중 예측 확률 0.5 미만의 비율, 음성 오탐률은 실제 치료 라벨 0인 환자 중
예측 확률 0.5 이상의 비율이며 각 초기값 비율을 평균했다.
작은 집단의 비율은 불안정하다. 집단마다 검사값·치료 비율이 달라 단순 격차는 피부색의 인과 효과가 아니다.</p>
<h3>일관성 항 자체의 추가 기여</h3>{ablation_table}
<p>mae_mgdl은 전체 오차, worst_group_mae_mgdl은 최대 집단 오차, group_mae_gap_mgdl은 집단 격차,
stress_prediction_shift_mgdl은 합성 명도 변화 민감도다. 두 방법의 변환 범위·처리 영상 수는 같고 일관성 항만 다르다.
음수는 단순 증강 대비 해당 지표 감소이며 구간에 0이 있으면 추가 효과는 불명확하다.</p>
{standard_table}<p>검사값 구간 0: 8 미만, 1: 8 이상 12 미만, 2: 12 이상 16 미만, 3: 16 이상(mg/dL).
세 집단에 각각 최소 3명이 있는 공통 구간만 남기고, 공통 구간 전체 시험 환자의 구간 비중으로 각 집단 오차를 표준화했다.
이는 사후 기술적 구성 보정이며 인과적 교정이 아니다. 포함 수를 명시했으며 제외 환자는 위 주평가에서 모두 유지했다.</p>
{table(['한 것', '안 한 것'], [['세 억제 방법과 단순 증강 대조·두 자료 조건·세 초기값 전수 평가', '본래 피부색·멜라닌 측정에 따른 공정성 검증'], ['명도 집단별 회귀·치료 누락률, 짝지은 재표집 구간', '다른 병원·다른 환자 분할·병리학적 색 신호 보존 검증'], ['학습 범위 안팎 합성 명도 스트레스와 검사값 구성 보정', '실제 멜라닌·혈색소·빌리루빈의 광학적 분리 및 조명 보정']])}
<p>따라서 결론은 이 데이터·이 분할·이 명도 변환 방식 한정이다. 실제 피부색 편향 검증에는 황달 전후 사진 또는 독립적인
멜라닌/색도계 측정과 여러 피부색을 포함한 외부 병원 평가가 필요하다.</p>
<p>근거: <a href="https://arxiv.org/abs/2110.11205">자료 증강·일관성 정규화 원논문</a>의 접근에서 착안한 별도 출력 일관성 구현이며 논문 재현은 아니다.
피부색 측정의 혼란변수와 독립적 측정 필요성은 <a href="https://www.nature.com/articles/s43856-024-00550-7">멜라닌 측정법 검토</a>를 참고했다.
재현 산출물: skin_bias/protocol.json, summary.json, summary.csv, subgroup_metrics.csv, 모든 모델별 원본·변환 시험 예측 및 검증 이력.</p>'''
    (OUTPUT / "report_section.html").write_text(section, encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
