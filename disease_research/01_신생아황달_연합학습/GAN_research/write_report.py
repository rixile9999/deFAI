"""Summarize all recorded arms without hiding unsuccessful generators."""
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CASES = ["central_real", "isolated_real", "federated_real", "federated_procedural_050", "federated_procedural_100",
         "federated_gan_050", "federated_gan_100", "federated_paired_gan_050", "federated_paired_gan_100"]
NAMES = ["중앙집중·원본", "기관 단독·원본", "연합학습·원본", "연합·기존 단색 생성 50%", "연합·기존 단색 생성 100%",
         "연합·잡음 조건부 GAN 50%", "연합·잡음 조건부 GAN 100%", "연합·촬영쌍 GAN 50%", "연합·촬영쌍 GAN 100%"]


def table(headers, rows):
    cells = lambda row, tag: "".join(f"<{tag}>{escape(str(x))}</{tag}>" for x in row)
    return "<table><tr>"+cells(headers, "th")+"</tr>"+"".join("<tr>"+cells(r, "td")+"</tr>" for r in rows)+"</table>"


def main():
    studies = [json.loads((ROOT / "results" / p / "study.json").read_text(encoding="utf-8"))
               for p in ["prediction", "paired_prediction", "procedural_half_prediction"]]
    runs = [r for s in studies for r in s["runs"]]
    if len(runs) != 27 or any(sum(r["case"] == name for r in runs) != 3 for name in CASES):
        raise RuntimeError("All nine conditions and three seeds must finish before reporting")
    audit = json.loads((ROOT / "data/audit.json").read_text(encoding="utf-8"))
    quality = json.loads((ROOT / "results/generation_quality.json").read_text(encoding="utf-8"))
    network = json.loads((ROOT / "results/network/check.json").read_text(encoding="utf-8"))
    summary, predictions = [], {}
    for name in CASES:
        selected = [r for r in runs if r["case"] == name]
        path = ROOT / "results" / ("procedural_half_prediction" if name == "federated_procedural_050" else
                                    "paired_prediction" if "paired" in name else "prediction")
        per_seed = []
        for seed in [17, 29, 43]:
            frame = pd.read_csv(path / f"{name}_seed{seed}_test.csv", dtype={"patient": str}).sort_values("patient")
            per_seed.append(frame)
        predictions[name] = per_seed
        summary.append({"case": name, "label": NAMES[CASES.index(name)],
                        "mae_mean": float(np.mean([r["metrics"]["mae_mgdl"] for r in selected])),
                        "mae_seed_sd": float(np.std([r["metrics"]["mae_mgdl"] for r in selected], ddof=1)),
                        "f1_mean": float(np.mean([r["metrics"]["f1"] for r in selected])),
                        "accuracy_mean": float(np.mean([r["metrics"]["accuracy"] for r in selected])),
                        "auc_mean": float(np.mean([r["metrics"]["roc_auc"] for r in selected]))})
    reference = predictions["federated_real"]
    comparisons = []
    rng = np.random.default_rng(20261016)
    for name in CASES:
        if name == "federated_real":
            continue
        differences = []
        for frame, base in zip(predictions[name], reference):
            assert np.array_equal(frame.patient, base.patient)
            differences.append(abs(frame.prediction-frame.blood).to_numpy()-abs(base.prediction-base.blood).to_numpy())
        difference = np.mean(differences, axis=0)
        bootstrap = difference[rng.integers(0, len(difference), size=(5000, len(difference)))].mean(1)
        comparisons.append({"case": name, "mae_difference_vs_real_fl": float(difference.mean()),
                            "patient_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist()})
    matched_generator_comparisons = []
    for kind in ["gan", "paired_gan"]:
        for ratio in ["050", "100"]:
            name, baseline = f"federated_{kind}_{ratio}", f"federated_procedural_{ratio}"
            difference = np.mean([abs(a.prediction-a.blood).to_numpy()-abs(b.prediction-b.blood).to_numpy()
                                  for a, b in zip(predictions[name], predictions[baseline])], axis=0)
            boot = difference[rng.integers(0, len(difference), size=(5000, len(difference)))].mean(1)
            matched_generator_comparisons.append({"case": name, "baseline": baseline,
                "mae_difference": float(difference.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()})
    source = Path(audit["source"])
    unchanged = sha256((source / "metadata.csv").read_bytes()).hexdigest() == audit["metadata_sha256"]
    unchanged = unchanged and all(sha256((source/"images"/name).read_bytes()).hexdigest() == digest
                                 for name, digest in audit["all_source_image_sha256"].items())
    if not unchanged:
        raise AssertionError("Original dataset changed")
    report = {"stage": "research_only", "summary": summary, "paired_comparisons": comparisons,
              "source_unchanged": unchanged, "image_quality": quality, "network_smoke": network,
              "test_patients": len(reference[0]), "seeds": [17, 29, 43], "runs": 27,
              "matched_generator_comparisons": matched_generator_comparisons,
              "uncertainty": "5000 paired patient bootstrap draws after averaging the three seed-specific errors; fixed split, no multiplicity adjustment"}
    bias_summary_path = ROOT / "results/skin_bias/summary.json"
    bias_section_path = ROOT / "results/skin_bias/report_section.html"
    bias_section = ""
    if bias_summary_path.exists() and bias_section_path.exists():
        report["skin_bias_research"] = json.loads(bias_summary_path.read_text(encoding="utf-8"))
        bias_section = bias_section_path.read_text(encoding="utf-8")
    bias_navigation = '<p><a href="#skin-bias">추가 연구: 피부색 관련 명도 편향 억제 방법·효과·한계 보기</a></p>' if bias_section else ''
    algorithm_section = ""
    algorithm_summary = ROOT / "results/algorithm_study/summary.json"
    algorithm_section_path = ROOT / "results/algorithm_study/report_section.html"
    if algorithm_summary.exists() and algorithm_section_path.exists():
        report["federated_algorithm_research"] = json.loads(algorithm_summary.read_text(encoding="utf-8"))
        algorithm_section = algorithm_section_path.read_text(encoding="utf-8")
        bias_navigation += '<p><a href="#federated-optimization">추가 연구: FedProx·SCAFFOLD·FedBN 최적화 비교</a></p>'
    (ROOT / "results/final_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(ROOT / "results/final_metrics.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 5))
    plt.barh(CASES[::-1], [r["mae_mean"] for r in summary][::-1],
             xerr=[r["mae_seed_sd"] for r in summary][::-1], color="#287f9b", capsize=3)
    plt.xlabel("Patient-level bilirubin MAE (mg/dL); lower is better")
    plt.title("Real held-out patients; mean and SD over 3 initialization seeds")
    plt.tight_layout(); plt.savefig(ROOT / "results/mae_comparison.png", dpi=160); plt.close()
    scores = table(["전체 비교 조건", "평균절대오차 ± 초기값 간 표준편차(mg/dL)", "F1", "정확도", "판별 곡선 아래 면적"], [
        [r["label"], f"{r['mae_mean']:.4f} ± {r['mae_seed_sd']:.4f}", f"{r['f1_mean']:.4f}",
         f"{100*r['accuracy_mean']:.2f}%", f"{r['auc_mean']:.4f}"] for r in summary])
    delta = table(["원본 연합학습 대비 조건", "평균절대오차 차이(mg/dL)", "환자 재표집 95% 구간"], [
        [NAMES[CASES.index(r["case"])], f"{r['mae_difference_vs_real_fl']:+.4f}",
         f"[{r['patient_bootstrap_ci95'][0]:+.4f}, {r['patient_bootstrap_ci95'][1]:+.4f}]"] for r in comparisons])
    quality_table = table(["영상 자료", "검증 영상과의 특징 분포 거리(낮을수록 가까움)"],
                          [[k, f"{v:.6f}"] for k, v in quality["distances"].items()])
    matched = table(["동일 증강 비율의 GAN − 기존 생성", "오차 차이(mg/dL)", "환자 재표집 95% 구간"], [
        [NAMES[CASES.index(r["case"])], f"{r['mae_difference']:+.4f}", f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"]
        for r in matched_generator_comparisons])
    changes = table(["항목", "기존 문제", "수정"], [
        ["평가 분할", "기관 안에서 사진 행을 나누어 동일 환자 유입 가능", "환자별 학습·검증·시험 분리, 시험 112명 고정"],
        ["자료 중복", "환자 번호가 다른 동일 사진에 상이한 라벨", "모호한 6묶음·12장 격리, 환자 내 중복 18장 정리"],
        ["라벨·결측", "0값·나이 결측을 암묵적으로 사용", "0값 1명 격리, 기관별 학습 중앙값으로 결측 처리"],
        ["정규화", "기관별 배치 통계 혼합, 단일 표본 배치 오류", "공통 사전학습 배치 통계 고정, 메타정보 LayerNorm"],
        ["집계", "일부 기관만으로 시작·정수 카운터 평균 가능", "3기관 모두 대기, 비정수 파라미터 가중평균·정수 버퍼 불일치 거부"],
        ["학습", "마지막 배치 누락·비정상 손실을 건너뜀", "모든 실제 표본 사용, 비정상 갱신 중단, mg/dL 회귀 손실 사용"],
        ["기관 가중치", "합성자료를 독립 환자처럼 셀 위험", "실제 학습 환자 수로 기관 가중치 고정"]])
    scope = table(["한 것", "안 한 것"], [
        ["기관별 두 종류 GAN 학습·총 3,088장 생성", "새로운 독립 환자·새 병원 자료 확보"],
        ["9조건 × 3초기값 × 20회 학습, 실제 환자번호 기준 시험", "여러 환자 분할·기관 배정·생성 초기값의 반복 검증"],
        ["3기관 전체 GPU 연합학습 계산과 별도 2회 실제 Flower 통신 시험", "기관 간 실배포·암호화·차등프라이버시"],
        ["생성 실패·특징 불일치 및 모든 비교 결과 공개", "합성 영상의 임상적 진실성·치료 판단 타당성 입증"],
        ["원본 자료 체크섬 확인·기존 소스 백업", "원본 환자번호·검사 라벨 출처의 임상적 판정"]])
    by_case = {r["case"]: r for r in summary}
    base_mae = by_case["federated_real"]["mae_mean"]
    paired_mae = by_case["federated_paired_gan_100"]["mae_mean"]
    observed = (f"이 고정 분할에서 촬영쌍 GAN 100% 증강은 평균절대오차 {paired_mae:.4f} mg/dL로, "
                f"원본 연합학습 {base_mae:.4f} mg/dL보다 {100*(1-paired_mae/base_mae):.2f}% 작았다. "
                "다만 같은 증강량의 기존 생성 대비 차이 구간에는 0이 포함되어 GAN 자체의 우위를 확정할 수 없다. "
                "잡음 조건부 GAN도 같은 비율의 기존 생성 대비 명확한 추가 이득을 확인하지 못했다. "
                "이는 이 데이터·이 분할·이 방식 한정이며 실패 결과를 포함한 전체 표를 함께 봐야 한다.")
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>황달 GAN·연합학습 재검증</title>
<style>body{{font:16px/1.8 'Malgun Gothic',sans-serif;background:#f2f6fa;color:#243247}}main{{max-width:1180px;background:white;margin:auto;padding:40px}}
h1,h2{{color:#173c60}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #d7e1ec;text-align:left;vertical-align:top}}th{{background:#eaf2fa}}
img{{max-width:100%}}.note{{padding:18px;background:#fff3d4}}code{{background:#edf3f9}}</style><main>
<p>2026-10-16 · RTX 4080 16GB · 연구 단계·실전 미적용</p><h1>신생아 황달 생성 모델 재구축과 연합학습 재계산</h1>
{bias_navigation}
<p>기존 가상 생성 코드는 빌리루빈 값에 따라 색을 칠하고 잡음을 넣는 절차식이었다.
실제 피부 영상의 기관별 조건부 생성 모델을 학습하고, 생성자료가 실제 시험 자료의 예측을 개선하는지 비교했다.
첫 GAN의 질감 결함도 공개하고, 같은 환자 촬영쌍을 이용하는 제한적 영상 증강 GAN을 추가했다.</p>
<div class="note">이 연구는 보유 자료의 환자번호와 라벨을 기준으로 한 계산 실험이다. 합성 영상이 새로운 환자를 뜻하지 않으며,
영상의 외관·특징 거리 개선이 임상 타당성이나 예측 개선을 입증하지 않는다. 기존 2.76 mg/dL 기록과는 분할·해상도·평가 단위가 달라 직접 개선율을 계산하지 않는다.</div>
<p>{observed}</p><h2>1. 자료를 먼저 정리했다</h2><p>원본 745개 환자번호·2,235장에서 사진 중복과 모호한 라벨을 점검했다.
서로 다른 번호의 동일 사진 6묶음(12장), 같은 환자 안의 중복 18장, 의미 미확인 빌리루빈 0값 1명(3장)을 주분석에서 격리했다.
정제 후 742개 환자번호·2,202장이다. 0값을 임상적으로 잘못된 수치라고 판정한 것은 아니며 출처 확인이 필요하다.</p>
<p>학습 519명·1,544장, 검증 111명·330장, 시험 112명·328장으로 나눴다. 동일 환자·동일 파일 내용은 분할 간 중복되지 않는다.
생후 나이가 누락된 2명의 6개 행은 각 기관의 학습자료에서만 구한 중앙값으로 대체했다.
피부 중앙 영역을 고정 비율로 잘라 색상 차트를 제외하고 모든 비교에 64×64 해상도를 적용했다. 원본은 수정하지 않았고 체크섬도 일치했다.</p>
<h2>2. 두 종류의 GAN과 한 가지 기존 생성 대조군</h2><p>GAN은 생성적 적대 신경망이다. 세 가상 기관마다 자기 학습 환자만 사용해 모델을 따로 학습했다.</p>
<ul><li><b>잡음 조건부 GAN</b>: 잡음·메타정보·빌리루빈·치료 라벨에서 피부 영상 생성. 기관당 3,000단계, 총 9,000단계.
판별기 스펙트럴 정규화·보조 회귀·미분 가능한 기하 증강·생성기 지수이동평균을 사용했다. RGB 색상은 임의 변형하지 않았다.</li>
<li><b>촬영쌍 영상 조건부 GAN</b>: 같은 학습 환자의 다른 사진을 학습쌍으로 사용. 실제 피부 구조를 보존하면서 촬영 변화를 학습하는 잔차 생성기다.
기관당 2,000단계, 총 6,000단계. RGB 잔차는 0~255 기준 최대 약 19로 제한했다.
이는 기존 사진의 작은 파생 증강이며, 원본에서 독립적인 새로운 환자 영상 생성으로 해석해서는 안 된다.</li>
<li><b>기존 생성 대조군</b>: 원래 단색·잡음 생성 방식을 같은 학습 환자 라벨에 적용했다.</li></ul>
<p>두 GAN은 각각 1,544장을 만들었다. 새 표본의 메타정보와 라벨은 출처인 학습 환자에서만 가져왔으며 검증·시험 환자는 생성에 쓰지 않았다.
50%/100%는 실제 학습 표본 수 대비 미니배치에 추가한 생성 표본 비율이다. 새 독립 환자 수가 증가한 것은 아니다.</p>
<img src="generation_comparison.jpg" alt="동일 출처 학습 영상과 세 생성 방식 비교">
<h2>3. 영상 품질도 별도로 확인했다</h2>{quality_table}
<p>위 값은 고정된 ImageNet ResNet18 특징의 비편향 다항 커널 분포 거리다. 표준 Inception 기반 지표가 아니며 임상적 실제성을 측정하지 않는다.
첫 잡음 GAN은 색과 무늬가 다양해도 인공 패턴이 남았다. 촬영쌍 GAN은 원본 구조를 직접 보존하므로 특징 거리가 작아지는 것이 자연스럽다.
이것만으로 더 좋은 증강이나 개인정보 보호를 주장하지 않는다. 사진을 기억·재현할 가능성과 라벨의 생리적 일치 여부는 별도 문제다.</p>
<h2>4. 연합학습 문제 수정</h2>{changes}
<p>예측기는 같은 ImageNet ResNet18·4개 메타정보·회귀/분류 출력 구조를 사용한다.
회귀는 mg/dL 단위 SmoothL1 손실, 분류는 이진 교차엔트로피의 0.5배를 더했다.
세 기관은 같은 공통 모델에서 시작하고, 지역 갱신 후 실제 학습 환자 수로 가중평균한다.
원본 연합학습과 모든 증강 조건은 같은 실제 자료 통과 수·같은 최적화 갱신 수를 사용한다. 증강 조건은 같은 갱신에 추가 표본을 넣으므로 처리 영상 수와 계산량은 더 많다.</p>
<h2>5. 실제 시험 환자에서의 결과</h2><p>20회 학습, 초기값 17·29·43의 세 실행을 비교했다.
각 실행은 환자 기준 검증 평균절대오차가 가장 작은 체크포인트를 선택한 후에만 시험 자료를 평가했다.
같은 환자의 여러 영상 예측은 먼저 평균해 환자 한 명을 한 평가 단위로 사용했다. 치료 판정 확률 기준은 모든 조건에서 0.5다.
전체 9조건은 표 순서대로 공개하며 시험 성능으로 선별해 숨긴 조건은 없다.
기존 생성의 50% 조건은 GAN과 증강량을 맞춘 대조가 빠져 있음을 보완하기 위해 추가했다.</p>
{scores}<img src="mae_comparison.png" alt="초기값 세 개에서 환자별 평균절대오차 비교">
<h2>6. 원본 연합학습 대비 차이</h2>{delta}
<p>GAN 자체의 추가 이득을 분리하기 위해 같은 증강 비율의 기존 생성 방식과도 직접 비교한다.</p>{matched}
<p>차이는 음수일수록 오차 감소다. 세 초기값의 환자별 오차 차이를 평균한 뒤 같은 112명을 5,000회 짝지어 재표집했다.
초기값 간 표준편차와 환자 재표집 구간은 서로 다른 불확실성이다. 한 번의 고정 분할에 조건부인 탐색 결과이고 다중 비교 보정을 하지 않았으므로,
작은 차이나 일부 구간만으로 일반적인 개선을 확정하지 않는다. 추가 모델은 첫 GAN의 검증 영상 특징·질감 문제를 근거로 추가했으며 두 모델의 시험 결과를 모두 남겼다.</p>
<h2>7. 통신과 실행 범위</h2><p>본 성능 계산은 한 대의 RTX 4080에서 세 기관을 순차 실행한 연합학습 모의실험이다.
별도로 수정한 실제 Flower 서버 1개·GPU 클라이언트 3개의 통신을 2회 검증했고 전 프로세스가 정상 종료했다.
이 통신 시험은 기관당 학습 영상 64장을 사용한 동작 확인이며, 위 전체 성능 수치와 혼동하지 않는다.</p>
{scope}<p>이 데이터·이 분할·이 방식 한정의 결과다. 미탐색 영역은 외부 병원 자료, 환자번호와 검사 라벨의 출처 검증,
색상 보정·더 높은 해상도, 다른 기관 간 이질성, 여러 데이터 분할·생성 모델 초기값, 합성 영상의 임상 평가와 기억 여부다.</p>
{bias_section}
{algorithm_section}
<h2>산출물·재현</h2><p>프로젝트 README.md에 실행 순서를 적었다. final_summary.json과 final_metrics.csv가 요약 정본이며,
prediction·paired_prediction에는 모든 초기값의 환자별 시험 예측·검증 이력·체크포인트가 있다.
gan·paired_gan에는 기관별 생성기·학습 로그·표본이 있고 data에는 분할·격리 목록·생성자료 출처가 있다.</p>
<p>용어: 평균절대오차는 예측과 검사값 차이의 절댓값 평균(mg/dL). F1은 정밀도·재현율의 조화평균.
판별 곡선 아래 면적은 확률 순위의 분류 성능이며 정확도와 다르다. %는 비율이고 %p는 비율의 차이다.</p>
<p>방법 근거: <a href="https://pytorch.org/get-started/previous-versions/">공식 PyTorch 설치·호환성</a>,
<a href="https://github.com/mit-han-lab/data-efficient-gans">소량 자료의 미분 가능 증강</a>,
<a href="https://arxiv.org/abs/1711.04340">자료 증강 GAN 원논문</a>. 본 코드는 해당 개념을 이용한 별도 연구 구현이며 논문의 전체 구현을 재현했다고 주장하지 않는다.</p>
</main></html>'''
    (ROOT / "results/gan_study_report.html").write_text(html, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False)); print("COMPARISONS", comparisons)


if __name__ == "__main__":
    main()
