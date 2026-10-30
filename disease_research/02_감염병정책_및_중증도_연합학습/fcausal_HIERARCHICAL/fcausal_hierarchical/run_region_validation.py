"""Generate more provinces and test the real count-data analysis pipeline.

The 16-cell main matrix is 8/32 provinces x null/reduction x equal/unequal
district sizes x serial correlation 0/.8. Two 64-province cases and four
boundary/identification checks are reported separately (22 datasets total).
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from hashlib import sha256
from html import escape
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

from core.policy_stats import AnalysisConfig, analyze, district_vector, province_summary
from core.synthetic_regions import RegionGeneration, export_regions, generate_regions
from run_policy_validation import (ROOT, load_records, make_contexts, reference_analysis,
                                   run_strategies, validate_network)


def scenarios():
    specs = []
    for provinces, ratio, unequal, rho in itertools.product([8, 32], [1.0, .7], [False, True], [0.0, .8]):
        name = f"p{provinces:02d}_ratio{round(100*ratio):03d}_{'unequal' if unequal else 'equal'}_rho{round(100*rho):02d}"
        specs.append((name, "main", RegionGeneration(n_provinces=provinces, n_treated=provinces//2,
                                                    policy_ratio=ratio, unequal_districts=unequal,
                                                    serial_correlation=rho)))
    for ratio in [1.0, .7]:
        specs.append((f"p64_ratio{round(100*ratio):03d}", "scale", RegionGeneration(
            n_provinces=64, n_treated=32, policy_ratio=ratio)))
    specs.extend([
        ("one_treated", "inference_boundary", RegionGeneration(n_treated=1)),
        ("low_counts", "low_counts", RegionGeneration(base_cases=1)),
        ("differential_trend", "identification_violation", RegionGeneration(policy_ratio=1, differential_trend=.006)),
        ("reporting_shift", "identification_violation", RegionGeneration(policy_ratio=1, reporting_ratio=.6))])
    return specs


def summary_from_records(records, config):
    provinces = []
    for region in sorted({r["province"] for r in records}):
        selected = [r for r in records if r["province"] == region]
        vector = sum(district_vector(r["counts"], config) for r in selected)
        provinces.append(province_summary(vector, region, selected[0]["treated"], config, len(selected)))
    return analyze(provinces)


def independent_regression(records, config):
    y = np.log1p(np.stack([r["counts"] for r in records]))
    pre = config.calendar < pd.Timestamp(config.policy)
    changes = y[:, ~pre].mean(1)-y[:, pre].mean(1)
    model = sm.OLS(changes, sm.add_constant([r["treated"] for r in records])).fit()
    cov = cov_cluster(model, np.array([r["province"] for r in records]), use_correction=True)
    return float(model.params[1]), float(np.sqrt(cov[1, 1]))


def validate_dataset(folder, records, config, contexts):
    checks, output = [], None
    for column in ["발병일", "진단일"]:
        analysis_config = replace(config, date_column=column)
        loaded = load_records(folder, analysis_config)
        original = {r["district"]: r for r in records}
        for record in loaded:
            source = original[record["district"]]
            expected = source["counts"] if column == "발병일" else source["diagnosed_counts"][:len(config.calendar)]
            np.testing.assert_array_equal(record["counts"], expected)
        reference = reference_analysis(loaded, analysis_config)
        with redirect_stdout(io.StringIO()):
            plain = run_strategies(loaded, analysis_config, contexts, False)
            encrypted = run_strategies(loaded, analysis_config, contexts, True)
        effect, se = independent_regression(loaded, analysis_config)
        error = abs(encrypted["effect"]-reference["effect"])
        assert abs(plain["effect"]-reference["effect"]) < 1e-12
        assert abs(effect-plain["effect"]) < 1e-12
        assert error < 1e-6
        se_error = None
        if plain["inference"]["status"] != "withheld":
            assert abs(plain["inference"]["se"]-se) < 1e-10
            se_error = abs(encrypted["inference"]["se"]-se)
            assert se_error < 1e-6
        checks.append({"date_column": column, "effect": plain["effect"],
                       "ckks_effect_abs_error": error, "ckks_se_abs_error": se_error,
                       "inference": plain["inference"]})
        if column == "발병일":
            output = plain
    return checks, output


def repeat_validation(settings, config, repetitions):
    errors, estimates, oracles, covers, rejections, placebos = [], [], [], [], [], []
    withheld = 0
    for seed in range(1000, 1000+repetitions):
        records, truth = generate_regions(replace(settings, seed=seed), config)
        result = summary_from_records(records, config)
        oracle = truth["oracle_policy_effect_log1p"]
        estimates.append(result["effect"]); oracles.append(oracle)
        errors.append(result["effect"]-oracle); placebos.append(result["pre_placebo_effect"])
        inf = result["inference"]
        if inf["status"] == "withheld":
            withheld += 1
        else:
            covers.append(inf["ci95"][0] <= oracle <= inf["ci95"][1])
            rejections.append(inf["p_value"] < .05)
    rejection = float(np.mean(rejections)) if rejections else None
    return {"replications": repetitions, "seed_start": 1000, "mean_estimate": float(np.mean(estimates)),
            "mean_oracle_policy_effect": float(np.mean(oracles)), "bias_vs_oracle": float(np.mean(errors)),
            "rmse_vs_oracle": float(np.sqrt(np.mean(np.square(errors)))),
            "coverage95_vs_realized_oracle": float(np.mean(covers)) if covers else None,
            "reject_zero_rate": rejection, "withheld_count": withheld,
            "mean_pre_placebo": float(np.mean(placebos)),
            "rejection_mc_se": float(np.sqrt(rejection*(1-rejection)/len(rejections))) if rejections else None}


def write_report(evidence, output):
    def table(headers, rows):
        cells = lambda row, tag: "".join(f"<{tag}>{escape(str(x))}</{tag}>" for x in row)
        return "<table><tr>"+cells(headers, "th")+"</tr>"+"".join("<tr>"+cells(r, "td")+"</tr>" for r in rows)+"</table>"
    fmt = lambda x: "보류" if x is None else f"{100*x:.1f}%"
    cases = table(["조건", "구분", "광역(처치/대조)", "시군구", "추정 대비", "생성 기준 대비", "추론"], [
        [r["name"], r["kind"], f"{r['settings']['n_provinces']} ({r['settings']['n_treated']}/{r['settings']['n_provinces']-r['settings']['n_treated']})",
         r["truth"]["n_districts"], f"{r['analysis']['effect']:+.4f}",
         f"{r['truth']['oracle_policy_effect_log1p']:+.4f}", r["analysis"]["inference"]["status"]] for r in evidence["datasets"]])
    simulations = table(["조건", "반복 수", "평균 편향(로그 단위)", "95% 구간 포함률", "효과 0 기각률", "사전 가짜개입 대비"], [
        [r["name"], r["repeated"]["replications"], f"{r['repeated']['bias_vs_oracle']:+.5f}",
         fmt(r["repeated"]["coverage95_vs_realized_oracle"]), fmt(r["repeated"]["reject_zero_rate"]),
         f"{r['repeated']['mean_pre_placebo']:+.4f}"] for r in evidence["datasets"]])
    max_error = max(c["ckks_effect_abs_error"] for r in evidence["datasets"] for c in r["agreement"])
    calibration_rows = []
    for provinces in [8, 32, 64]:
        selected = [r for r in evidence["datasets"] if r["settings"]["n_provinces"] == provinces
                    and r["kind"] in ("main", "scale")]
        cover = [r["repeated"]["coverage95_vs_realized_oracle"] for r in selected]
        nulls = [r["repeated"]["reject_zero_rate"] for r in selected if r["settings"]["policy_ratio"] == 1]
        calibration_rows.append([provinces, f"{100*min(cover):.1f}~{100*max(cover):.1f}%",
                                 f"{100*min(nulls):.1f}~{100*max(nulls):.1f}%"])
    calibration = table(["광역 수", "조건별 95% 구간 포함률 범위", "효과 0 조건의 거짓 양성률 범위"], calibration_rows)
    network = table(["방식", "프로세스", "광역/처치 광역/시군구", "중앙집중 대비 오차", "추론"], [
        [r["mode"], r["processes"], f"{r['n_provinces']}/{r['n_treated_provinces']}/{r['n_districts']}",
         f"{r['absolute_error_vs_pooled']:.3e}", r["inference_status"]] for r in evidence["network"]])
    scope = table(["한 것", "안 한 것"], [
        ["가상 광역 8·32·64개 및 명시적 다중 처치 배정", "실제 국내 행정구역·정책의 추가 수집"],
        ["22개 조건 파일 생성, 두 날짜 기준 중앙집중·계층·암호화 비교", "모든 효과 크기·지역 수·보고 과정의 전수 탐색"],
        [f"22개 조건 각각 {evidence['replications']}회 발생·보고 자료 재생성", "실제 감염 전파·지역 간 파급의 구현"],
        ["8개 광역 실제 프로세스 통신, 64개 광역까지 암호화 전략 실행", "32·64개 광역 전체의 동시 프로세스 부하 시험"],
        ["기존 환자 목록·키 보존 및 체크섬 대조", "보고율 변화·평행추세 위반의 자동 보정"]])
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>합성 지역 확장·검증</title>
<style>body{{font:16px/1.75 'Malgun Gothic',sans-serif;background:#f3f6fa;color:#243247}}main{{max-width:1350px;background:white;margin:auto;padding:36px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}}th,td{{padding:8px;border-bottom:1px solid #d9e3ed;text-align:left}}th{{background:#edf3f9}}
h1,h2{{color:#163d62}}.note{{background:#fff3d2;padding:18px}}code{{background:#edf3f9}}</style><main>
<p>2026-10-16 · 연구 단계 · 합성 자료 · 실전 미적용</p><h1>지역을 늘려 무엇을 검증했는가</h1>
<p>기존 포아송 발생·관측 누락·주간 패턴·점진적 정책효과·진단 지연 생성 과정을 확장했다.
실제 지역의 관측을 늘린 것은 아니며 Region_001 같은 가상 광역을 사용한다. 처치는 광역 단위로 무작위 배정하고 파일에 명시한다.</p>
<div class="note">합성 지역을 늘리면 계산과 불확실성 방법을 검증할 수 있다. 실제 자료의 처치 광역이 한 곳이라는 제약이 사라지거나 실제 정책의 인과효과가 입증되는 것은 아니다.</div>
<h2>생성·관측·분석의 연결</h2><p>지역별 잠재 환자 수 → 관측 누락 → 정책에 의한 감소 → 추가 보고율 변화 → 진단 지연 순으로 생성한다.
환자 개별 속성을 쓰지 않는 분석이므로 같은 포아송·독립 표집 과정의 일별 집계 CSV를 저장한다.
0건 날짜와 진단 지연의 끝부분 5일을 명시적으로 보존한다. 기존 XLSX 환자 목록과 새 CSV 집계를 구별하여 읽는다.</p>
<p>주요 목표값은 이전과 같은 시군구 동등 가중 <code>log(1+관측 일별 환자 수)</code>의 전후 변화 차이다.
정책 배율 0.7은 충분히 시행된 뒤 잠재 발생률에 30% 감소를 넣었다는 뜻이다. 로그 변환·관측 누락·7일 시행 경사 때문에 추정 목표값은 log(0.7)과 같지 않다.</p>
<p>검증 기준인 생성 기준 대비(oracle)는 같은 표집된 발생에서 만든 정책 유무 쌍의 처치 시군구 사후 log1p 차이다.
이는 안정된 보고율에서의 <b>실현된 처치군 정책 대비</b>로, 모수의 log 배율이나 환자 감소율과 다르다.
반복 검증의 신뢰구간 포함률도 이 실현 대비를 기준으로 한다. 진단일 분석은 중앙집중 일치와 민감도만 확인하며 발병일용 oracle에 대한 포함률로 해석하지 않는다.</p>
<h2>생성 자료 전체</h2><p>주 행렬 16개: 광역 8/32 × 정책 배율 1/0.7 × 동일/상이한 시군구 수 × 시계열 상관 0/0.8.
64개 광역 두 사례와 처치 광역 1개·낮은 발생수·차별적 추세·보고율 변화 네 사례를 추가해 총 22개다.
equal은 광역당 3개 시군구, unequal은 1~6개다. 나머지 생성 조건은 JSON에 모두 기록했다.</p>
{cases}<p>다중 처치 사례에서는 표준오차·유의확률·신뢰구간을 계산할 수 있으며, 처치 광역 한 곳인 경계 사례에서는 계속 보류한다. 적은 광역 수의 결과는 탐색적이다.</p>
<h2>계산 일치와 독립 회귀 대조</h2><p>22개 자료를 발병일·진단일 두 정의로 읽어 총 44개 비교를 수행했다.
평문 계층 합산·암호화 합산의 점추정치를 중앙집중 계산과 비교하고, 유의성 계산이 가능한 자료의 군집 강건 표준오차는 독립적인 statsmodels 회귀와 대조했다.
암호화 점추정치의 최대 절대오차는 {max_error:.3e} 로그 단위다.</p>
<h2>실제 발생·관측 생성기를 반복한 검증</h2><p>조건마다 난수 초기값 1000부터 {evidence['replications']}개를 사용했다.
각 반복에서 지역 배정·환자 발생·관측 누락·보고 지연까지 다시 생성했다.
효과가 없는 주 행렬의 기각률은 거짓 양성률, 효과가 있는 조건에서는 검정력이다.
보고율 변화·차별적 추세 조건의 기각률은 식별 가정 위반에 대한 취약성을 나타낸다.
{evidence['replications']}회에서 5% 비율의 모의실험 표준오차는 {100*np.sqrt(.05*.95/evidence['replications']):.2f}%p다.</p>
{calibration}<p>명목 목표는 포함률 95%, 거짓 양성률 5%다. 위 값은 조건 사이의 범위이며 모수 전체에 대한 신뢰구간이 아니다.
반복 수에 따른 오차를 고려해도 작은 광역 수의 추론을 자동으로 신뢰해서는 안 된다는 점을 확인할 수 있다.</p>
{simulations}<p>지역을 늘리는 것만으로 잘못된 식별 가정이 해결되지는 않는다. 특히 정책 후 보고율 변화는 사전 가짜개입 대비가 작아도 정책효과처럼 보일 수 있다.
이 자료·이 생성 방식에서 확인한 결과이며, 미탐색 조건은 서로 다른 정책 시행일, 감염 파급, 다른 효과 크기와 장기 추세, 비무작위 처치 배정이다.</p>
<h2>실제 다중 프로세스 시험</h2>{network}<p>같은 컴퓨터의 평문·암호화 서버/클라이언트 시험이다. 중간 서버에는 공개키만 로드하며 테스트 키는 임시 생성 후 정리한다.</p>
<h2>범위와 산출물</h2>{scope}<p>각 자료 폴더의 manifest.json에 설정·처치 배정·생성 기준 대비·파일 체크섬이 있다.
counterfactual_no_policy_cases와 policy_only_cases는 합성 연구의 정답 대조용이며 실제 연합 분석은 이 열을 읽지 않는다.
상세 측정은 region_validation.json, 반복 표는 region_simulation.csv에 저장한다. 생성·검증 명령은 프로젝트 README.md를 따른다.</p></main></html>'''
    (output / "region_expansion_report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--data-output", type=Path, default=ROOT / "data/expanded_validation")
    parser.add_argument("--results-output", type=Path, default=ROOT / "results/expanded_regions")
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("Need at least two repetitions")
    args.data_output.mkdir(parents=True, exist_ok=False)
    args.results_output.mkdir(parents=True, exist_ok=False)
    config, contexts = AnalysisConfig(), make_contexts()
    evidence = {"stage": "research_only_synthetic", "replications": args.replications,
                "config": asdict(config), "datasets": [], "network": []}
    for name, kind, settings in scenarios():
        records, truth = generate_regions(settings, config)
        folder = args.data_output / name
        manifest = export_regions(folder, settings, config, records, truth)
        agreement, analysis = validate_dataset(folder, records, config, contexts)
        repeated = repeat_validation(settings, config, args.replications)
        for filename, digest in manifest["files"].items():
            assert sha256((folder/filename).read_bytes()).hexdigest() == digest
        evidence["datasets"].append({"name": name, "kind": kind, "settings": asdict(settings),
                                     "truth": truth, "data_path": str(folder.resolve()), "agreement": agreement,
                                     "analysis": analysis, "repeated": repeated})
        # Checkpoints preserve finished evidence if a later network node needs attention.
        (args.results_output / "region_validation.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"{name}: {truth['n_districts']} districts, effect={analysis['effect']:+.4f}, "
              f"oracle={truth['oracle_policy_effect_log1p']:+.4f}, coverage={repeated['coverage95_vs_realized_oracle']}", flush=True)
    if args.network:
        folder = args.data_output / "p08_ratio070_equal_rho80"
        evidence["network"] = validate_network(contexts, args.results_output, folder, config)
    old = json.loads((ROOT / "results/policy_validation.json").read_text(encoding="utf-8"))
    evidence["legacy_workbooks_unchanged"] = all(
        sha256((ROOT/path).read_bytes()).hexdigest() == digest for path, digest in old["source_workbook_sha256"].items())
    assert evidence["legacy_workbooks_unchanged"]
    (args.results_output / "region_validation.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{ "scenario": r["name"], "kind": r["kind"], **r["repeated"]} for r in evidence["datasets"]]).to_csv(
        args.results_output / "region_simulation.csv", index=False, encoding="utf-8-sig")
    write_report(evidence, args.results_output)
    print(f"Complete: {args.results_output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
