"""Render the measured evidence as a standalone Korean HTML research report."""
from datetime import datetime
from html import escape
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def table(headers, rows):
    head = "".join(f"<th>{escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>"+"".join(f"<td>{escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    evidence = json.loads((ROOT / "results/policy_validation.json").read_text(encoding="utf-8"))
    dataset_table = table(
        ["자료·날짜", "기존 잘못된 계산", "수정된 연합 추정", "암호화와 중앙집중 차이", "유의성"],
        [[f"{r['dataset']} ({r['n_districts']}개) / {r['config']['date_column']}",
          f"{r['legacy_invalid_effect']:+.6f}", f"{r['plaintext']['effect']:+.6f}",
          f"{r['ckks_effect_abs_error']:.3e}", "광역 수 부족으로 보류"] for r in evidence["comparisons"]])
    simulation_table = table(
        ["설정된 효과(로그 단위)", "시계열 상관", "시군구 수", "반복 수", "편향", "95% 구간 포함률", "5% 검정 기각률"],
        [[r["true_effect_log1p"], r["serial_correlation"], r["district_sizes"], r["replications"],
          f"{r['bias']:+.5f}", f"{100*r['coverage95']:.1f}%", f"{100*r['rejection_rate05']:.1f}%"]
         for r in evidence["simulations"]])
    network_table = table(["전송 방식", "프로세스 수", "광역·시군구 수", "중앙집중 대비 절대오차", "종료 상태"],
                          [[r["mode"], r["processes"], f"{r['n_provinces']} / {r['n_districts']}",
                            f"{r['absolute_error_vs_pooled']:.3e}", "전 노드 정상 종료"] for r in evidence["network"]])
    counter_table = table(["조건", "실제 정책효과", "계산된 전후 차이의 차이", "정책 전 가짜 개입 대비"],
                          [[r["scenario"], r["true_policy_effect"], f"{r['estimated_contrast']:+.4f}",
                            f"{r['pre_placebo']:+.4f}"] for r in evidence["identification_counterexamples"]])
    scope = table(["한 것", "안 한 것"], [
        ["기존 9개·21개 시군구 전체, 발병일·진단일 두 정의 재분석", "실제 정책·환자 자료를 이용한 인과효과 입증"],
        ["중앙집중·평문 계층 합산·CKKS 동형암호 결과 비교", "독립 기관·실제 네트워크에 걸친 운영 검증"],
        ["합성 자료에서 효과·시계열 상관·시군구 수 불균형의 8개 조합 반복 검증", "정책의 파급효과·보고 누락·시간변화 표집편향의 해결"],
        ["광역 단위 불확실성, 사전 가짜 개입 대비, 실패 입력 차단", "개별 통계의 차등프라이버시·악성 단말의 자료 진위 인증"],
        ["시군구 원본 파일의 전후 체크섬 확인", "기존 발표자료·과거 동영상의 모든 수치를 새 결과로 재작성"]])
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y.%m.%d %H:%M KST")
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>3계층 정책효과 분석 보완 연구</title>
<style>body{{font:17px/1.75 'Malgun Gothic',sans-serif;color:#243247;background:#f4f6fa;margin:0}}
main{{max-width:1160px;margin:auto;padding:44px 32px;background:white}}h1,h2{{line-height:1.4;color:#123252}}
h2{{margin-top:42px}}table{{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px}}
th,td{{text-align:left;border-bottom:1px solid #d9e1eb;padding:11px;vertical-align:top}}th{{background:#edf3fa}}
.note{{background:#fff4d8;padding:18px;border-left:5px solid #c99420}}.flow{{display:flex;gap:15px;align-items:center;flex-wrap:wrap}}
.node{{background:#eaf3f6;padding:18px;flex:1;min-width:200px;border-radius:8px}}code{{background:#edf1f6;padding:2px 5px}}
.meta{{color:#66758a;font-size:14px}}a{{color:#075ca2}}p{{max-width:100ch}}</style><main>
<p class="meta">{now} · 연구 단계 · 실전 미적용</p><h1>3계층 정책효과 분석: 집계·추론·암호화 검증</h1>
<p>평균과 표준편차를 더하던 오류를 수정하고, 중앙집중 분석과 동일한 계산이 가능한 3계층 통계 프로토콜을 구현했다.
다만 기존 자료의 정책 적용 광역은 서울 한 곳이다. 수정본은 효과 대비값을 계산하되 일반적인 통계적 유의성은 보류한다.</p>
<div class="note">구현의 정확성과 실제 정책효과의 인과적 식별은 별개다. 과거 시연의 “71.2% 감소·유의”는 검증된 결론으로 사용하지 않는다.</div>
<h2>1. 어떤 질문에 답하는가</h2>
<p>모든 시군구에 2024년 1월 1일~4월 9일의 동일한 달력을 적용한다. 정책 기준일은 1월 31일이다.
관측된 환자가 없는 날짜도 0건으로 포함하고, 기본 날짜는 자료 생성기의 정책 시점과 대응하는 발병일이다.
진단일 분석은 보고 지연에 대한 민감도 분석으로 별도 제시한다. 각 파일의 한 행을 한 관측 환자로 센다.</p>
<p>분석 대상은 <code>y = log(1 + 일별 관측 환자 수)</code>의 평균이다.
정책 전후 변화의 처치군 평균에서 대조군 평균을 뺀다. 시군구는 동등한 가중치를 받으며 환자 수가 많은 지역에 추가 가중하지 않는다.
이 값은 평균 환자 수의 로그나 지역 전체 환자 수의 로그와 다르다. 지수변환해도 “환자 수+1의 기하평균 비율 대비”이므로
원래 환자 수가 몇 % 감소했다고 직접 해석하지 않는다.</p>
<h2>2. 왜 합산이 이제 성립하는가</h2>
<div class="flow"><div class="node"><b>시군구</b><br>자기 자료에서 개수·합·제곱합 계산<br>공개키 암호화</div><b>→</b>
<div class="node"><b>시도</b><br>같은 정의의 암호문만 합산<br>비밀키 없음</div><b>→</b>
<div class="node"><b>중앙</b><br>광역별 집계 복호화<br>평균·분산·정책 대비 재구성</div></div>
<p>각 구간의 <code>(n, S=Σy, Q=Σy²)</code>를 보내면 평균은 <code>S/n</code>, 표본분산은
<code>(Q−S²/n)/(n−1)</code>이다. 이 세 값은 지역 간 덧셈이 가능하다. 기존의 평균·표준편차 자체는 덧셈 가능한 통계가 아니다.
전기간을 정책 전 앞부분·뒷부분·정책 후로 나누고 시군구 개수를 추가해 총 10개 수를 전송한다.
정책 전 두 구간의 대비는 사전 추세 진단에 사용한다. 구간 내 일별 분산은 기술통계이며 정책효과의 표준오차로 사용하지 않는다.</p>
<h2>3. 기존 자료 전체 대조</h2>
<p>표의 효과는 모두 로그 단위다. 기존 열은 잘못된 합산·진단일·0건 날짜 제외를 재현한 값이다.
새 열은 올바른 합산 외에도 날짜·변환·관측기간 정의를 명시했으므로, 두 열의 차이를 전부 합산 오류의 크기로 해석해서는 안 된다.</p>
{dataset_table}<p>두 자료군과 두 날짜 정의 모두에서 수정된 계층형 결과가 중앙집중 계산과 설정한 허용오차 안에서 일치한다.</p>
<h2>4. 불확실성은 광역 단위로 계산한다</h2>
<p>정책이 광역 단위로 적용되므로 시군구 수나 날짜 수를 독립적인 정책 반복 수로 세지 않는다.
시군구별 정책 전후 차이를 절편·처치 여부에 회귀한 결과와 같은 점추정치를 계산하고,
광역 내 잔차 합을 이용한 군집 강건 분산에 소표본 보정(CR1)을 적용한다.
보정계수는 <code>G/(G−1) × (N−1)/(N−2)</code>이며 G는 광역 수, N은 시군구 수이다.
탐색적 검정은 자유도 G−1의 t 분포를 사용한다. 양 군에 최소 2개 광역이 없으면 표준오차·유의확률·신뢰구간을 내지 않는다.
이 최소 조건을 만족해도 적은 광역 수에서는 근사 검정이 부정확할 수 있어 탐색적으로 표시한다.</p>
<p>기존 9개 시군구 자료는 3개 광역, 21개 자료는 8개 광역이지만 처치 광역은 둘 다 1개다.
시군구를 더 늘리는 것만으로 이 제약을 해결할 수 없다.</p>
<h2>5. 추가 모의실험</h2><p>정책효과 0/−0.4 로그 단위, 시계열 상관 0/0.8, 광역별 시군구 수 동일/상이의 8개 조합을 전부 평가했다.
30개 광역 중 15개를 무작위 처치하며 광역 공통 충격과 시군구 잡음을 포함한다.
이는 변환된 결과값을 직접 생성하는 통계 실험으로, 감염 확산이나 환자 발생을 재현하는 모형은 아니다.
구간 포함률은 알려진 효과가 95% 신뢰구간에 들어온 비율, 효과 0에서 기각률은 거짓 양성률이며 효과 −0.4에서는 검정력이다.
500회에서 거짓 양성률 5%의 모의실험 표준오차는 약 0.97%p이다.</p>
{simulation_table}<p>이 표는 해당 생성 조건에서 계산법의 편향과 불확실성 동작을 확인한다. 실제 정책 자료의 식별 가정을 입증하지는 않는다.</p>
<h2>6. 집계가 정확해도 인과 결론이 틀릴 수 있는 반례</h2>
{counter_table}<p>정책효과가 0이어도 처치 지역만의 시간 추세나 정책 후 보고율 변화가 있으면 대비값이 발생한다.
사전 가짜 개입 대비는 차별적 사전 추세를 드러낼 수 있지만 정책 후에만 생기는 보고 변화는 놓친다.
따라서 사전 진단이 0이라는 이유만으로 인과적 타당성을 선언하지 않는다. 지역 간 감염 파급도 별도의 검증 과제다.</p>
<h2>7. 실제 프로세스 통신 검증</h2>{network_table}
<p>평문과 TenSEAL의 CKKS(근사 실수 동형암호)를 별도로 실행했다. 공개키 컨텍스트에 비밀키가 없음을 확인하고,
중간 서버의 복호화 시도가 실패하는 테스트를 통과시킨다. 기준 기간·버전·키 식별자·처치군 불일치와 중복·누락 집계는 거부한다.
시험은 같은 컴퓨터의 별도 프로세스로 진행했으며 운영 환경의 비밀키 접근권한 격리·전송 구간 보안·악성 입력 인증을 증명하지 않는다.
중앙은 광역 집계를 볼 수 있고, 광역에 시군구가 하나면 그 집계가 해당 시군구를 드러낸다. 동형암호만으로 차등프라이버시를 보장하지 않는다.</p>
<h2>8. 범위와 재현</h2>{scope}
<p>수정은 연구용 계산·전송 경로를 완성하는 범위다. 실제 정책에 대한 결론은 기존 합성 자료만으로 닫을 수 없다.
미탐색 영역은 실제 관측 누락, 보고체계 변화, 평행추세 이탈, 감염 파급, 소수 처치 광역을 위한 별도 식별 설계다.</p>
<p>상세 실행법은 프로젝트 README.md에 있으며, 기계 판독 결과는 policy_validation.json,
모의실험 표는 policy_simulation.csv, 통신 로그는 results/network에 저장된다.
기존 발표 문서·영상은 역사 자료다. 현재 결과를 표시하는 영상 생성 코드는 JSON을 읽도록 수정했으며 과거 영상 자체를 새로 검증한 것은 아니다.</p>
<h2>근거 문헌·용어</h2><ul>
<li><a href="https://www.nber.org/papers/w8841">Bertrand·Duflo·Mullainathan: 시계열 상관과 차이의 차이 추론</a></li>
<li><a href="https://www.nber.org/papers/t0312">Conley·Taber: 소수의 정책 변경에서의 추론</a></li>
<li><a href="https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html">statsmodels 군집 강건 공분산 — 독립 구현 대조</a></li>
<li>차이의 차이: 처치군 변화에서 대조군 변화를 뺀 값. CKKS: 실수의 근사 덧셈을 지원하는 동형암호.
CR1: 군집 강건 공분산의 소표본 보정. %p: 비율 사이의 차이. 로그 단위와 %는 서로 다른 단위다.</li></ul>
</main></html>'''
    path = ROOT / "results/policy_research_report.html"
    path.write_text(html, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
