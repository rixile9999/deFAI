# 3계층 정책효과 분석 — 2026-10-16 수정본

**연구 단계·실전 미적용.** 평균·표준편차를 더하던 기존 프로토콜을 폐기하고,
공통 달력의 개수·합·제곱합을 암호화 합산하는 v2 프로토콜로 교체했다.
기존 시연 문서·동영상의 “71.2% 감소·통계적으로 유의”는 검증된 결과가 아니다.
현재 수치와 한계의 정본은 `results/policy_research_report.html`과 `results/policy_validation.json`이다.
지역 확장 추가 연구는 `results/expanded_regions/region_expansion_report.html`과
`results/expanded_regions/region_validation.json`을 함께 확인한다.
옛 실행 코드·README 등은 `archive/statistics_before_20261016`에 보존했다.

## 정의와 해석

- 달력: 2024-01-01~2024-04-09(양 끝 포함), 정책일 2024-01-31. 모든 시군구에 동일하게 적용한다.
- 기본 날짜는 **발병일**이다. 진단일은 `--date-column 진단일`로 별도 분석한다.
- 결과값: `log(1 + 일별 관측 환자 수)`. 환자가 없는 날도 0건으로 포함한다.
- 목표값: 시군구 동등 가중 정책 전후 변화의 처치군 평균 − 대조군 평균.
  이는 전체 환자 수의 변화나 로그 평균 환자 수의 변화와 다르다.
- `100 * expm1(effect)`는 **환자 수+1의 기하평균 비율 대비**이며, 환자 수 감소율로 단정하지 않는다.
- 완전한 보고기간을 전제로 행이 없는 날을 0건으로 해석한다. 실제 자료의 미보고 날짜를 0건으로 채우면 안 된다.
- 기존 합성 시연은 서울=처치, 나머지=대조다. 실제 연구는 지정된 정책 배정에 맞춰
  시군구의 `--treated 0/1`과 광역의 `--treated`를 명시해야 한다.

## 계산과 불확실성

시군구는 정책 전 앞부분·뒷부분·정책 후 각 구간에서 `(n, Σy, Σy²)`와 시군구 수를 계산한다(총 10개 실수).
시도는 공개키로 암호문 덧셈만 수행한다. 중앙만 광역 집계를 복호화한다.
중앙은 `mean=Σy/n`, `sample variance=(Σy²-(Σy)²/n)/(n-1)`로 기술통계를 복원한다.
날짜별 기술분산은 정책효과의 표준오차로 쓰지 않는다.

정책효과 불확실성은 시군구별 전후 변화 회귀의 **광역 군집 강건 분산(CR1)**으로 계산한다.
G개 광역·N개 시군구에 대해 `G/(G-1) * (N-1)/(N-2)` 보정, t 분포 자유도 G-1을 사용한다.
양 군에 최소 2개 광역이 없으면 표준오차·유의확률·신뢰구간을 **보류**한다.
조건을 만족해도 적은 광역 수의 추론은 탐색적이다. 기존 9개·21개 자료는 처치 광역이 한 곳이므로 보류 대상이다.
정책 전 두 구간의 가짜 개입 대비도 출력하지만, 그 값이 작다고 평행추세를 입증하는 것은 아니다.

인과 해석에는 미처치 잠재 결과의 평행추세, 선행 반응 부재, 지역 간 파급 부재,
정책 전후 보고체계의 안정성이 별도로 필요하다. 현재 합성 자료는 실제 정책효과의 증거가 아니다.
표집 편향 보정은 본 프로토콜에 통합하지 않았다. 시간에 따라 달라지는 누락은 별도 과제다.

## 환경과 재현

프로젝트 루트에서 PowerShell로 실행한다. 전역 Python 환경은 수정하지 않았다.

```powershell
uv venv --python 3.11 .venv-policy-research
uv pip install --python .venv-policy-research\Scripts\python.exe -r requirements-policy-research.lock.txt
$env:PYTHONUTF8='1'
.\.venv-policy-research\Scripts\python.exe -m pytest tests/test_policy_stats.py -q
.\.venv-policy-research\Scripts\python.exe run_policy_validation.py --replications 500 --network
.\.venv-policy-research\Scripts\python.exe write_policy_report.py
```

검증은 9개·21개 시군구의 모든 원본 파일, 발병일·진단일 두 정의를 사용한다.
중앙집중·계층형 평문·CKKS 동형암호 계산을 대조하고, 원본 파일 SHA256 불변을 확인한다.
모의실험은 알려진 효과 0/−0.4, 시계열 상관 0/0.8, 동일/상이한 시군구 수의 8개 조합을 각 500회 평가한다.
30개 광역 중 15개를 무작위 처치하는 **변환된 결과값의 통계 실험**이며 감염 확산 모형은 아니다.
`--network`는 9개 시군구 전체에 대해 평문·암호화 각각 총 13개 프로세스를 실행한다.
테스트 키는 임시 디렉토리에 생성 후 정리하며 기존 `keys`를 덮어쓰지 않는다.

빠른 데이터 점검:

```powershell
.\.venv-policy-research\Scripts\python.exe verify_hierarchical_data.py --data-dir data/districts21
```

## 직접 실행: 서울 3개 구 + 경기 3개 시

`setup_hierarchical.py`는 새 키를 생성하여 **기존 키를 덮어쓴다**. 키가 없다면 모든 노드를 시작하기 전에 한 번 실행한다.
각 명령은 별도 터미널에서 실행하며 중앙 → 광역 → 시군구 순서로 시작한다.

```powershell
.\.venv-policy-research\Scripts\python.exe run_tier0.py --min-provinces 2
.\.venv-policy-research\Scripts\python.exe run_tier1.py --port 9001 --region Seoul --treated --min-districts 3
.\.venv-policy-research\Scripts\python.exe run_tier1.py --port 9002 --region Gyeonggi --min-districts 3
.\.venv-policy-research\Scripts\python.exe run_tier2.py --data data/districts/seoul_gangnam.xlsx --target 127.0.0.1:9001
.\.venv-policy-research\Scripts\python.exe run_tier2.py --data data/districts/seoul_seocho.xlsx --target 127.0.0.1:9001
.\.venv-policy-research\Scripts\python.exe run_tier2.py --data data/districts/seoul_jongno.xlsx --target 127.0.0.1:9001
.\.venv-policy-research\Scripts\python.exe run_tier2.py --data data/districts/gyeonggi_suwon.xlsx --target 127.0.0.1:9002
.\.venv-policy-research\Scripts\python.exe run_tier2.py --data data/districts/gyeonggi_seongnam.xlsx --target 127.0.0.1:9002
.\.venv-policy-research\Scripts\python.exe run_tier2.py --data data/districts/gyeonggi_yongin.xlsx --target 127.0.0.1:9002
```

평문 시험은 **모든 계층**에 `--plaintext`를 붙인다. 기간 변경 시 모든 계층에
`--start`, `--policy`, `--end`, `--date-column`을 동일하게 전달한다.
`--min-provinces`와 `--min-districts`는 계획된 정확한 참여 수다. 누락·중복·다른 기간·다른 키·처치군 혼합은 중단한다.
구형 7개 수 전송 형식은 호환하지 않는다. `--key-dir`로 다른 키 디렉토리를 지정할 수 있다.
직접 실행 결과는 `results/latest_policy_analysis.json`에 기록된다.

## 산출물과 한계

| 한 것 | 안 한 것 |
|---|---|
| 충분통계 합산과 독립적인 중앙집중 계산의 일치 확인 | 실제 정책효과의 인과적 식별 |
| 15개 회귀·경계조건·암호화 테스트 | 실데이터 관측누락·보고율 변화 보정 |
| 8개 조합 × 500회 모의실험 | 소수 처치 광역의 별도 준실험 설계 |
| 실제 3계층 프로세스의 평문·암호화 통신 검증 | 서로 다른 기관·호스트에서의 배포 |
| 기존 데이터·키 보존, 실패 입력 차단 | 악성 입력 인증·차등프라이버시 |

- `results/policy_validation.json`: 전체 측정 결과, 원본 체크섬, 실행 조건
- `results/policy_simulation.csv`: 모든 모의실험 조합 결과
- `results/network/{plaintext,ckks}`: 별도 프로세스 실행 로그와 분석 JSON
- `results/policy_research_report.html`: 방법·결과·한계 설명
- `requirements-policy-research.lock.txt`: 검증에 사용한 Python 패키지 버전

중앙은 광역별 집계를 열람한다. 광역에 시군구가 하나면 그 집계가 개별 시군구를 드러낸다.
같은 PC의 프로세스 시험은 파일 접근권한 분리를 증명하지 않는다.
운영 시 비밀키 격리·전송 구간 보안·신원 인증은 별도로 구성해야 한다.

과거 `.md` 시연 자료와 `demo1_hierarchical.mp4`는 역사 자료다.
현재 영상 생성 코드는 JSON 결과를 읽으며 결과가 없으면 실패한다. 과거 영상은 새 결과로 간주하지 않는다.
영상 생성은 선택 사항으로 matplotlib·Pillow·FFmpeg가 추가로 필요하다.

문헌: [Bertrand·Duflo·Mullainathan](https://www.nber.org/papers/w8841),
[Conley·Taber](https://www.nber.org/papers/t0312),
[statsmodels 군집 공분산](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html).

## 지역 확장 생성기와 추가 검증

실제 관측 지역을 추가한 것이 아니라 기존 포아송 발생·표집 누락·주간 패턴·진단 지연 생성기를
**가상 광역 다중 처치 실험**으로 확장했다. 과거 XLSX 환자 목록은 그대로 보존한다.
추가 자료는 환자별 속성을 쓰지 않는 분석에 맞춰 같은 발생·표집 과정의 일별 집계 CSV로 저장한다.

기본 자료는 `data/expanded_regions`의 32개 가상 광역·96개 시군구다.
16개 광역을 무작위 처치하고 16개를 대조로 배정했으며, 배정은 CSV의 `treated`와 `manifest.json`에 명시한다.
서울이라는 이름으로 처치를 추측하지 않는다. CSV의 `record_format=daily_counts_v1`은
환자 목록과 구별되며 `cases`는 발병일 관측수, `diagnosed_cases`는 진단일 관측수다.
집계 파일은 모든 달력 날짜를 명시해야 하며 빠진 날짜를 자동으로 0건 처리하지 않는다.

추가 생성 예시(기존 출력 폴더는 덮어쓰지 않으므로 새 경로를 지정한다):

```powershell
.\.venv-policy-research\Scripts\python.exe data_gen_hierarchical.py --expanded --n-provinces 64 --n-treated 32 --output data/regions64_new --seed 100
.\.venv-policy-research\Scripts\python.exe verify_hierarchical_data.py --data-dir data/expanded_regions
```

선택 옵션:

| 옵션 | 의미 |
|---|---|
| `--n-provinces`, `--n-treated` | 광역 수와 무작위 처치 광역 수 |
| `--districts-per-province` | 기본 광역당 시군구 수 |
| `--unequal-districts` | 광역별 시군구 수를 1~기본 수의 2배로 다르게 생성 |
| `--policy-ratio` | 정책 시행 완료 후 잠재 발생률 배율, 0 초과 1 이하 |
| `--ramp-days` | 정책효과가 완전히 적용될 때까지 걸리는 날 수 |
| `--base-cases`, `--baseline-sigma` | 기본 발생률과 지역 간 규모 차이 |
| `--serial-correlation`, `--province-shock-sd` | 광역 공통 충격의 시계열 상관과 크기 |
| `--differential-trend` | 처치 지역에만 추가되는 추세(식별 위반 실험) |
| `--reporting-ratio` | 처치 지역의 정책 후 보고율 배율(보고 변화 실험) |
| `--seed` | 재현 가능한 난수 초기값 |
| `--start`, `--policy`, `--end` | 공통 달력과 정책 시점 |

전체 추가 검증:

```powershell
.\.venv-policy-research\Scripts\python.exe run_region_validation.py --replications 200 --network
```

이 명령은 보호 목적으로 기존 출력 폴더가 있으면 중단한다. 다시 실행할 때는
`--data-output data/expanded_validation_run2 --results-output results/expanded_regions_run2`처럼 새 경로를 준다.
평가 범위는 다음 **22개 조건**이며, 다른 모수의 전수 탐색을 뜻하지 않는다.

- 주 행렬 16개: 광역 8/32 × 정책 배율 1/0.7 × 시군구 수 동일/상이 × 시계열 상관 0/0.8.
- 광역 64개의 효과 없음/효과 있음 2개.
- 처치 광역 1개, 낮은 발생수, 차별적 추세, 정책 후 보고율 변화 4개.

각 조건을 200개 난수 초기값으로 **발생·관측 자료부터 재생성**한다(총 4,400회).
한 번 생성해 저장한 22개 자료는 발병일·진단일 두 기준으로 모두 중앙집중·평문·동형암호 계산을 대조한다.
유의성 계산이 가능한 자료는 표준오차도 독립적인 statsmodels 군집 회귀와 비교한다.
`--network`는 8개 광역·24개 시군구·중앙의 **33개 실제 프로세스**를 평문·암호화 각각 실행한다.
32·64개 광역 전체는 암호화 전략을 실행하지만 모든 노드를 동시에 별도 프로세스로 띄우지는 않는다.

### 정답 기준과 한계

생성기의 정책 배율 0.7과 log1p 분석의 참값은 같지 않다. 표집·저발생수·7일 정책 경사가 있기 때문이다.
합성 검증은 동일한 발생·표집에서 만든 정책 유무 쌍을 비교해
**안정된 보고율에서 실현된 처치 시군구 사후 log1p 정책 대비**를 계산한다.
`counterfactual_no_policy_cases`와 `policy_only_cases`는 이 정답 대조에만 사용하며 실제 분석 코드는 읽지 않는다.
신뢰구간 포함률은 이 실현 대비에 대한 값이며 실제 정책의 일반적인 인과 타당성 보장과 구분한다.
진단일 분석의 경우에는 발병일용 정답 기준을 적용하지 않고 계산 일치·민감도만 평가한다.

낮은 광역 수에서는 군집 추론이 잘 보정되지 않을 수 있다. 보고율 변화·차별적 추세는
지역 수를 늘려도 정책효과와 혼동된다. 실제 감염 파급·다른 시행일·비무작위 배정은 이번 범위 밖이다.

| 한 것 | 안 한 것 |
|---|---|
| 가상 광역 8·32·64개와 여러 처치·대조 광역 생성 | 실제 국내 행정구역 자료의 추가 확보 |
| 22개 조건·두 날짜 정의의 계산 대조와 4,400회 재생성 | 모든 지역 수·효과 크기·보고 과정 탐색 |
| 33개 프로세스의 평문·암호화 통신 시험 | 32·64개 광역의 전체 동시 부하 시험 |
| 경계 조건·보고 변화·추세 위반의 결과 공개 | 식별 가정 위반 자동 보정 |

추가 산출물:

- `data/expanded_regions`: 기본 32개 광역 자료와 설정·배정·체크섬.
- `data/expanded_validation`: 22개 조건별 시군구 CSV와 manifest.json.
- `results/expanded_regions/region_expansion_report.html`: 설명·결과 보고서.
- `results/expanded_regions/region_validation.json`: 전체 조건·계산 일치·반복 검증·통신 결과.
- `results/expanded_regions/region_simulation.csv`: 22개 조건의 반복 검증 요약.
- `results/expanded_regions/network`: 확장 자료의 실제 통신 로그.
