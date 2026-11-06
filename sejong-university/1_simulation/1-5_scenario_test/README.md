# 1-5 연합학습 시나리오 테스트

계층형 연합학습 시뮬레이터(`hierarchical_fl`)에서 예외 상황 시나리오를 돌려 보고, 기준 환경과 비교해 성능을 얼마나 유지하는지(적응력)와 클라이언트별 성능, 기여도를 평가합니다. 3차년도 보고서 가-(5) 항목이며 패키지는 `scenario_test`입니다. 소스는 [`src/scenario_test`](src/scenario_test), 테스트는 [`tests/modules/test_scenarios.py`](../../tests/modules/test_scenarios.py)에 있습니다.

시나리오는 `ScenarioPlugin`으로 구현되어 있고, 시뮬레이터의 라운드 상태(참여, 지연, 엣지 장애)와 데이터 분할을 바꿉니다. 계층형 시뮬레이터의 플러그인이기 때문에 `--backend kist`를 주면 KIST ClientApp과 전략으로 같은 시나리오가 실행됩니다.

## 시나리오

| 시나리오 | 제안서 구성 | 구현 | 결과 분석 |
|---|---|---|---|
| REF | 기준 환경 | 지연·변동 없음 | P_old |
| S1 네트워크 지연 | 클라이언트마다 다른 지연, 지연을 점진적으로 증가 | 시작 라운드 이후 통신 시간 라운드당 +15%, 10% 확률 5배 스파이크, 적응형 마감시간 | 라운드 가상 시간(업데이트 주기), 지각 클라이언트 수, 정확도 |
| S2 클라이언트 참여 비율 변동 | 주기마다 무작위로 참여 클라이언트 선택 | 라운드별 참여율 U(0.3, 1.0), 참여자 중 10% 학습 중 이탈 | 참여 수, 수렴 속도, 최종 성능 |
| S3 데이터 불균형 | 클라이언트마다 데이터 할당량 상이 | 데이터량 log-normal(σ=1) 분할(옵션: Dirichlet 레이블 편중) | 전체 성능, 클라이언트별 정확도, 기여도 Gini |
| S4 엣지 서버 장애 | (3차 시뮬레이션 환경 예외 처리) | 지정 라운드 동안 엣지 HTTP 503, 매칭에서 제외·재배정 | 장애 구간 성능, 이벤트 로그 |
| S5 내부자 협력 공격 | (3차 보안 고도화와 연계) | `security_sim` 내부자 공모 공격 + 내부자 모니터 | 탐지·성능 |

성능 P는 최근 3라운드의 평균 테스트 정확도입니다. 적응력은 `A_adapt = (1/N) Σ_i P_new,i / P_old,i`로 계산하며, 여기서 i는 데이터셋과 시드별 예외 시나리오이고 P_old는 같은 데이터셋, 같은 시드의 REF 결과입니다.

클라이언트 기여도는 라운드마다 `n_j · max(0, 2q_j − 1)`을 누적한 값입니다. q_j는 엣지 중앙값 업데이트와 방향이 얼마나 일치하는지를 나타내며, 누적값을 정규화한 뒤 Gini 계수를 구합니다. 클라이언트별 정확도는 최종 전역 모델을 각 클라이언트의 검증 데이터로 평가해 최솟값과 표준편차를 봅니다.

## 실행

```bash
python -m scenario_test --clients 12 --edges 3 --rounds 6 --seeds 0,1
python -m scenario_test --scenarios network_delay,participation --set backend-config.dataset=mnist
python -m scenario_test --backend kist --rounds 4
```

```python
from scenario_test.runner import run_scenario, run_suite
from scenario_test.scenarios import ScenarioPlugin
from hierarchical_fl.server import HierarchicalServer

HierarchicalServer({"num-clients": 12, "num-edges": 3}, plugins=[ScenarioPlugin({"scenario": "edge_failure", "edge": 1, "from": 3, "to": 5})]).run()
results, summary = run_suite(["network_delay", "imbalance"], seeds=[0])
```

결과는 `results/scenarios/`에 저장됩니다. `scenario_results.json`에는 시나리오별 라운드 기록, 추적, 기여도, 클라이언트 정확도, 적응력 요약이 들어가고, 함께 `scenario_table.csv`와 `scenario_curves.png`(정확도, 가상 라운드 시간, 집계 참여 수)가 생깁니다. 각 실행의 `runs/<run-id>/hfl.sqlite`는 대시보드에서 열어 볼 수 있습니다.

## 측정 결과

`scripts/run_demo.py` 기본 설정(synthetic 데이터, 3라운드 안팎의 짧은 실행, CPU i9-11900F)으로 `python -m scenario_test --rounds 3 --clients 9`를 실행한 결과입니다(엣지 3, 시드 0). 실행이 짧아 절대값보다는 같은 설정 안의 비교로 보아야 합니다.

| 시나리오 | P | P/P_ref | 클라이언트 최저 정확도 | 기여도 Gini | 가상 시간 합계 | 이벤트 |
|---|---|---|---|---|---|---|
| REF | 0.7797 | 1.000 | 0.902 | 0.012 | 123.1 s | 0 |
| S1 network_delay | 0.7160 | 0.918 | 0.857 | 0.297 | 79.0 s | 0 |
| S2 participation | 0.7902 | 1.013 | 0.895 | 0.238 | 121.1 s | 0 |
| S3 imbalance | 0.8767 | 1.124 | 0.944 | 0.336 | 122.9 s | 0 |
| S4 edge_failure | 0.8073 | 1.035 | 0.887 | 0.016 | 130.6 s | 2 |
| S5 insider | 0.8008 | 1.027 | 0.872 | 0.028 | 123.1 s | 12 |

예외 시나리오 5개가 모두 끝까지 돌았고 적응력은 `A_adapt = 1.024`입니다. S1에서는 적응형 마감시간이 지각 클라이언트를 빼면서 라운드 가상 시간이 줄었고 성능은 낮아졌습니다. 3라운드 실행에서는 P가 초기 라운드 변동에 크게 흔들리기 때문에 S3이 REF보다 높게 나왔습니다. 보고용 수치는 `--rounds 10 --seeds 0,1,2` 이상으로, 실제 데이터셋을 써서 내는 것을 권장합니다.
