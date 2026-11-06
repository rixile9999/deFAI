# 1-6 다계층 네트워크 비용 최적화

2차년도 개발 항목으로, 3차년도 보고서에서는 가. 시뮬레이션 환경과 연계됩니다. 패키지는 `network_cost`([`src/network_cost`](src/network_cost))이고 테스트는 [`tests/modules/test_network_cost.py`](../../tests/modules/test_network_cost.py)입니다.

클라이언트, 접속 노드(AP), 지역 라우터, 엣지, 클라우드로 이어지는 다계층 DAG 토폴로지를 두고, 연합학습의 실제 Flower 메시지(모델 배포와 업데이트 업로드)가 그 위로 전송된다고 가정합니다. 그리고 최적화 기술을 적용하기 전과 후의 데이터 전송량, 네트워크 비용을 라운드별로 비교합니다. 구현은 ClientApp 미들웨어(전송 계산, 압축)와 Strategy 래퍼(비용 수집)로 되어 있어서 KIST 시스템과 계층형 시뮬레이터 어디에나 붙습니다.

## 최적화 기술

최적화는 `optimizations` 설정에 `edge_compute`, `cache`, `dag`를 넣어 켭니다. 제안서의 엣지 컴퓨팅은 로컬 노드에서 전처리하고 경량화해 업링크 데이터를 최소화하는 기술입니다. `edge_compute`는 업데이트(로컬−전역)에서 상위 k%(`ratio`)만 남기는 희소화와 8비트 양자화를 하고 오차 피드백을 씁니다. 압축했다 복원한 업데이트가 실제 집계에 들어가므로 정확도에 주는 영향까지 측정됩니다.

캐싱은 LRU 캐시로 자주 오가는 데이터와 모델 업데이트의 중복 전송을 막는 기술입니다. `cache`는 모델 배포를 64KB 청크(SHA-1)로 나누고, 하위 노드의 LRU 캐시에 이미 있는 청크는 다시 보내지 않습니다. 같은 AP 아래 있는 여러 클라이언트나 라운드 사이에 변하지 않은 청크가 이 경우에 해당합니다.

DAG 라우팅은 전송 경로를 DAG로 모델링하고 비용 함수로 최적 경로를 고릅니다. `dag`에서 링크 비용은 가격·MB + (지연 + 전송시간) + 혼잡도(라운드 내 누적 부하)이고, 다익스트라로 최소 비용 경로를 찾습니다. 비교 기준은 고정된 기본 경로입니다.

같은 실행 안에서 기준(최적화 없음, 고정 경로) 비용도 함께 계산하므로 `*_base`와 `*_opt`를 바로 비교할 수 있습니다.

## 실행

```bash
python -m network_cost --clients 10 --rounds 5
python -m network_cost --variants baseline,all --mode hier --clients 9 --edges 3
python -m network_cost --backend kist --ratio 0.25
```

```python
from network_cost.plugin import NetworkCostPlugin
plugin = NetworkCostPlugin({"optimizations": ["edge_compute", "cache", "dag"], "ratio": 0.25, "cache-mb": 64})
```

계층형 시뮬레이터에서는 엣지 세션마다 클라이언트와 엣지 사이 구간의 비용을 계산해 보고하고, 전역 서버가 이를 합산합니다. 메시지별 비용은 ClientApp 응답의 MetricRecord(`net-*`)에 실려 서버 쪽 `CostCollector`가 합산합니다. 그래서 KIST의 `flwr run` 배포(`fl_core.kist_apps`)처럼 클라이언트와 서버가 서로 다른 프로세스여도 집계됩니다.

결과 파일은 `results/network_cost/network_cost.json`(변형별 라운드 기록), `network_cost.csv`(정확도, 전송량, 비용, 절감률, 캐시 적중률), `network_cost.png`(라운드별 전송량과 비용)입니다.

## 측정 결과

`python -m network_cost --rounds 3 --clients 8`로 측정했습니다. 모델 크기는 6.3 MB이고 업데이트는 상위 25%와 8비트 양자화를 썼습니다. 설정은 `scripts/run_demo.py` 기본값(synthetic 데이터, 3라운드 안팎의 짧은 실행, CPU i9-11900F)이므로 절대값보다는 같은 설정 안의 비교로 보아야 합니다.

| 변형 | 최종 정확도 | 전송량 (모든 홉) | 절감 | 네트워크 비용 | 절감 | 캐시 적중률 |
|---|---|---|---|---|---|---|
| baseline | 0.6125 | 1218.3 MB | 0% | 348.2 | 0% | - |
| edge_compute | 0.5935 | 796.9 MB | 34.6% | 242.7 | 30.3% | - |
| cache | 0.6125 | 875.6 MB | 28.1% | 300.5 | 13.7% | 0.56 |
| dag | 0.6145 | 1129.5 MB | 7.3% | 229.2 | 34.2% | - |
| all | 0.6025 | 447.8 MB | 63.2% | 125.9 | 63.8% | 0.51 |

캐시와 DAG 경로는 학습 결과를 바꾸지 않아 정확도가 같고, 엣지 압축은 전송량을 줄이는 대신 정확도가 조금 떨어집니다.
