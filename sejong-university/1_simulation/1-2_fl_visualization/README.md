# 1-2 계층형 연합학습 평가 시각화

연합학습 알고리즘과 매칭 환경을 평가하는 Flask 대시보드로, 3차년도 보고서 가-(2)에 해당합니다. 모듈 안에 있는 계층형 시뮬레이터를 직접 돌려 볼 수도 있고, KIST 시스템 같은 외부 실행이 보내는 라운드 이벤트나 이미 끝난 결과 파일을 받아 화면에 그릴 수도 있습니다. 코드는 `defai_addons.monitor` 패키지([`src/defai_addons/monitor`](src/defai_addons/monitor))이고 테스트는 [`tests/addons`](../../tests/addons)에 있습니다.

화면에서는 매칭 과정의 학습 추세와 업데이트 이상치, 계층 구조 tree, 우선순위 rank, 서버 통신 상태(pixel-based), 매칭 행렬을 볼 수 있습니다. FedAvg, FedProx, q-FedAvg의 학습 추세와 성능을 나란히 비교하는 화면이 있고, 데이터 분포와 엔트로피, 클라이언트 유사도, 통신 흐름(sankey), CPU·메모리 사용량도 표시됩니다.

1~3차년도 화면은 이 대시보드 하나에 모았습니다. 최신 구현은 3차년도의 매칭·알고리즘 평가 화면(계층 tree, 우선순위 rank, 서버 통신 상태, 매칭 행렬, FedAvg·FedProx·q-FedAvg 비교)입니다. 2차년도의 시뮬레이션 모니터링(training trend, outlier detection, hyperbolic 통신 구조, 데이터 비율·엔트로피 heatmap)은 그대로 살려 같은 대시보드에 넣었습니다. 1차년도에 설계한 단일 컴퓨터의 클라이언트·전역 모델 성능 모니터링(지역 모델 평가, 로그, 자원 사용량)은 자원(CPU·메모리), 실행 기록, 모델 관계 시각화로 남아 있습니다.

Flask 앱과 실행 관리, REST API는 `src/defai_addons/monitor/server.py`에 있습니다. 모듈 자체의 계층형 시뮬레이션은 `src/defai_addons/monitor/simulator.py`, 매칭 알고리즘은 `src/defai_addons/monitor/matching.py`, 학습 추세와 이상치, 유사도 계산은 `src/defai_addons/monitor/analytics.py`가 맡습니다. 외부 Flower 실행에서 이벤트를 보낼 때 쓰는 `MonitoringStrategy`는 `src/defai_addons/monitor/deploy.py`에 있고, 대시보드 화면은 `src/defai_addons/monitor/templates`와 `static`입니다. 폴더의 `scripts/`에는 대시보드 실행, Flower 서버·클라이언트 예제, 외부 이벤트 예제가 있습니다. `run.py`와 `workspace.code-workspace`는 이 폴더만 열어서 실행할 때 쓰는 진입점과 VS Code 작업 공간입니다.

## 실행

저장소 루트에서 `pip install -e .`로 설치한 뒤 실행합니다.

```bash
defai-addons dashboard --port 5055 --store results/addon-monitor
python 1_simulation/1-2_fl_visualization/run.py --port 5055
```

브라우저에서 `http://127.0.0.1:5055`를 엽니다. MNIST는 처음 쓸 때 내려받으며, 캐시 폴더는 `DEFAI_DATA` 환경변수로 바꿀 수 있습니다.

다른 서비스 안에서 대시보드를 띄울 수도 있습니다.

```python
from defai_addons.api import create_dashboard

app = create_dashboard(store_dir="results/addon-monitor")
app.run(host="127.0.0.1", port=5055)
```

대시보드가 제공하는 API는 다음과 같습니다.

| API | 용도 |
|---|---|
| `GET /api/defaults` | 시뮬레이션·비교 설정 조회 |
| `POST /api/runs` | `{"config": {"algorithm": "fedavg", "rounds": 3}}`로 모듈 시뮬레이션 실행 |
| `GET /api/runs/<id>` | 진행 상태와 시각화 데이터 조회 |
| `POST /api/ext/runs` | `config`, `topology`, `clients`로 외부 실행 등록 |
| `POST /api/ext/runs/<id>/rounds` | 라운드 이벤트 전달 |
| `POST /api/ext/runs/<id>/finish` | 외부 실행 종료 |

외부 이벤트를 보내는 예제는 `scripts/ext_client_example.py`와 `src/defai_addons/monitor/deploy.py`에 있습니다. 대시보드를 네트워크에 공개하려면 별도의 운영 서버와 접근 제어가 필요합니다.

## KIST 결과 읽기

KIST 시스템이 실행을 마치고 남기는 `history.json`을 그대로 읽어 HTML로 그립니다. KIST 코드는 바꾸지 않습니다.

```bash
defai run --strategy fedavg --rounds 3 --clients 4 --dataset synthetic --output results/kist-fedavg
defai-addons visualize results/kist-fedavg/history.json --out results/kist-fedavg.html
```

```python
from defai_addons.api import read_history, render_histories

record = read_history("results/kist-fedavg/history.json")
render_histories(["results/kist-fedavg/history.json"], "results/kist-fedavg.html")
```

HTML에는 라운드별 전역 정확도, 표본 수로 가중한 클라이언트 정확도, 누적 통신량(MiB), 최종 결과표, 입력 설정과 원본 해시가 들어갑니다. KIST의 `history.json`에는 매칭·계층 정보와 F1이 없기 때문에 tree나 매칭 행렬은 그리지 않으며, `eval_acc`를 F1로 바꿔 쓰지도 않습니다. 실행 중에 실시간으로 보려면 라운드 이벤트를 `/api/ext/...`로 보내야 합니다.

`MonitoringStrategy`는 Flower의 이전 Strategy API(`aggregate_fit`)용이어서, KIST의 Message API 전략(`configure_train`/`aggregate_train`)에 바로 끼울 수 있는 어댑터가 아닙니다. KIST를 실행하면서 계층 정보까지 보고 싶다면 [1-1 계층형 시뮬레이터](../1-1_hierarchical_fl/README.md)를 `backend=kist`로 돌리고 그 대시보드를 쓰면 됩니다.

## 검증

- `tests/addons/test_host_integration.py`: KIST `history.json` 변환, 누락 지표 처리, 원본 파일 불변
- `tests/addons/check_workspace_layout.py`: 패키지 위치, Flask 페이지와 API 응답
- `tests/addons/test_dashboard_metrics.js`: 비교 카드의 지표 렌더링(`node`로 실행)
- 정리 이전 검증 로그: [docs/evidence](../../docs/evidence)
