# 1-1 계층형 연합학습 시뮬레이터

3차년도 보고서 가-(1) 항목입니다. 클라이언트, 엣지 서버, 중앙 서버로 이루어진 3계층 연합학습을 컴퓨터 한 대에서 돌리거나 여러 프로세스, 여러 컴퓨터에 나눠 실행합니다. 엣지 서버는 Flask REST 서버이고 학습 메시지는 protobuf로 직렬화한 Flower `Message`입니다. 계층마다 집계는 Flower `Strategy`가 맡으며, 내장 전략과 KIST `defai.strategies` 중에서 고를 수 있습니다.

패키지 이름은 `hierarchical_fl`이고 소스는 [`src/hierarchical_fl`](src/hierarchical_fl), 테스트는 [`tests/modules/test_hierarchical.py`](../../tests/modules/test_hierarchical.py)에 있습니다.

## 1~3차년도 기능을 합친 방식

이 모듈의 본체는 3차년도의 Flask 기반 계층적 연합학습 시뮬레이션입니다. 엣지 서버 구성, 서버와 엣지와 클라이언트 사이의 REST 통신, 게임 이론 매칭, 계층 트리·우선순위·통신 상태·매칭 행렬 시각화, FedAvg/FedProx/q-FedAvg 비교가 모두 아래 구조 안에 구현되어 있습니다.

1차년도에 설계한 단일 컴퓨터 시뮬레이션 환경(멀티스레딩·멀티프로세싱 병렬화, asyncio 비동기 업로드)은 실행 방식으로 남았습니다. `transport=local`은 한 프로세스 안에서 Flask test client로 실제 REST 경로를 실행합니다. `client-workers`는 엣지 안에서 클라이언트 학습을 스레드로 병렬화하고, `--spawn-edges`는 엣지마다 프로세스를 따로 띄웁니다. 엣지 호출은 `asyncio.gather`로 비동기 처리합니다.

2차년도의 Flask 기반 멀티 컴퓨팅 환경(Flower 라운드 오케스트레이션, RESTful API, SQLite DB 래퍼, 데이터 비율·정보 엔트로피 산출, 모니터링 대시보드)도 그대로 들어 있습니다. REST API 엣지 서버, SQLite 레지스트리(`hfl.sqlite`), `/clients/stats` 엔드포인트가 계산하는 데이터 비율 R=n_i/N과 엔트로피 H=−Σp log₂p, 대시보드의 R/H 히트맵과 정확도/손실 추적이 여기에 해당합니다. 엣지 1개, 라운드 1회로 설정하면 단일 서버 구조가 됩니다.

## 구조

```text
HierarchicalServer (server.py)
 ├─ Flower Strategy.start(grid=EdgeGrid)              전역 집계 (fedavg | fedprox | qfedavg | fedmedian | krum ... 또는 KIST 전략)
 ├─ RoundHook.configure_train → prepare_round()        참여 상태 → 엣지 헬스체크(503) → 기대 시간·마감 → 게임 이론 매칭
 └─ EdgeGrid (edge_grid.py, flwr.serverapp.Grid)       asyncio로 엣지에 Flower Message 동시 전송
       │ POST /api/v1/sessions/<sid>/messages  (application/x-flower-messages, protobuf)
       ▼
EdgeSession (edge_server.py, Flask)                    edge-rounds회 엣지 집계 (HierFAVG)
 ├─ Flower Strategy (엣지 계층)                         configure_train / aggregate_train
 └─ InProcessGrid → ClientApp 핸들러                    내장 ClientApp 또는 KIST defai.client_app.train
```

게임 이론 매칭은 `matching.py`에 있습니다. 클라이언트 효용은 `u_ij = E(R_ij) − cd_j`, 기대 보상은 `E(R_ij) = 보상_i · q̂_j · P(T_ij ≤ D)`이고, `cd_j`는 데이터 비용과 통신 비용(라운드 중앙값 대비)을 더한 값입니다. 엣지의 품질 추정량은 망각 효과가 있는 지수 평활(EMA) `q̂_j ← β q̂_j + (1−β) q_j`로 갱신하며 `q_j = (1 + cos(업데이트_j, 엣지 중앙값 업데이트))/2`입니다. 엣지 우선순위는 `q̂_j · P(T_ij ≤ D)`입니다. 클라이언트가 제안하는 지연 수락 방식(Gale–Shapley, 엣지 용량 반영)으로 안정 매칭을 구하고 blocking pair 수를 기록합니다. 비교용으로 `static`(최근접) 매칭과 `random` 매칭도 있습니다.

네트워크 모델(`network.py`)은 노드를 단위 정사각형 위에 놓고 지연을 15 ms + 180 ms × 거리로 계산합니다. 경로 대역폭 감쇠와 log-normal 분포의 단말 속도 계수도 반영합니다. 학습은 실제로 하지만 시간은 가상 시간(ms)으로 기록합니다. 마감 정책은 `none | fixed | adaptive`(기대 시간 분위수 × 여유) 중에서 고릅니다.

엣지 헬스체크가 실패하면(HTTP 503) 그 엣지는 매칭에서 빠지고 클라이언트는 다른 엣지에 다시 배정됩니다. 응답 지연(타임아웃)과 클라이언트 중도 이탈은 Flower 오류 응답으로 처리합니다.

SQLite 레지스트리(`registry.py`)에는 runs, clients(데이터 비율, 엔트로피, 역할), edges, rounds, assignments(효용, 우선순위, q̂, 상태), edge_rounds(통신량, 시간), events(보안·장애 로그)가 기록됩니다.

대시보드(`dashboard.py`)에서는 실험 목록을 보고 실행할 수 있습니다. 계층 트리, 매칭 행렬, 엣지 우선순위 rank, 엣지 통신 상태, Accuracy/Loss, 라운드별 통신량과 가상 시간, R/H/레이블 분포 히트맵, 클라이언트 학습 추이와 이상치(training trend / outlier detection), 참여와 예외 처리 표, 이벤트 로그를 보여 줍니다.

## 실행

```bash
python -m hierarchical_fl run --set num-clients=12 --set num-edges=3 --set global-rounds=5
python -m hierarchical_fl run --spawn-edges --set edge-strategy=fedprox
python -m hierarchical_fl run --set backend=kist --set backend-config.dataset=synthetic
python -m hierarchical_fl run --set matching.mode=static --set deadline.policy=adaptive --set backend-config.partitioner=dirichlet
python -m hierarchical_fl dashboard --results results
```

여러 컴퓨터에서 실행할 때는 엣지마다 `python -m hierarchical_fl edge --edge-id 0 --host 0.0.0.0 --port 6101`을 띄우고, 전역 서버에서 `python -m hierarchical_fl run --edges http://host1:6101,http://host2:6101,http://host3:6101`로 연결합니다.

`config.py`의 주요 설정은 `backend`, `backend-config`(dataset, partitioner, dirichlet-alpha, local-epochs, learning-rate ...), `num-clients`, `num-edges`, `edge-capacity`, `global-rounds`, `edge-rounds`, `edge-strategy`, `global-strategy`, `matching.*`, `deadline.*`, `transport`(local|http), `client-workers`, `plugins`입니다.

실행 결과는 `<output-dir>/<run-id>/` 아래에 생깁니다. `hfl.sqlite`, `summary.json`(라운드별 정확도, 손실, 가상 시간, 통신량, 매칭 정보, 플러그인 요약), `network.json`이 저장되고, `--spawn-edges`로 실행했다면 `edge_logs/`도 만들어집니다.

## 엣지 서버 REST API

| 메서드 | 경로 | 내용 |
|---|---|---|
| GET | `/api/v1/health` | 상태, 세션, 요청 수, 송수신 바이트, CPU·메모리 |
| POST | `/api/v1/sessions` | 실험 세션 생성 (백엔드, 클라이언트 수, 엣지 전략, 플러그인) |
| POST | `/api/v1/sessions/<sid>/messages` | Flower Message(TRAIN/EVALUATE) 처리 후 응답 Message 반환 |
| GET | `/api/v1/sessions/<sid>/health?round=r` | 장애 주입 시 503 |
| POST | `/api/v1/sessions/<sid>/fault` | 장애 주입 (down/hang 라운드) |
| GET | `/api/v1/sessions/<sid>/clients/stats` | 클라이언트 데이터 수, 레이블 분포, 데이터 비율, 정보 엔트로피 |
| GET | `/api/v1/sessions/<sid>/history` | 엣지 라운드 보고 |
| DELETE | `/api/v1/sessions/<sid>` | 세션 삭제 |

## KIST 백엔드로 실행

`backend=kist`로 실행하면 각 엣지 서버가 KIST의 `defai.client_app.train/evaluate`로 클라이언트를 학습시키고, 엣지 집계와 전역 집계에는 `defai.strategies.build_strategy`가 만든 KIST 전략을 씁니다. 모델, 데이터 분할, run config 검증도 KIST 코드를 그대로 사용합니다.

```bash
python -m hierarchical_fl run --set backend=kist --set edge-strategy=fedprox --set backend-config.dataset=synthetic
```

## 검증과 측정 결과

`tests/modules/test_hierarchical.py`는 지연 수락 매칭의 안정성(blocking pair 0), 2계층 실행, SQLite 기록, 대시보드 API를 확인합니다. 이와 별도로 in-process(Flask test client) 방식과 HTTP 엣지 프로세스 방식 양쪽에서 내장 백엔드와 KIST 백엔드(KIST ClientApp + KIST FedProx 엣지 집계 + KIST FedAvg 전역 집계)를 실행해 보았습니다.

아래 수치는 `scripts/run_demo.py` 기본 설정에서 얻었습니다. synthetic 데이터로 3라운드 안팎만 돌린 짧은 실행이고 CPU는 i9-11900F입니다. 실행이 짧으므로 절대값보다는 같은 설정 안에서 비교하는 용도로 봐야 합니다.

| 실행 | 전송 방식 | 최종 정확도 | 최근 3R 평균 | 가상 시간 합계 | 전송량 |
|---|---|---|---|---|---|
| 클라이언트 9, 엣지 3, 3라운드, FedAvg | in-process Flask | 0.9065 | 0.778 | 123.1 s | 114.2 MB |
| 같은 설정 | 엣지 3개 별도 프로세스(HTTP) | 0.9050 | 0.776 | 123.1 s | 114.2 MB |

두 전송 방식의 라운드별 결과가 사실상 같으므로 REST 경로가 학습 결과를 바꾸지 않는다고 볼 수 있습니다. 남은 차이는 스레드 수에 따라 부동소수점 누적 순서가 달라져서 생깁니다. KIST 백엔드(엣지 FedProx·q-FedAvg, 전역 FedAvg)도 같은 경로로 실행됩니다.
