# KIST 시스템 통합 가이드

세종대학교 연구 코드(이 저장소)를 KIST 연합학습 시스템([rixile9999/deFAI](https://github.com/rixile9999/deFAI))에 붙이는 방법을 정리했습니다. KIST 커밋 `b21d628`(2026-10-30)을 기준으로 확인했습니다.

## 요약

KIST 저장소 코드는 고치지 않아도 됩니다. 이 저장소를 KIST와 같은 파이썬 환경에 패키지로 설치하면 Flower 1.42의 세 인터페이스를 통해 붙습니다. 서버 쪽 `Strategy`를 감싸는 래퍼, ClientApp 핸들러(`defai.client_app.train`) 앞뒤에서 동작하는 미들웨어, 메시지를 전달하는 `Grid`가 그 셋이고, 여기에 KIST가 남기는 `history.json`을 읽는 기능이 더해집니다.

Flower 배포(`flwr run`)에서는 ServerApp과 ClientApp 경로만 `fl_core.kist_apps`로 바꾸고, 켤 기능은 run config의 `sejong-plugins` 한 줄로 고릅니다. KIST의 설정 검증, 전략 생성, 중앙 평가, `history.json` 기록은 그대로 동작합니다.

13개 항목 중 10개는 KIST 코드를 고치지 않고 붙습니다. 나머지 3개(2-3 MapReduce 집계, 2-4 중복 제거, 3-1 멀티모달)는 따로 실행할 수는 있지만, KIST 시스템 안의 기능으로 넣으려면 KIST 쪽에 코드를 더해야 합니다. 그 방법은 5절에 있습니다.

## 1. 설치

KIST와 이 저장소를 같은 가상환경에 설치합니다.

```bash
git clone https://github.com/rixile9999/deFAI
git clone https://github.com/DVL-Sejong/Federated_Learning
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e "./deFAI[all]"
.venv/Scripts/python -m pip install -e "./Federated_Learning[kist,test]"
```

KIST를 `uv sync --locked`로 쓰고 있다면 그 환경에 `pip install -e ./Federated_Learning`만 추가하면 됩니다.

확인한 호환성은 다음과 같습니다.

| 항목 | KIST | 이 저장소 | 확인 |
|---|---|---|---|
| Flower | `flwr==1.42.0` | `flwr==1.42.0` | 같음 |
| PyTorch | `>=2.6,<2.7` | `>=2.6` | torch 2.6.0과 2.9.1 두 환경에서 테스트 93개 통과 |
| Python | `>=3.11,<3.14` | `>=3.11,<3.14` | 3.12.7에서 확인 |
| 패키지 이름 | `defai` | `fl_core`, `defai_research`, `defai_addons` 외 7개 | 겹치지 않음 |
| 명령 | `defai` | `defai-research`, `defai-addons` | 겹치지 않음 |

이 저장소에도 원래 이름이 `defai`인 모듈이 있었습니다. KIST의 `defai/aggregation.py`, `defai/compression.py`와 이름이 겹쳐 한 환경에 둘 수 없었기 때문에 `defai_research`로 바꿨습니다.

## 2. 연결 확인

설치한 뒤 아래 두 명령을 실행하면 연결 상태를 바로 볼 수 있습니다.

```bash
.venv/Scripts/python Federated_Learning/scripts/check_kist_integration.py
.venv/Scripts/python -m pytest Federated_Learning/tests/modules/test_kist_integration.py Federated_Learning/tests/addons/test_host_integration.py
```

`check_kist_integration.py`는 연결 방식 6가지를 KIST 코드로 실제 실행하고 결과를 `results/kist_integration_check.json`에 남깁니다. 이 컴퓨터(i9-11900F, CPU)에서 얻은 결과는 다음과 같습니다.

| 점검 | 결과 | 시간 |
|---|---|---|
| KIST `run_simulation` 결과를 1-2 시각화로 읽기 | OK, 라운드 기록 3개, HTML 생성 | 수 초 |
| KIST ServerApp·ClientApp + 보안·불균형·네트워크 비용 플러그인 | OK, test_acc 0.932, 공격자 1/1 탐지, 전송량 36% 감소 | 8.3초 |
| KIST 전략·모델 위 연산 그래프 가지치기 | OK, test_acc 0.989, 파라미터 34% 감소 | 12.5초 |
| KIST 백엔드로 계층형 시뮬레이터(Flask 엣지) 실행 | OK, final_acc 0.749 | 12.9초 |
| KIST 데이터·모델 계열로 연합 지식 증류 | OK, 학생 정확도 1.0 | 12.0초 |
| KIST 데이터로 지역 분산/연합 비교와 평가 지표 계산 | OK | 4.1초 |

KIST의 `synthetic` 데이터는 쉬워서 정확도 값 자체는 의미가 없습니다. 연결이 끝까지 돌아가는지 보려는 점검입니다.

## 3. 연결 방식

### 3.1 Flower 배포(`flwr run`)에 플러그인으로 붙이기

`fl_core.kist_apps`의 `server_app`과 `client_app`은 KIST의 `defai.server_app`, `defai.client_app`과 같은 일을 합니다. 서버에서는 `validate_config`, `build_strategy`, 중앙 평가, `history.json` 기록을 KIST 코드가 맡고, 그 사이에서 플러그인이 전략을 감쌉니다. 클라이언트는 KIST의 `defai.client_app.train/evaluate`를 그대로 부르고, 미들웨어가 그 앞뒤에서 동작합니다.

KIST의 `pyproject.toml`을 건드리지 않도록 Flower 앱 설정을 따로 만들어 두었습니다. [examples/kist_flower_app/pyproject.toml](../examples/kist_flower_app/pyproject.toml)은 KIST의 run config를 그대로 복사한 것으로, 컴포넌트 경로와 `sejong-plugins`만 다릅니다.

```toml
[tool.flwr.app.components]
serverapp = "fl_core.kist_apps:server_app"
clientapp = "fl_core.kist_apps:client_app"

[tool.flwr.app.config]
sejong-plugins = '[{"target": "security_sim.plugin:SecurityPlugin", "config": {"attack": "none", "defense": ["hmac", "stat"]}}, {"target": "network_cost.plugin:NetworkCostPlugin", "config": {}}]'
```

```bash
FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1 flwr run Federated_Learning/examples/kist_flower_app --stream --federation-config "num-supernodes=4" --run-config "num-server-rounds=3 fraction-train=1.0"
```

`FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1`은 KIST README와 같은 설정으로, 앱마다 가상환경을 새로 만들지 않고 현재 환경을 쓰게 합니다. Flower 시뮬레이션 엔진은 `pip install "flwr[simulation]==1.42.0"`로 설치합니다.

이 명령은 Ray 시뮬레이션 엔진(`flwr[simulation]==1.42.0`)으로 실제로 실행해 확인했습니다. SuperNode 4개로 3라운드를 돌리면서 `sign_flip` 공격자 1명을 넣었을 때, HMAC 서명은 모두 통과했고 공격자는 매 라운드 통계 탐지에 걸러졌으며(정밀도와 재현율 1.0) 정확도는 1.0이었습니다. 결과는 KIST와 같은 형식의 `history.json`으로 남습니다. 폴더 이름 앞에 `sejong_`이 붙고, 플러그인 결과는 `plugins`, `alerts` 키에 들어갑니다.

KIST 앱 설정을 그대로 쓰고 싶다면 KIST `pyproject.toml`의 `[tool.flwr.app.components]` 두 줄만 바꾸고 `sejong-plugins`를 추가해도 결과는 같습니다. `sejong-plugins`가 없으면 KIST 원래 동작과 똑같고, 환경변수 `SEJONG_PLUGINS`로 지정할 수도 있습니다.

`flwr run`에서 쓸 수 있는 플러그인은 다음 네 가지입니다.

| 플러그인 | 설정 예 | 동작 |
|---|---|---|
| `security_sim.plugin:SecurityPlugin` | `{"attack": "none", "defense": ["cert", "hmac", "stat"]}` | 클라이언트가 업데이트에 서명하고 서버가 인증·이상 탐지 후 집계 |
| `data_imbalance.plugin:ImbalancePlugin` | `{"sampling": "none", "weighting": "size_loss"}` | 데이터량·손실 기반 동적 가중치 |
| `network_cost.plugin:NetworkCostPlugin` | `{"optimizations": ["edge_compute", "cache", "dag"]}` | 업데이트 압축, 다계층 전송 비용 계산 |
| `graph_pruning.federated:GraphPruningPlugin` | `{"start": 2, "every": 2}` | 전역 모델 채널 가지치기 |

공격 시뮬레이션(`attack`)은 실험용이므로 실제 운영에서는 `"attack": "none"`으로 두고 방어만 켭니다. 서명용 비밀값은 `"secret"`로 바꿀 수 있고, 서버와 클라이언트가 같은 값을 써야 합니다. 지금 구현은 시뮬레이션용 공유 비밀 방식이어서 운영용 키 관리를 대신하지 못합니다.

`sampling`(조합 샘플링)을 켜면 KIST의 학습 루프 대신 이 저장소의 학습 핸들러가 KIST 데이터와 모델로 학습합니다. KIST 학습 루프를 그대로 두려면 `sampling: "none"`으로 설정합니다.

### 3.2 파이썬에서 직접 실행

KIST의 `defai.simulation.run_simulation`처럼 한 프로세스 안에서 돌리는 방법으로, 실험하거나 디버깅할 때 편합니다.

```python
from fl_core.backends import get_backend
from fl_core.plugins import run_flat
from security_sim import SecurityPlugin
from data_imbalance.plugin import ImbalancePlugin

backend = get_backend("kist", {"dataset": "mnist-mirror", "partitioner": "dirichlet", "dirichlet-alpha": 0.3})
out = run_flat(backend,
               [SecurityPlugin({"attack": "insider_collusion", "defense": ["monitor", "hmac"]}),
                ImbalancePlugin({"sampling": "none", "weighting": "size_loss"})],
               num_clients=10, rounds=10, strategy="fedprox")
```

`get_backend("kist", overrides)`는 KIST의 기본 run config에 `overrides`를 덮어쓴 뒤 `validate_config`로 검사합니다. `strategy`에는 KIST 전략 이름(`fedavg`, `fedprox`, `scaffold`, `krum` 등)이나 전략 객체를 넘깁니다.

### 3.3 계층형 시뮬레이터에서 KIST 구성요소 쓰기

클라이언트, 엣지, 전역으로 이루어진 3계층 구조에서 KIST ClientApp과 전략을 그대로 씁니다. 엣지 서버는 Flask 프로세스이고 메시지는 Flower `Message`의 protobuf 바이트로 오갑니다.

```bash
python -m hierarchical_fl run --set backend=kist --set edge-strategy=fedprox --set global-strategy=fedavg --set backend-config.dataset=mnist-mirror
python -m hierarchical_fl run --spawn-edges --set backend=kist
python -m scenario_test --backend kist
python -m hierarchical_fl dashboard --results results
```

### 3.4 KIST 결과 파일 읽기

KIST 실행이 끝난 뒤 남은 `history.json`을 읽기만 하므로, 실행 중에는 아무것도 바꾸지 않습니다.

```bash
defai run --strategy fedprox --rounds 5 --clients 4 --dataset synthetic --output results/kist-run
defai-addons visualize results/kist-run/history.json --out results/kist-run.html
defai-addons history results/kist-run/history.json --out results/kist-run-normalized.json
```

KIST가 `save-model=true`로 저장한 `final_model.pt`는 3-2 평가 지표(`defai_research.metrics.evaluate16`)나 2-1 가지치기(`graph_pruning`)에 바로 입력으로 넣을 수 있습니다.

## 4. 항목별 연결 지점

| 항목 | 연결 방식 | KIST에서 쓰는 것 | KIST 수정 |
|---|---|---|---|
| 1-1 계층형 시뮬레이터 | `backend=kist` | `client_app.train/evaluate`, `build_strategy`, `task.build_model`, 데이터 분할 | 없음 |
| 1-2 평가 시각화 | `history.json` 읽기 | `history.json` | 없음 (실시간 연동은 5.4) |
| 1-3 보안 | 전략 래퍼 + 클라이언트 미들웨어 | 모든 전략, `client_app.train` | 없음 |
| 1-4 지역 분산/연합 비교 | 데이터 배열 전달 | `load_client_data`, 테스트셋 | 없음 |
| 1-5 시나리오 테스트 | 계층형 시뮬레이터 플러그인 | 1-1과 같음 | 없음 |
| 1-6 네트워크 비용 | 클라이언트 미들웨어 + 전략 래퍼 | 모든 전략, `client_app.train` | 없음 |
| 2-1 가지치기 | 전략 래퍼(채널 마스크) | 모든 전략, `Net` | 없음 |
| 2-2 지식 증류 | 전략 래퍼 + 증류용 학습 핸들러 | 전략, 데이터 분할, `Net` 계열 | 없음 |
| 2-3 MapReduce 집계 | KIST `FedAvg` 하위 클래스 | `strategies.base.FedAvg` | 전략 이름으로 쓰려면 한 줄 (5.2) |
| 2-4 중복 제거 | 학습 전 데이터 필터 | `load_client_data` | 학습 전 단계를 넣으려면 필요 (5.3) |
| 3-1 멀티모달 | 별도 워크플로 | 없음 | 연합학습에 넣으려면 필요 (5.1) |
| 3-2 평가 지표 | 예측 배열 전달 | 모델 예측, `history.json` | 없음 |
| 3-3 데이터 불균형 | 전략 래퍼 + (선택) 학습 핸들러 | 모든 전략, 데이터 분할 | 없음 |

## 5. KIST 저장소를 수정해야 하는 부분

아래 네 가지는 KIST 쪽에 코드를 더해야 완전히 들어갑니다. 꼭 해야 하는 작업은 아니고, KIST 시스템 안의 기능으로 쓰고 싶을 때 하면 됩니다.

### 5.1 멀티모달 모델(3-1)을 KIST 연합학습에 넣기

KIST의 `defai/task.py`는 모델(`Net`)과 이미지 입력 형식이 고정되어 있습니다. 멀티모달 모델로 연합학습을 하려면 다음 작업이 필요합니다.

1. `DATASETS`에 멀티모달 데이터셋 항목을 추가합니다.
2. `build_model(dataset)`에서 해당 데이터셋이면 `defai_addons.api.create_multimodal_model(...)`을 돌려줍니다.
3. `load_client_data`가 `image`, `text`, `audio`, `mask`, `y` 배열을 담은 배치를 돌려주도록 분기합니다.
4. `train`, `test` 함수에서 배치가 사전이면 `model(batch)`로 부르도록 바꿉니다.

그 전까지는 [3-1 README](../3_heterogeneous_data/3-1_multimodal/README.md)에 있는 별도 워크플로로 씁니다.

### 5.2 MapReduce 집계(2-3)를 KIST 전략 이름으로 등록하기

[examples/kist_mapreduce_strategy.py](../examples/kist_mapreduce_strategy.py)에 `MapReduceFedAvg`가 있습니다. KIST `FedAvg`를 상속하고 `aggregate_updates`만 `defai_research.aggregation.map_reduce`로 바꾼 클래스이며, 그룹은 노드 순서로 나눕니다. KIST 데이터로 실행했을 때 일반 가중 평균과의 최대 차이는 0이었습니다.

```bash
python Federated_Learning/examples/kist_mapreduce_strategy.py --clients 8 --rounds 3 --groups 4
```

`--run-config "strategy='mapreduce'"`로 고를 수 있게 하려면 KIST `defai/strategies/__init__.py`의 `STRATEGIES`에 이 클래스를 추가해야 합니다. KIST의 `config.py`가 import 시점에 전략 목록을 만들기 때문에, 바깥에서 사전에 넣기만 해서는 설정 검증을 통과하지 못합니다.

### 5.3 중복 제거(2-4)를 학습 전 단계로 넣기

`GraphSAGECachedDeduplicator`는 단말 데이터 요약으로 단말 그래프를 학습하고 제거 마스크를 돌려줍니다. 그래프를 학습하려면 모든 단말의 요약이 있어야 하므로, KIST 흐름에 넣으려면 먼저 요약을 모으는 라운드(서버가 클라이언트에 보내는 `query` 메시지)를 두고 받은 마스크를 `load_client_data` 결과에 적용해야 합니다. 지금은 2-3과 2-4의 자체 실험 경로(`defai-research benchmark`)에서 동작합니다.

### 5.4 시각화(1-2)를 KIST 실행 중 실시간으로 연결하기

1-2 대시보드는 `/api/ext/runs/<id>/rounds`로 라운드 이벤트를 받으면 실행 중에도 화면을 갱신합니다. 기존 `MonitoringStrategy`는 Flower 이전 API용이라 KIST 전략에는 바로 쓸 수 없습니다. KIST 전략의 `aggregate_train`이 끝날 때 라운드 지표를 이 API로 POST하는 작은 래퍼를 더하면 됩니다. 계층 구조와 매칭 정보까지 실시간으로 보고 싶다면 1-1 계층형 시뮬레이터를 `backend=kist`로 돌리고 그 대시보드를 쓰는 편이 간단합니다.

## 6. 주의할 점

- `flwr run`으로 실행하면 SuperNode의 노드 ID는 무작위 64비트 값입니다. 보안 플러그인은 서명 주체를 실제 노드 ID로 정하고 파티션 번호는 보고용으로만 씁니다. 처음 구현은 노드 ID가 `파티션 번호 + 1`이라고 가정해서 실제 배포에서 서명이 모두 거부됐고, 이번 정리에서 고쳤습니다(`tests/modules/test_kist_integration.py::test_kist_apps_with_deployment_node_ids`).
- 계층형 시뮬레이터에서는 클라이언트가 라운드마다 다른 엣지에 매칭될 수 있습니다. 클라이언트 상태(`context.state`)는 엣지 프로세스마다 따로 유지되므로, SCAFFOLD나 MOON처럼 클라이언트 상태를 쓰는 KIST 전략은 엣지가 바뀌면 상태가 이어지지 않습니다. FedAvg, FedProx, q-FedAvg와 견고 집계 전략은 영향을 받지 않습니다.
- 가지치기 플러그인은 채널을 0으로 만드는 마스크 방식이라 학습 중에는 모델 크기가 줄지 않습니다. 실제로 작아진 모델은 학습이 끝난 뒤 `final_models()`로 얻습니다.
- 계층형 시뮬레이터와 네트워크 비용 모듈의 지연과 대역폭은 네트워크 모델로 계산한 가상 값입니다. 메시지 바이트 수와 학습은 실제 값입니다.
- 결과 폴더(`results/`), 데이터(`data/`), 모델 파일(`*.pt`)은 두 저장소 모두 Git에서 제외되어 있습니다.

## 7. 문의

담당자는 GitHub 커밋 기준입니다.

- ghk71: 1-1, 1-3, 1-5, 1-6, 2-1, 2-2, 3-3, `fl_core`, 이 문서
- Soobin: 1-2, 1-4, 3-1, `defai_addons` 공통부
- skang132: 2-3, 2-4, 3-2
