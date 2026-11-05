# 세종대학교 deFAI 연합학습 연구 코드

IITP 「분산형 데이터 연합컴퓨팅 기술(deFAI) 개발 및 실증」 과제에서 세종대학교가 1~3차년도에 개발한 코드를 모은 저장소입니다. 폴더는 3차년도 연차보고서에 나오는 세종대학교 연구 항목 순서대로 나눴고, 차년도별 복사본은 따로 두지 않았습니다. 후속 차년도에서 발전한 기능은 최신 구현만 남겼고, 대체되지 않은 이전 기능은 그대로 두었습니다.

전체 코드는 파이썬 환경 하나에 함께 설치되며 KIST 연합학습 시스템([rixile9999/deFAI](https://github.com/rixile9999/deFAI))과 같은 Flower 1.42 위에서 돌아갑니다. KIST 시스템에 붙이는 방법은 [docs/KIST_통합_가이드.md](docs/KIST_통합_가이드.md)에 따로 정리했습니다.

## 보고서 항목과 폴더

### 가. 단말/계층형 연합 학습 실험을 위한 시뮬레이션 환경 구축 (`1_simulation/`)

| 보고서 항목 (3차년도) | 폴더 | 파이썬 패키지 | 담당 |
|---|---|---|---|
| (1) Flask 기반 클라이언트–엣지–전역 서버 3계층 연합학습 | [1-1_hierarchical_fl](1_simulation/1-1_hierarchical_fl/README.md) | `hierarchical_fl` | ghk71 |
| (2) 계층적 연합학습 및 매칭 환경 평가 시각화 | [1-2_fl_visualization](1_simulation/1-2_fl_visualization/README.md) | `defai_addons.monitor` | Soobin |
| (3) 보안 시뮬레이션 환경 고도화 | [1-3_security](1_simulation/1-3_security/README.md) | `security_sim` | ghk71 |
| (4) 지역 분산 컴퓨팅과 연합 컴퓨팅 비교 | [1-4_distributed_comparison](1_simulation/1-4_distributed_comparison/README.md) | `defai_addons.compare` | Soobin |
| (5) 연합학습 시나리오 테스트 | [1-5_scenario_test](1_simulation/1-5_scenario_test/README.md) | `scenario_test` | ghk71 |
| (2차년도) 다계층 네트워크 비용 최적화 | [1-6_network_cost](1_simulation/1-6_network_cost/README.md) | `network_cost` | ghk71 |

### 나. 분석 속도 향상을 위한 다계층 분산 연합 분석 기술 (`2_analysis_speed/`)

| 보고서 항목 (3차년도) | 폴더 | 파이썬 패키지 | 담당 |
|---|---|---|---|
| (1) 연산 그래프 기반 가지치기와 임계값 동적 조정 | [2-1_graph_pruning](2_analysis_speed/2-1_graph_pruning/README.md) | `graph_pruning` | ghk71 |
| (2) 교사–학생 구조와 중간층 매칭 지식 증류 | [2-2_knowledge_distillation](2_analysis_speed/2-2_knowledge_distillation/README.md) | `layer_kd` | ghk71 |
| (3) MapReduce 기반 다계층 연합 분석 | [2-3_mapreduce_analysis](2_analysis_speed/2-3_mapreduce_analysis/README.md) | `defai_research` | skang132 |
| (4) 그래프 기반 동적 클러스터링 중복 제거 | [2-4_graph_deduplication](2_analysis_speed/2-4_graph_deduplication/README.md) | `defai_research.federated` | skang132 |

### 다. 단말/기관 간 불균일 데이터를 고려한 학습 모델 정확도 향상 기술 (`3_heterogeneous_data/`)

| 보고서 항목 (3차년도) | 폴더 | 파이썬 패키지 | 담당 |
|---|---|---|---|
| (1) 트랜스포머 기반 멀티모달 학습 모델 | [3-1_multimodal](3_heterogeneous_data/3-1_multimodal/README.md) | `defai_addons.multimodal` | Soobin |
| (2) 연합학습 알고리즘 성능 평가 지표 | [3-2_fl_metrics](3_heterogeneous_data/3-2_fl_metrics/README.md) | `defai_research.metrics` | skang132 |
| (1차년도) 데이터 불균형 해결 | [3-3_data_imbalance](3_heterogeneous_data/3-3_data_imbalance/README.md) | `data_imbalance` | ghk71 |

1-6과 3-3은 3차년도 보고서 항목에 없는 이전 차년도 개발 항목입니다. 후속 항목으로 대체되지 않았기 때문에 성격이 가까운 장 아래에 넣었습니다. 1~3차년도 개발 항목이 각각 어느 폴더로 들어갔는지는 [docs/개발항목_통합정리.md](docs/개발항목_통합정리.md)에서 볼 수 있습니다.

항목 폴더마다 `README.md`에 설명과 실행 방법이 있고 소스는 `src/` 아래에 있습니다. 실행 스크립트는 `scripts/`나 `examples/`에, 이전 실험 기록은 `evidence/`에 두었습니다.

```text
Federated_Learning/
├── 1_simulation/            가. 시뮬레이션 환경 (6개 항목)
├── 2_analysis_speed/        나. 분석 속도 향상 (4개 항목)
├── 3_heterogeneous_data/    다. 불균일 데이터 (3개 항목)
├── common/                  여러 항목이 함께 쓰는 코드 (fl_core, defai_addons 공통부)
├── examples/                KIST 시스템 결과를 읽는 예제
├── scripts/                 데모 실행, KIST 연동 점검, 소스 묶음 생성
├── tests/                   modules/ · addons/ · research/ 테스트 (93개)
└── docs/                    KIST 통합 가이드, 개발 항목 정리, 지표 설명, 이전 검증 기록
```

## 설치

Python 3.11~3.13에서 동작하며, Windows 11과 Python 3.12.7, CPU 환경에서 검증했습니다.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[kist,test]"
```

소스 폴더 이름과 패키지 이름이 달라서 저장소 루트에서 한 번 설치(`-e`)해야 합니다. 그러면 모든 패키지와 함께 `defai-research`, `defai-addons` 두 명령이 등록됩니다. KIST 시스템과 같이 쓰려면 같은 환경에 KIST `defai`도 설치하는데, 순서는 [통합 가이드](docs/KIST_통합_가이드.md#1-설치)에 있습니다.

## 동작 확인

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_demo.py
.venv/Scripts/python scripts/check_kist_integration.py
.venv/Scripts/python -m hierarchical_fl dashboard --results results/demo
```

`pytest`는 세 팀의 테스트 93개를 실행하며 2분 30초쯤 걸립니다. KIST `defai`가 없는 환경에서는 KIST 연동 테스트 4개를 건너뜁니다.

`run_demo.py`는 1-1, 1-3, 1-5, 1-6, 2-1, 2-2, 3-3 항목의 실행 명령을 짧은 설정으로 차례로 돌리고 결과를 `results/demo/`에 저장합니다. 30분 정도 걸립니다. `check_kist_integration.py`는 KIST `defai`가 설치된 환경에서 연결 방식 6가지를 실제로 실행해 보며, 1분 정도 걸립니다. 마지막 명령으로 띄운 대시보드(`http://127.0.0.1:5000`)에서는 계층 구조, 매칭 행렬, 통신 상태, 정확도, 보안 이벤트를 볼 수 있습니다.

1-2, 1-4, 3-1 항목은 `defai-addons` 명령으로, 2-3, 2-4, 3-2 항목은 `defai-research` 명령으로 실행합니다. 구체적인 명령은 각 항목 README에 있습니다.

## 패키지 구성

`fl_core`에는 Flower Grid와 전략 래퍼, 플러그인 인터페이스, 내장/KIST 백엔드, KIST용 ServerApp과 ClientApp이 들어 있습니다. 1-1, 1-3, 1-5, 1-6, 2-1, 2-2, 3-3은 항목마다 패키지가 하나씩 있습니다(`hierarchical_fl`, `security_sim`, `scenario_test`, `network_cost`, `graph_pruning`, `layer_kd`, `data_imbalance`). 1-2, 1-4, 3-1은 `defai_addons`에 함께 들어 있고 그 공통부는 `common/src/defai_addons`에 있습니다. 2-3, 2-4, 3-2는 `defai_research` 패키지입니다.

`defai_research`의 원래 이름은 `defai`였습니다. KIST 시스템의 패키지 이름도 `defai`이고 `aggregation.py`, `compression`처럼 이름이 같은 모듈까지 있어서, 두 코드를 한 환경에 설치하려고 이름을 바꿨습니다. 명령 이름도 그에 맞춰 `defai`에서 `defai-research`가 되었습니다.

## 문서

- [docs/KIST_통합_가이드.md](docs/KIST_통합_가이드.md): KIST 시스템에 붙이는 방법, 연결 지점, 점검 절차
- [docs/개발항목_통합정리.md](docs/개발항목_통합정리.md): 1~3차년도 개발 항목을 어떤 폴더로 통합했는지
- [docs/metrics.md](docs/metrics.md): 3-2 평가 지표 16개의 정의와 실험 순서
- [docs/history/](docs/history/): 정리 이전에 각 팀이 남긴 범위 문서와 검증 기록
- [docs/evidence/](docs/evidence/): 1-2, 1-4, 3-1 항목의 이전 설치 및 통합 검증 로그

## 라이선스

MIT License ([LICENSE](LICENSE))
