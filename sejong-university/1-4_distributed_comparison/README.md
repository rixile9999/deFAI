# 1-4 지역 분산 컴퓨팅과 연합 컴퓨팅 비교

같은 데이터와 같은 평가 지표로 지역 분산 컴퓨팅과 연합 컴퓨팅을 비교하는 환경입니다(3차년도 보고서 가-(4)). 전역·로컬 정확도, 정밀도, 재현율, F1 같은 성능과 함께 원본 데이터 이동량, 모델과 통계의 통신량을 나란히 계산합니다. 패키지는 `defai_addons.compare`([`src/defai_addons/compare`](src/defai_addons/compare))이고 테스트는 [`tests/addons`](../../tests/addons)에 있습니다.

1차년도에는 지역 분산 분석 성능을 평가하는 시뮬레이션을 설계했습니다(데이터 정제·분할, 독립 분석, 중앙 비교, Precision·Recall·F1). 이 설계는 3차년도 비교 환경의 전처리와 평가 절차로 들어갔습니다. 최신 구현은 3차년도에 만든 지역 분산과 연합 컴퓨팅 비교 환경으로 전처리 파이프라인(정제, 정규화, 특징 추출)과 실시간 결과 시각화를 포함하며, 결과 화면은 [1-2](../1-2_fl_visualization/README.md)의 비교 카드를 같이 씁니다.

비교하는 환경과 결과는 다음과 같습니다.

- 지역 분산: 사이트 독립 학습(`ld_independent`), 데이터 재분할 후 동기화(`ld_sync`), 중앙 집중 학습(`ld_central`)
- 연합: Flower 기반 FedAvg·FedProx, 쌍별 마스크 집계, AES-GCM 업데이트 보호, 차분 프라이버시(`fl_fedavg`, `fl_fedavg_dp` 등)
- 전처리: 정제, 정규화, PCA
- 결과: 전역·로컬 정확도, 정밀도, 재현율, Macro F1, 원본 이동량, 모델·통계 통신량, F1 차이의 부트스트랩 신뢰구간

보안과 통신은 시뮬레이션 수준입니다. 사이트는 한 프로세스 안에서 논리적으로만 나뉘고, 서버 간 운영 배포를 자동으로 구성하지는 않습니다.

## 파일

- `src/defai_addons/compare/envs.py`: 지역 분산·연합 환경
- `src/defai_addons/compare/privacy.py`: 업데이트 보호와 차분 프라이버시
- `src/defai_addons/compare/transport.py`: 전송량 기록
- `src/defai_addons/compare/runner.py`: 비교 실행과 결과 정리
- `scripts/run_comparison.py`, `run.py`: MNIST 비교 실험 실행
- `evidence/`: 3차년도 보고서에 쓴 비교 실험 결과

## 실행

```bash
defai-addons compare --config examples/comparison_config.json --out results/comparison.json
python 1_simulation/1-4_distributed_comparison/run.py --help
```

`examples/comparison_config.json`은 동작을 확인하기 위한 작은 설정이라서, 이 결과를 연구 성능 수치로 쓰지는 않습니다.

## KIST 데이터로 비교하기

KIST 쪽 데이터를 사이트별 배열로 넘기면 같은 비교 엔진을 돌릴 수 있습니다. 입력 배열은 복사해서 쓰기 때문에 호출한 쪽 데이터는 바뀌지 않습니다.

```python
from defai_addons.api import compare_arrays
from fl_core.backends import get_backend

backend = get_backend("kist", {"dataset": "mnist-mirror"})
sites = []
for k in range(3):
    x, y = backend.client_data(k, 3)
    xv, yv = backend.client_data(k, 3, "val")
    sites.append({"X": x.flatten(1).numpy(), "y": y.numpy(),
                  "X_test": xv.flatten(1).numpy(), "y_test": yv.numpy()})
xt, yt = backend.test_set()
report = compare_arrays(sites, xt.flatten(1).numpy(), yt.numpy(),
                        config={"epochs": 10, "use_pca": True, "envs": ["ld_independent", "ld_sync", "fl_fedavg"]})
```

`X`, `X_test`는 `[N, D]` 실수 배열이고 `y`, `y_test`는 0부터 시작하는 정수 `[N]` 배열입니다. 모든 사이트의 특징 차원이 같아야 합니다. 비교 엔진은 MLP를 쓰므로 KIST의 CNN 학습 결과를 그대로 재현하지는 않습니다.

별도 프로세스로 넘길 때는 NPZ로 저장합니다(`n_sites`, `X_test`, `y_test`, `site_<i>_<key>`).

```bash
defai-addons compare --config examples/comparison_config.json --data results/sites.npz --out results/site-comparison.json
```

KIST의 `history.json`에는 F1과 원시 예측이 없어서 F1을 역산할 수 없습니다. 정확도와 통신량만 필요하면 `defai_addons.api.read_history`로 읽으면 되고, F1까지 비교하려면 위처럼 평가 데이터를 넘겨 비교를 다시 실행합니다.

반환되는 JSON의 `results[]`에는 `global`, `local`, `local_mean`, `cost`, `privacy`, `history`, `f1_diff_vs_ref`가 들어 있습니다. 원본 데이터가 서버로 가는 환경인지는 `privacy.raw_data_leaves_site`로 구분합니다.

## 검증

- `tests/addons/test_host_integration.py::test_external_sites_use_real_comparison_engine`: 외부 배열로 지역 분산·연합 실행, F1 계산, 입력 불변
- `scripts/check_kist_integration.py`: KIST 데이터로 비교 실행
- 이전 검증 로그: [evidence/launcher_verification.log](evidence/launcher_verification.log), [docs/evidence](../../docs/evidence)
