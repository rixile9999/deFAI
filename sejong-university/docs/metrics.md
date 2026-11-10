# 연합학습 성능 평가 지표

16개 지표를 분류 성능, 수렴과 통신, 실제 센서 활용, 적응과 지식 통합, 자원 사용, 프라이버시 및 정보 전달의 평가 목적으로 구성
예측 파일만으로 계산할 수 있는 항목과 별도의 학습 이력, 송수신 기록 또는 대조 실험이 필요한 항목을 구분

## 지표와 입력

단위가 proportion인 값은 0에서 1 사이의 비율이며, 백분율로 표시할 때만 100을 곱함
F1 차이는 두 비율의 차이이며 비율 변화량과 상대 변화율을 구분

| ID | 지표 | 단위 | 공통 API와 필요한 관측 | 목적별 실험 |
|---|---|---|---|---|
| 1 | 정밀도 | proportion | `classification_metrics`의 `macro_precision`, 실제 라벨과 클래스별 예측 확률 | `multimodal` |
| 2 | 재현율 | proportion | `classification_metrics`의 `macro_recall`, 클래스별 실제 표본수와 누락 수 | `multimodal` |
| 3 | F1 | proportion | `classification_metrics`의 `macro_f1`, 클래스별 F1의 평균 | `multimodal` |
| 4 | 수렴 시간 TC | second, round | `convergence_time`, 검증 성능과 누적 측정 시간 및 사전 고정한 목표 | `system` |
| 5 | 통신 효율 | byte | `communication_metrics`, 실제 응용 프레임 크기와 비교 전송량 | `system` |
| 6 | 멀티모달 균일성 MUI | bit, coverage | `multimodal_alignment`, 실제 모달리티별 예측과 센서 가용성, 단독 및 융합 모델의 분류 성능 | `multimodal` |
| 7 | 적응력 | F1, F1-round, round | `adaptation_metrics`, 정상 입력과 변경 입력의 성능 궤적 및 변경 시점 | `multimodal` |
| 8 | 지식 통합도 KID | F1 difference | `knowledge_integration`, 동일 시험 자료에 대한 전역 모델과 독립 로컬 모델 및 노드 제외 모델의 성능 | `multimodal` |
| 9 | 자원 사용 RU | CPU-second, MiB | `resource_usage`, 프로세스 CPU 시간과 측정한 RSS를 별도 입력 | `system` |
| 10 | 프라이버시 | epsilon, delta, attack score | `gaussian_privacy_accountant`, 실제 적용한 Gaussian 기법의 라운드와 잡음 배율, 별도 회원추론 공격 결과 | `privacy` |
| 11 | 데이터 엔트로피 H(D) | bit | `entropy_counts`, 로컬 라벨 또는 사전 고정한 구간별 특징 빈도 | `multimodal` |
| 12 | 공동 정보 | bit | `categorical_joint`, `entropy_counts`, `conditional_mi`, 실제 센서 및 융합 모델의 공동 빈도 | `multimodal` |
| 13 | 전송 정보 손실 | relative L2, MSE, cosine | `update_distortion`, 동일 업데이트의 원본과 실제 수신 복원 배열 | `system` |
| 14 | 정보 복원 IRI | recovered fraction | `recovery_metrics`, 원본과 손상 및 복원 배열, 같은 조건에서의 분류 성능 | `system`, `recovery` |
| 15 | 정보 전송률 | application bit/second | `transmission_rate`, 실제 프레임 바이트와 측정 구간의 소요 시간 | `transport-echo`, `system` |
| 16 | 노드 간 정보 활용 | bit, F1 difference | `conditional_mi`와 `knowledge_integration`, 로컬 및 전역 예측의 공동 빈도와 노드 제외 대조 | `crossnode`, `multimodal` |

정밀도와 재현율은 출력 클래스 전체를 대상으로 계산하고 예측되지 않은 클래스의 분모와 0 값을 보존
HAR의 6개 활동과 Digits의 10개 클래스 모두 같은 API로 처리하며, F1은 클래스별 조화평균을 계산한 뒤 평균
평균 정밀도와 평균 재현율의 조화평균을 macro F1로 대신하지 않음

JSD와 엔트로피 또는 상호 정보량은 그 값만으로 융합이나 지식 통합의 우수성을 판정하지 않음
센서 가용성, 단독 센서 대조, 독립 로컬 모델 및 노드 제외 대조의 분류 성능을 함께 확인
자원 사용은 CPU 시간과 메모리를 더하지 않으며 전송 정보 손실은 라벨 상호 정보량과 구분

## 공통 API

`metric_definitions()`는 ID, 이름, 단위, 평가 목적과 `field`를 반환
`evaluate16`은 실제 관측을 이 `field` 이름에 연결하여 16개 항목으로 반환하는 조립 API
관측 JSON을 입력했다는 사실만으로 실험의 타당성이 자동 검증되는 것은 아니며 원시 자료의 측정 조건과 독립 검산을 함께 보존

```python
import json
import numpy as np
from defai_research.metrics import (
    communication_metrics, convergence_time, evaluate16,
)

# 평가를 위해 보존한 실제 예측과 학습 및 전송 기록
with np.load("predictions.npz", allow_pickle=False) as saved:
    labels = saved["labels"]
    probabilities = saved["probabilities"]
with open("measured_run.json", encoding="utf-8") as stream:
    run = json.load(stream)

observations = {
    "convergence": convergence_time(
        run["validation_macro_f1"],
        run["cumulative_wall_seconds"],
        threshold=0.90,  # 실험 전에 고정한 목표
        streak=3,
    ),
    "communication": communication_metrics(run["frame_bytes"]),
}
rows = evaluate16(
    labels=labels,
    probabilities=probabilities,
    observations=observations,
)
```

이 예제는 정밀도, 재현율, F1과 라벨 엔트로피 및 라벨과 예측의 공동 정보에 더해 실제 수렴 시간과 전송량을 연결
센서 대조, 노드 대조, CPU 및 메모리, 실제 수신 복원과 프라이버시 실험을 수행하지 않았다면 해당 항목은 `not_measured`로 유지
라벨과 예측에서 계산한 공동 정보를 전체 연속 센서 특징의 엔트로피로 해석하지 않으며, 특징 빈도를 측정한 경우 해당 관측을 별도로 전달

누락된 항목의 `value`는 `null`이며 이를 0점 또는 실험 실패로 바꾸지 않음
목표 성능에 도달하지 않은 수렴 측정은 관측 자체가 있으므로 `status=measured`이며 내부 결과의 `reached=false`와 `elapsed_seconds=null`로 표시
복원에서는 원본과 손상 상태의 차이가 없거나 효용 회복의 분모가 충분하지 않으면 회복률을 `null`로 유지하고 해당 상태와 분모를 함께 반환
이와 같은 정의역 문제는 `not_measured`와 구분

```sh
defai-research metrics-schema
defai-research metrics --input predictions.npz --observations observations.json --output results/metrics.json
defai-research metrics --help
```

## 실제 목적별 실험 실행

설치한 환경에서 다음 명령을 실행
기본 생성 위치는 현재 폴더의 `results`이며 `metric-study`의 `--workspace`는 실험 이름 앞에 지정
기존 완료 실험과 새 실행의 출력 위치를 구분하고, 아래 순서는 생성된 자료의 의존 관계에 따라 실행

```sh
# 실제 가속도 및 자이로 자료 준비, 학습과 센서 및 노드 대조, 독립 검산
defai-research metric-study multimodal -- --download --prepare-only
defai-research metric-study multimodal
defai-research metric-study verify-multimodal
defai-research metric-study crossnode
defai-research metric-study cross-verify-multimodal

# 별도 프로세스 Flower 학습, 실제 TCP 프레임과 복원 및 자원 측정
defai-research metric-study system -- --stage protocol
defai-research metric-study system -- --stage main
defai-research metric-study transport-echo
defai-research metric-study verify-system
defai-research metric-study export-system
defai-research metric-study recovery

# 실제 클리핑과 Gaussian 잡음, 고정된 회원추론 공격 및 공개 범위 검산
defai-research metric-study privacy
defai-research metric-study privacy-control
defai-research metric-study verify-privacy
defai-research metric-study export-privacy
```

시스템 파이프라인의 짧은 실행 확인은 다음 명령을 사용
5라운드의 진단 실험이며 전체 평가 결과를 대신하지 않음

```sh
defai-research metric-study --workspace ./metric-workspace system -- --stage smoke
defai-research metric-study --workspace ./metric-workspace verify-system

defai-research metric-study system -- --help
defai-research metric-study multimodal -- --help
defai-research metric-study privacy -- --help
python -m defai_research.metrics.studies.purpose_system_study --help
```

HAR 실험의 `runs/<실행명>/classification_metrics.json`에 클래스별 정밀도, 재현율, F1, 실제 표본수, 예측 표본수와 혼동행렬을 기록
시스템 실험은 실제 수신 프레임, 복원 배열, 모델 및 검증 이력을 남기고 독립 검산에서 이를 다시 계산
학습을 제외한 echo 전송률과 학습 및 집계와 검증을 포함한 pipeline 전송률은 측정 구간이 다르므로 구분하여 보고

## 프라이버시 평가 범위

회계 API는 고정된 클라이언트 집합의 대체 인접성과 실제 클리핑 및 Gaussian 잡음 적용을 전제로 전체 출력 라운드의 이상적 Gaussian 보호예산을 계산
회계 함수를 호출하는 것만으로 기존 그래프나 해시 또는 원본 업데이트에 보호가 부여되지는 않음
잡음이 없는 대조는 유한한 보호예산을 제시하지 않으며, 회원추론 AUC가 낮다는 이유만으로 보호를 입증하지 않음

실제 프라이버시 실험은 사전에 지정한 하나의 공개 후보와 그 출력 범위를 검산
분할 식별자, 잡음 없는 모델, shadow 모델, 다른 비교 모델과 공격 및 효용 진단은 `research_private`에 분리하며 소스 공개 대상에 포함하지 않음
고정밀도 또는 유한 정밀도 구현의 보안 인증, 악성 서버 방어와 WAN 배포 검증을 이 실험의 결과로 주장하지 않음
