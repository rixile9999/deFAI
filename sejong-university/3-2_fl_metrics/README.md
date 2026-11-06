# 3-2 연합학습 알고리즘 성능 평가 지표

연합학습 성능 평가 지표 16개의 정의와 단위, 계산을 API 하나로 묶었습니다. 3차년도 보고서 다-(2) 항목으로, 전통 지표와 연합학습 특화 지표, 정보량과 불확실성, 정보 손실과 복원, 프라이버시를 다룹니다. 패키지는 `defai_research.metrics`([`src/defai_research/metrics`](src/defai_research/metrics))이고 테스트는 [`tests/research/test_metrics.py`](../../tests/research/test_metrics.py)입니다.

예측만 있으면 계산되는 지표와 별도 실험이나 송수신 기록이 있어야 하는 지표를 나눠 두었고, 관측하지 않은 항목은 `not_measured`로 남깁니다. 분모가 정의되지 않을 때 임의의 숫자로 채우지 않습니다.

16개 지표는 다음과 같이 묶입니다.

- 정밀도, 재현율, F1
- 수렴 시간, 통신 효율
- 멀티모달 균일성, 적응력, 지식 통합도
- CPU 시간·메모리 사용, 프라이버시
- 데이터 엔트로피, 공동 정보
- 전송 정보 손실, 정보 복원, 정보 전송률
- 노드 간 정보 활용

지표마다 입력, 단위, 검증 실험은 [docs/metrics.md](../../docs/metrics.md)에 정리했습니다.

## 코드 구성

`api.py`에 16개 지표 정의(`metric_definitions`), 분류 지표(`classification_metrics`), 통합 평가(`evaluate16`)가 있습니다. 정밀도와 재현율 계산이 보고서 생성 코드와 분리되어 있어서 HAR의 6개 활동과 Digits의 10개 클래스를 같은 API로 평가합니다.

실험 코드는 `studies/` 아래에 있습니다.

- `studies/purpose_multimodal_study.py`: 실제 센서 조합, 입력 중단과 복귀, 로컬·전역 모델 비교
- `studies/purpose_system_study.py`: 실제 송수신, 수렴, 전송 비용, CPU·메모리 측정
- `studies/purpose_recovery_stress.py`: 업데이트 손상과 복원 대조
- `studies/purpose_privacy_study.py`: 클리핑과 가우시안 잡음, 프라이버시 예산, 회원추론 공격 대조
- `studies/verify_purpose_*.py`: 보존한 원시 자료와 전송 기록으로 독립 재계산

## 실행

```bash
defai-research metrics-schema
defai-research metrics --input predictions.npz --observations observations.json --output results/metrics.json
defai-research metric-study multimodal -- --download --prepare-only
defai-research metric-study multimodal
defai-research metric-study system -- --stage smoke --output results/system_smoke
defai-research metric-study privacy
```

`predictions.npz`에는 `labels`와 `probabilities`를 넣고, 추가 관측 JSON의 키는 지표 정의의 `field` 값을 씁니다. HAR 데이터는 `--download`를 줄 때만 내려받습니다. 결과 폴더는 `DEFAI_WORKSPACE` 환경변수로 바꿀 수 있습니다.

## KIST 모델 평가

`evaluate16`과 `classification_metrics`는 레이블과 예측 확률 배열만 받으므로 KIST 모델의 예측에 바로 쓸 수 있습니다.

```python
import torch
from defai_research.metrics import classification_metrics, evaluate16
from fl_core.backends import get_backend

backend = get_backend("kist", {"dataset": "mnist-mirror"})
model = backend.build_model()
model.load_state_dict(torch.load("results/kist-run/final_model.pt"))
x, y = backend.test_set()
with torch.no_grad():
    probs = torch.softmax(model(x), 1).numpy()
rows = evaluate16(labels=y.numpy(), probabilities=probs, observations={})
```

KIST `history.json`의 통신량(`comm-mb-*`) 같은 값은 해당 지표의 `field` 이름으로 `observations`에 넣으면 함께 계산됩니다. 프라이버시 예산은 실제로 적용한 메커니즘과 공개 범위에 한해서만 해석해야 하고, 공격 AUC만 보고 보호 수준을 판단하지 않습니다.

## 검증

`tests/research/test_metrics.py`는 16개 지표가 등록되고 미관측 값이 유지되는지, 정밀도·재현율·F1이 정답과 맞는지 확인합니다. 엔트로피와 조건부 정보량, 수렴, 통신, 적응, 통합, 자원, 프라이버시, 복원, 전송률 계산도 함께 검사합니다. 정리 이전의 검증 기록은 [docs/history/validation_mapreduce_graph_metrics_v1.0.1.md](../../docs/history/validation_mapreduce_graph_metrics_v1.0.1.md)에 있습니다.
