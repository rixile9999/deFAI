# 3-1 트랜스포머 기반 멀티모달 학습 모델

이미지, 텍스트, 오디오를 함께 입력받는 Transformer 분류 모델입니다(3차년도 보고서 다-(1)). 모달리티별 인코더 출력을 공통 공간에 놓은 다음 Transformer 인코더, 크로스모달 디코더, 게이트형 융합(GMU)을 거쳐 분류 확률과 표현 벡터를 냅니다. 기본 모델은 `MultimodalTransformer`이고, 패키지는 `defai_addons.multimodal`([`src/defai_addons/multimodal`](src/defai_addons/multimodal)), 테스트는 [`tests/addons`](../../tests/addons)에 있습니다.

## 1·2차년도 모델은 어떻게 되었나

1차년도에는 공동 임베딩으로 멀티모달 피처를 만들고 연결했습니다(이미지 CNN, 텍스트 Transformer, 오디오 CNN·RNN, 계층적 피처 융합). 인코더와 공통 임베딩은 최신 모델 안에 그대로 있고, 계층적 피처 융합은 보완 기능으로 남겼습니다. 2차년도의 어텐션 기반 멀티모달 학습(셀프·크로스 어텐션, 코어텐션) 가운데 셀프 어텐션과 크로스 어텐션은 3차 구조에 합쳤습니다. 어텐션 정렬과 그래프 어텐션 연결은 역할이 달라 별도 모듈로 유지했습니다. 3차년도에는 공통 임베딩 통합, Transformer 인코더, 크로스모달 디코더, GMU 융합, 선택형 생성 디코더를 구현했고 이것이 최신 구현입니다.

1·2차년도에 완성했던 공동 임베딩 모델과 어텐션 융합 모델은 `src/defai_addons/multimodal/baselines`에 재현용 기준선으로 남겨 두었습니다. 기본 학습 대상은 3차년도 Transformer 하나뿐입니다. 정렬 코드와 그래프 연결 코드는 남아 있지만 최신 Transformer의 기본 순전파에는 연결되어 있지 않습니다.

## 구성

| 파일 | 내용 |
|---|---|
| `encoders.py`, `joint.py` | 모달리티 인코더, 공통 임베딩 |
| `transformer.py`, `fusion.py` | Transformer 인코더·크로스모달 디코더, GMU·계층적 융합 |
| `generation.py` | 선택형 자기회귀 생성 디코더(`sequence_vocab` 지정 시) |
| `models.py` | `MultimodalTransformer`와 비교용 변형(통합 없음, 디코더 없음, 평균 융합, 단일 모달) |
| `federated.py` | 연합 학습 보조 실험 |
| `train.py`, `data.py` | 학습·평가, 합성 멀티모달 데이터 |
| `baselines/` | 1·2차년도 완성 모델 |
| `scripts/train_multimodal.py`, `run.py` | 실험 실행 |
| `evidence/` | 보고서에 쓴 시드별 실험 결과 |

## 실행

```bash
python 3_heterogeneous_data/3-1_multimodal/run.py --part central --epochs 5 --n-train 600 --n-test 120
python 3_heterogeneous_data/3-1_multimodal/run.py --part federated --help
```

기본 데이터는 이미지, 토큰, 파형으로 만든 합성 데이터라서, 여기서 나온 결과가 실제 서비스 데이터에서의 성능을 뜻하지는 않습니다.

```python
from defai_addons.api import create_multimodal_model, predict_multimodal

model = create_multimodal_model(d=64, n_classes=6, enc_layers=2, dec_layers=1)
result = predict_multimodal(model, batch)
result["probabilities"].shape
result["representation"].shape
```

`batch`에 넣는 입력은 다음과 같습니다.

| 입력 키 | 형태 |
|---|---|
| `image` | float `[N, 1, 28, 28]`, 학습 때와 같은 정규화 |
| `text` | 정수 `[N, 16]`, 토큰 ID 0~63, 0은 패딩 |
| `audio` | float `[N, 1, 800]`, 학습 때와 같은 정규화 |
| `mask` | bool `[N, 3]` (이미지, 텍스트, 오디오 순), 행마다 하나 이상 True |
| `y` | 학습·평가 때 정수 `[N]` |

문자열 토큰화와 실제 오디오 전처리는 하지 않으며, 텍스트 인코더의 어휘는 실험용 64개뿐입니다. 빠진 모달리티도 배열 형태는 그대로 두고 `mask=False`로 표시합니다.

학습한 모델은 체크포인트로 저장했다가 다시 불러올 수 있습니다. 이미 있는 파일은 덮어쓰지 않습니다.

```python
from defai_addons.api import create_multimodal_model, save_multimodal_checkpoint, load_multimodal_checkpoint
from defai_addons.multimodal.train import fit, evaluate

model = create_multimodal_model(d=64, n_classes=6)
fit(model, train_data, validation_data, epochs=10, batch_size=32)
save_multimodal_checkpoint(model, "results/multimodal.pt", {"d": 64, "n_classes": 6})
loaded = load_multimodal_checkpoint("results/multimodal.pt")
```

```bash
defai-addons multimodal-predict --checkpoint results/multimodal.pt --data results/multimodal-input.npz --out results/predictions.json
```

## KIST 시스템과 연결

KIST의 ServerApp은 이미지 모델(`Net`)과 이미지 입력 형식을 고정해서 쓰기 때문에 설정 값만 바꿔서는 이 모델로 바꿀 수 없습니다. KIST 저장소를 고치지 않는 범위에서는 별도의 멀티모달 워크플로로 호출합니다. 연결 스크립트에서 위 Python API를 부르거나 `defai-addons multimodal-predict`를 별도 프로세스로 실행하면 됩니다.

KIST 연합학습에 직접 넣으려면 KIST `task.py`의 `build_model`과 데이터 로더에 멀티모달 분기를 추가해야 합니다. 방법은 [통합 가이드](../../docs/KIST_통합_가이드.md#5-kist-저장소를-수정해야-하는-부분)에 정리했습니다.

## 검증

- `tests/addons/test_defai.py`: 결측 모달리티가 있을 때의 순전파, GMU가 결측 모달리티를 무시하는지
- `tests/addons/test_year3_requirements.py`: 생성 디코더의 인과 마스크와 종료 조건, 시퀀스 헤드를 학습한 뒤 기본 모델 보존
- `tests/addons/test_host_integration.py::test_checkpoint_prediction_roundtrip`: 저장 후 복원한 출력 일치, 잘못된 결측 마스크 거부
- 이전 학습 결과: [evidence](evidence)
