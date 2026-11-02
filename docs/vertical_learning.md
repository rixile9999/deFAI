# 수직연합학습 구현·검증 범위

연구 단계, 실제 기관에 배포하지 않은 소프트웨어입니다.
출발 문서: 사용자가 제공한 「모델별 연합학습 구현표」(2026-10-06),
수직형 열과 6-2절. 원본 PDF는 재배포하지 않습니다.

구현 순서는 대상자·시점 정렬 → 분할 신경망 → 통신·노출 검사 → FedBCD →
암호화 트리 → 모델별 확장 → 패키지·명령행·회귀 테스트입니다.
수평형·전이형의 새 구현 및 GV-HE는 이번 범위 밖입니다.

| 문서의 계열 | 검증할 수직형 경로 |
|---|---|
| DNN | 기관별 MLP와 라벨 보유자의 상위 모델 |
| CNN | 동일 개체의 기관별 영상 인코더와 풀링 |
| LSTM·GRU | 동일 개체·시간의 로컬 시계열 인코더 |
| TCN | 미래를 보지 않는 인과 합성곱 |
| Transformer | 풀링 벡터 또는 토큰 경계 |
| LLM | 로컬 동결 인코더의 특징·점수 결합; 대규모 생성 학습과 구분 |
| GNN | 동일 노드 집합의 로컬 특징·그래프 인코더; 원 논문의 보안 계산과 구분 |
| AE·VAE | 기관별 복원 손실을 사용하는 무라벨 학습 |
| GAN·확산 | 공통 잠재/시간에서 기관별 특징 생성·예측하는 연구 예제 |
| GBDT | 암호화된 기울기·헤시안 히스토그램으로 분할 선택 |

## 설치와 첫 실행

Python 3.11/3.12의 새 가상환경에서 `python -m pip install ".[vertical]"`로 설치합니다.
기존 `[all]`에도 포함됩니다. Flower를 사용하지 않는 별도 로컬 실행 경로입니다.
발견 명령은 PyTorch를 불러오지 않습니다. 기본 예제에는 다운로드가 없습니다.

```powershell
defai vertical list
defai vertical init studies/vertical01
defai vertical run studies/vertical01/manifest.json --method split --epochs 30 --output results/vfl-split01.json
defai vertical run studies/vertical01/manifest.json --method fedbcd-p --q 3 --output results/vfl-bcd01.json
defai vertical run studies/vertical01/manifest.json --method secureboost --epochs 3 --output results/vfl-tree01.json
defai vertical demo --family all --output results/vfl-families01.json
```

결과 경로·작업 폴더는 기존 파일을 덮어쓰지 않습니다. `init`은 **합성** 기관별 CSV와
manifest를 만들며, 한 기관의 행을 뒤집어 ID 정렬을 실제로 거치게 합니다.
자기 CSV를 사용하려면 manifest의 경로·특징 열·ID 열·라벨 보유자를 바꿉니다.
파일 경로는 manifest 기준이며 라벨은 한 기관의 CSV에서만 읽습니다.
라벨·ID 열을 입력 특징에 넣을 수 없고, 중복 ID·결측·비정상 수치는 거부합니다.
현재 표형 명령의 과제는 이진 분류입니다. 연속값 회귀와 다른 입력 형태는 Python API를 사용합니다.

공통 ID의 교집합만 학습합니다. `--alignment plain`은 식별자를 직접 비교하는 연구용 기본값이고,
`--alignment dh-psi`는 새로운 세션의 교환 가능한 Diffie–Hellman 지수 연산으로 교집합을 만듭니다.
집합 크기·교집합 소속과 행 순서는 여전히 노출됩니다. 실제 네트워크 인증이나 악의적 참여자
보안 증명이 없는 실험 구현입니다. 단순 해시를 안전한 교집합이라고 부르지 않습니다.

학습/평가 행을 먼저 분리하고, 각 기관의 평균·표준편차는 **학습 행만** 사용합니다.
결과의 accuracy는 평가 행 중 맞힌 비율(0~1), log_loss는 이진 로그 손실입니다.
loss_history는 학습 손실이며 평가 라벨을 바꿔도 학습 과정이 바뀌지 않는지 테스트합니다.
통신 수치는 텐서/배열 payload 바이트 수로, 실제 네트워크·암호·프로토콜 부가비용은 아닙니다.
암호화 트리의 통신량은 아직 계측하지 않아 `communication: null`을 기록합니다.

## Python API와 모델별 동작

```python
import torch
from torch import nn
from defai.vertical.models import mlp
from defai.vertical.split import FeatureParty, SplitTrainer

# 아래 행은 반드시 동일 개체 순서로 정렬된 상태여야 합니다.
a = FeatureParty("hospital", mlp(3, 4), torch.randn(8, 3))
b = FeatureParty("insurer", mlp(2, 4), torch.randn(8, 2))
trainer = SplitTrainer([a, b], nn.Linear(8, 2), torch.zeros(8, dtype=torch.long))
trainer.step([0, 1, 2, 3])
prediction = trainer.predict([4, 5])
```

`alignment.align_parties`는 기관별 `(IDs, local_array)`를 받고 정렬된 로컬 배열과
라벨 보유자의 정렬된 라벨을 반환합니다. 원본 배열을 덮어쓰지 않습니다.
`align_time_series`는 명시한 공통 시간축으로 로컬 관측을 보간하며 외삽은 거부합니다.
이는 오프라인 전처리입니다. 보간의 오른쪽 관측을 아직 모르는 실시간 예측에는 그대로 쓰지 마세요.

`FeatureParty`는 자기 인코더·입력·최적화기만 보유합니다. 상위 학습기는 분리된
임베딩을 받고 해당 임베딩의 기울기만 반환합니다. 이전 배치 기울기의 재사용을 거부합니다.
회귀는 `task="regression"`과 예측과 같은 모양의 실수 라벨을 사용합니다.
CPU를 기준으로 테스트하며, 체크포인트의 난수 복원도 CPU 범위입니다.

| 모델 계열 | 실행되는 경로 | 범위와 조건 |
|---|---|---|
| DNN | `models.mlp` + `SplitTrainer` | 중앙 SGD와 손실·모든 파라미터의 수치 일치 검사 |
| CNN | `ImageEncoder` | 기관별 영상, 전역 풀링; 실제 다기관 동시 촬영 데이터는 미검증 |
| LSTM·GRU | `RecurrentEncoder(cell=..., sequence=...)` | 마지막 상태 또는 전체 시점; 사전 시간 정렬 필요 |
| TCN | `TemporalEncoder(sequence=...)` | 왼쪽 패딩과 dilation을 쓰는 인과 합성곱; 미래 누출 검사 |
| Transformer | `TransformerEncoder(tokens=...)` | 평균 풀링 또는 시점별 표현; 고정 길이 입력, 대규모 사전학습 아님 |
| LLM | `FrozenEncoder(user_encoder)` | 로컬 모델을 동결해 특징/점수 결합; 예제는 소형 Transformer, 실제 LLM 가중치·성능 미검증 |
| GNN | `GraphEncoder`, 선택형 `SharedSumHead` | 로컬 그래프와 특징; 두 보조자의 합산 시뮬레이션, VFGNN 전체 보안 프로토콜 아님 |
| AE·VAE | `VerticalAutoencoder(variational=...)` | 각 기관이 자기 재구성 손실 계산, 라벨 없음; 잠재 표현은 공유됨 |
| GAN·확산 | `VerticalGAN`, `VerticalDiffusion` | 공통 잠재/시간과 로컬 특징 블록; 생성 품질·대규모 안정성 미검증 |
| 트리 | `HistogramParty`, `SecureBoostClassifier` | 이진 histogram boosting; 새 행은 `with_features`로 학습 기준값 유지 |

따라서 10개 계열에 실행 경로가 있지만, 조건부 계열은 연구 예제이며 대규모 실배포 구현과 구분합니다.
LLM 모델 블록을 앞/뒤로 자르는 방식 자체는 데이터 수직 분할과 다르므로,
일반적인 분할 모델을 수직형 새 알고리즘으로 집계하지 않습니다.

## FedBCD와 암호화 트리의 세부 범위

`bcd.LinearParty`와 `FedBCD`는 가산 선형/로지스틱 예측기의 FedBCD-p/s입니다.
`q=1`, 병렬 설정은 중앙의 같은 미니배치 SGD와 비교합니다. `q>1`은 같은 배치를
Q번 재사용하고 다른 기관에서 받은 중간값을 캐시합니다. 수동적인 특징 보유자에게
라벨 대신 미분값을 전달합니다. 임의의 심층 상위 모델에 적용하는 신경망 FedBCD는 별도입니다.
병렬·순차 모드의 통신량 정의는 다르며, Q를 키워도 성능이 보존된다고 보장하지 않습니다.

트리는 라벨 보유자가 샘플별 기울기·헤시안을 Paillier 암호화하고, 특징 보유자는
공개키로 후보별 합만 계산합니다. 중앙 모델에는 기관명과 분할 기록 번호만 남고,
특징 열·경계값 조회표는 해당 기관 객체에 보관합니다. 추론에도 그 조회표가 필요합니다.
기본 키는 암호학적 난수를 사용하는 2048비트입니다. `--insecure-test-key`는 빠른
테스트용 256비트 키를 **명시적으로** 허용하고 결과에도 표시합니다. 실제 자료에 사용하지 마세요.
고정소수점 정밀도·히스토그램 설정이 있으므로 XGBoost/FATE와 비트 단위 동일성을 주장하지 않습니다.

## 노출 감소 기능과 남는 한계

- `protection.clip_and_noise`: 샘플별 임베딩/기울기 클리핑과 Gaussian 잡음.
- `protection.max_norm`: 0이 아닌 기울기의 기대 제곱 norm을 맞추는 무작위 보정.
  방향은 남고 0인 행은 그대로입니다.
- `protection.marvell`: 이진 라벨 보유자 내부에서 Gaussian 클래스 공분산을 추정하고,
  평균 잡음 전력 제약 아래 구조화된 잡음을 최적화합니다. 작은 배치·모형 가정의 한계가 있습니다.
- `protection.binary_auc`: 기울기 norm 등의 점수로 라벨이 얼마나 구분되는지 진단합니다.
  특정 공격에서 값이 낮아져도 전체 공격에 안전하다는 뜻은 아닙니다.
- `sharing`/`fusion`: 두 비공모 보조자를 가정한 고정소수점 합산과 학습 예제.
  개별 보조자는 share만, 수신자는 합을 얻는 프로토콜이지만 여기서는 전부 한 프로세스에서 실행합니다.

임베딩/기울기 교환 자체는 원본 특징·라벨을 복원하는 공격을 배제하지 않습니다.
위 잡음에는 별도의 end-to-end 개인정보 보호 예산 증명이 없어 epsilon/delta를 보고하지 않습니다.
암호 구현은 감사되지 않은 Python 연구 코드이며 상수시간 연산·악의적 참여자 방어·
인증된 기관 간 통신·비선형 신경망 전체의 암호 계산·실제 VFGNN 보안 재현은 제공하지 않습니다.
동일 프로세스의 조정자는 객체 메모리에 접근할 수 있으므로 기관 격리를 구현했다고 해석하지 마세요.

## 저장·검증·진행 상태

`checkpoint.save_checkpoint(trainer, path)`는 모델·최적화기·CPU 난수 상태를 저장하며
특징·라벨·대상자 ID는 저장하지 않습니다. `load_checkpoint`는 구조와 수치 상태를 먼저 검증한 뒤
복원합니다. 재개할 때 사용자가 같은 정렬·입력·보호 설정을 제공해야 합니다.
예제와 체크포인트는 원본 기관 데이터를 재배포하는 수단이 아닙니다.

```powershell
python -m pip install ".[research,mobile,test,vertical]"
python -m pytest -q -k vertical
python -m pytest -q
python -m build
```

| 구분 | 내용 |
|---|---|
| 한 것 | 단위·통합·중앙 수치 비교·평가 라벨 격리·합성 예제·패키지 검사 |
| 안 한 것 | 실제 기관 간 네트워크, 실제 대형 LLM, 실제 의료·공장 자료 평가, 생성 품질 평가, 보안 감사·형식 증명 |

각 기능의 실제 테스트 통과 여부는 해당 커밋의 CI와 테스트 결과를 확인하세요.
문서 표는 실행·검증 범위를 설명하며 임상 성능이나 보안 인증을 의미하지 않습니다.

## 참고 원문

- Split learning: https://arxiv.org/abs/1812.00564
- FedBCD: https://arxiv.org/abs/1912.11187
- SecureBoost: https://arxiv.org/abs/1901.08755
- Label leakage and protection: https://arxiv.org/abs/2102.08504

원본 PDF 식별: 모델별 연합학습 구현표, 총 16쪽, 수직형 설명 1–6쪽 및 6-2절 10–11쪽.
SHA-256: `3b833964f16ba50afff64ba6676c9a438cf62a4d4dff7e26c91024c1469c5d47`.
