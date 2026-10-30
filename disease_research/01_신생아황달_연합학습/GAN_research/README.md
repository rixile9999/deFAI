# 황달 GAN 재구축·연합학습 재검증 — 2026-10-16

**연구 단계·실전 미적용.** 원본 환자번호·검사 라벨의 임상 출처는 별도로 확인해야 한다.
이 폴더가 이번 재계산의 실행·결과 정본이다. 기존 `NeoJaundice-virtual`의 생성기는
학습된 모델이 아니라 색과 잡음을 만드는 절차식이며 비교 대조군으로 보존했다.

## 실행 환경

실제 실행: NVIDIA GeForce RTX 4080 16GB, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Python 3.11.
기존 `compute` 환경의 검증된 GPU 패키지를 읽기 전용으로 상속하는 전용 `.venv`를 사용했다.
추가 패키지는 이 `.venv`에만 설치했으며 기반 환경을 수정하지 않았다.
패키지 버전·CUDA·GPU·기반 경로는 `results/environment.json`에 기록했다.

처음 다운로드하려던 대형 GPU 패키지를 재설치하는 대신 이미 설치된 동일 버전을 재사용했다.
완전히 독립된 환경을 새로 만드는 경우 공식 PyTorch 인덱스를 사용한다:

```powershell
uv venv --python 3.11 .venv-clean
uv pip install --python .venv-clean\Scripts\python.exe -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-clean\Scripts\python.exe -r requirements-used.txt
```

아래 명령은 현재 폴더에서 실행한다. UTF-8을 지정하고 각 단계의 종료 코드를 확인한다.

## 원본 점검·분할

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe prepare_data.py --output data_new
```

기본 출력은 `data`이며, 이미 완료된 준비 결과는 덮어쓰지 않는다. 새 출력 경로를 쓰면 후속 스크립트의
기본 입력 `data`와 구분해야 한다. 이번 결과는 기존 `data` 준비본을 그대로 재사용한다.

- 원본: 745개 환자번호, 2,235장.
- 환자 번호가 다른 동일 사진·상이한 라벨: 6묶음, 12장 격리.
- 동일 환자 안의 완전 중복: 18장 정리.
- 빌리루빈 0값의 의미가 확인되지 않은 1명·3장: 주분석에서 격리. 임상적으로 잘못된 값이라고 판정한 것은 아니다.
- 최종: 742개 환자번호·2,202장. 학습 519명·1,544장, 검증 111명·330장, 시험 112명·328장.
- 환자·사진 내용이 서로 다른 분할에 섞이지 않는다. 세 가상 기관 배정도 환자마다 고정한다.
- 생후 나이 결측 2명(6행)은 각 기관의 **학습자료** 중앙값으로 대체한다.
- 피부 중앙의 고정 비율 영역을 잘라 색상 차트를 제외하고 64×64로 처리한다. 영상은 실제로 열어 확인했다.
- 모든 원본 사진·메타정보의 체크섬을 기록하고 마지막에 재확인한다.

`data/audit.json`, `split_manifest.csv`, `quarantined_*.csv`, `removed_within_patient_duplicates.csv`를 확인한다.
`data/real_data.npz`가 모델 입력 캐시다. 원본 사진은 수정하지 않는다.
같은 번호가 실제 동일 환자임을 임상적으로 확인한 것은 아니며, 다른 환자 번호의 근접 촬영 중복까지 모두 판정한 것도 아니다.

## 기관별 GAN 학습·생성

```powershell
.\.venv\Scripts\python.exe train_gan.py --steps 3000
.\.venv\Scripts\python.exe train_paired_gan.py
```

두 생성 방식 모두 기관별 자기 학습 환자만 사용한다. 검증·시험 환자는 생성에 쓰지 않는다.

1. **잡음 조건부 GAN**: 잡음 96차원 + 메타정보·빌리루빈·치료 라벨 6차원.
   resize-convolution 생성기, 스펙트럴 정규화·조건 투영·보조 회귀 판별기, hinge 손실,
   기하·부분 가리기 증강, 생성기 지수이동평균. 기관당 3,000단계, 3기관 총 9,000단계.
   색상 자체를 임의 변경하는 증강은 하지 않는다.
2. **같은 환자 촬영쌍 영상 조건부 GAN**: 같은 학습 환자의 다른 사진을 학습쌍으로 사용한다.
   피부 구조를 보존하면서 촬영 변화의 작은 잔차를 생성한다. 기관당 2,000단계, 총 6,000단계.
   RGB 변화는 최대 약 19/255로 제한한다. 독립적인 새로운 환자가 아닌 원본 사진의 작은 파생 증강이다.

첫 모델에서 질감의 인공 패턴·검증 특징 분포 불일치가 확인되어 두 번째 모델을 추가했다.
두 모델 모두 결과를 남기며 첫 모델의 실패를 숨기지 않는다.

- `data/gan_synthetic.npz`: 잡음 GAN 생성 1,544장.
- `data/gan_paired_synthetic.npz`: 촬영쌍 GAN 생성 1,544장.
- 각 파일의 `source_index`·`client`로 학습 환자와 기관 출처를 추적한다.
- `results/gan`, `results/paired_gan`: 생성기 체크포인트, 단계별 로그·표본·GPU 실행 기록.

생성자료는 원래 학습 라벨을 조건으로 받는다. 생성 영상이 그 검사값을 생리학적으로 충실히 표현한다고 검증한 것은 아니다.
촬영쌍 증강은 실제 사진에 가깝게 남으므로 개인정보 보호용 공개 데이터로 간주해서는 안 된다.

## 연합학습 비교

```powershell
.\.venv\Scripts\python.exe run_federated_study.py --rounds 20 --seeds 17 29 43
.\.venv\Scripts\python.exe run_federated_study.py --paired --rounds 20 --seeds 17 29 43
.\.venv\Scripts\python.exe run_federated_study.py --procedural-half --rounds 20 --seeds 17 29 43
```

전체 9개 조건: 중앙집중 원본, 기관 단독 원본, 연합 원본,
기존 생성 50%/100%, 잡음 GAN 50%/100%, 촬영쌍 GAN 50%/100%.
초기값 3개로 총 27회 비교 실행한다. 기관 단독 조건은 기관별 모델 3개를 포함한다.
기존 생성 50%는 GAN과 증강 비율을 맞춘 대조를 보완하기 위해 추가했다.

예측기는 같은 ImageNet ResNet18과 메타정보 4개를 결합한다. 배치 정규화는 공통 사전학습 통계에 고정하고
메타정보 층은 LayerNorm으로 바꿨다. 회귀 손실은 실제 mg/dL 단위 SmoothL1, 치료 분류 손실은 이진 교차엔트로피의 0.5배다.

- 실제 학습 자료는 매 회 모두 사용하며 마지막 배치를 버리지 않는다.
- 모든 연합 증강 비교는 같은 실제 자료 통과 수·최적화 갱신 수를 사용한다. 생성자료는 그 미니배치에 추가한다.
  따라서 증강 조건의 처리 영상 수·계산량은 더 많다. 중앙집중은 배치 나눔에 따른 갱신 수 차이가 있으므로 실제 수를 JSON에 기록한다.
- 기관 가중치는 생성자료 수가 아닌 **실제 학습 환자 수**로 고정한다.
- 지역 최적화는 매 회 Adam 상태를 재시작하며 모든 연합 조건에 같은 규칙을 적용한다.
- 같은 환자의 사진 수 차이를 고려해 사진별 학습 손실에 환자 영상 수의 역수를 가중한다.
- 검증 환자의 평균절대오차로만 체크포인트를 고르고 그 뒤 실제 시험 자료를 평가한다.
- 한 환자의 여러 사진 예측을 평균한 뒤 환자 단위로 평균절대오차·분류 지표를 계산한다.
- 생성자료로 시험하지 않는다. 정확도 기준은 모든 조건에서 확률 0.5다.

본 성능 계산은 단일 RTX 4080에서 세 기관을 순차 실행하는 연합학습 모의실험이다.
실제 병원 네트워크·보안 배포를 검증했다고 주장하지 않는다.
기존 `NeoJaundice/src`의 모델·분할·클라이언트·서버 집계 문제도 관련 범위에서 수정했다.
예전 모델 가중치는 그대로 보존했지만 메타정보 정규화 층이 바뀌어 새 모델과 직접 호환되지 않는다.
수정 전 소스는 `archive/baseline_before_gan`에 있다. 과거 진입점의 기존 분할·로그를 이번 연구 결과와 혼동하지 않는다.

## 검증과 보고서

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe network_check.py
.\.venv\Scripts\python.exe check_generation.py
.\.venv\Scripts\python.exe write_report.py
```

`network_check.py`는 실제 Flower 서버 1개·GPU 클라이언트 3개를 별도 프로세스로 실행한다.
기관당 학습 사진 64장·2회 통신의 동작 시험이며 전체 성능 수치와 구분한다.
`check_generation.py`는 시험 사진 없이 검증 사진과의 고정 ResNet18 특징 분포 거리를 계산한다.
표준 Inception 기반 지표·임상적 실제성 점수가 아니다. 원본에 가까운 파생 영상은 당연히 작은 거리를 낼 수 있다.

`write_report.py`는 9조건·3초기값이 모두 완료돼야 보고서를 만든다.
세 초기값의 환자별 오차 차이를 평균하고 112명을 5,000회 짝지어 재표집해 탐색적 95% 구간을 계산한다.
같은 증강 비율의 GAN과 기존 생성 대조도 직접 비교한다. 다중 비교 보정은 하지 않았다.
단일 환자 분할·단일 기관 배정·한 가지 생성 초기값 조건이므로 일반적 개선으로 확정하지 않는다.

| 한 것 | 안 한 것 |
|---|---|
| RTX 4080에서 기관별 두 GAN 학습·생성 | 새로운 독립 환자·병원 자료 확보 |
| 사진 중복 정리·환자 단위 분할·학습 코드 수정 | 검사 라벨의 임상적 재판정 |
| 9조건 × 3초기값 × 20회 학습 | 다중 환자 분할·기관 배정·생성 초기값 검증 |
| 실제 서버·GPU 클라이언트 통신 확인 | 병원 간 배포·암호화·차등프라이버시 |
| 실패 생성 모델과 모든 결과 공개 | 합성 영상의 임상 타당성 입증 |

이 데이터·이 분할·이 방식 한정의 결과다. 미탐색 조건은 색상 보정·고해상도,
다른 기관 이질성, 외부 환자 자료, 원본에 대한 기억 여부와 생리학적 라벨 일치다.

주요 산출물:

- `results/gan_study_report.html`: 설명·결과·한계 보고서.
- `results/final_summary.json`, `results/final_metrics.csv`: 전체 조건 요약과 짝지은 비교.
- `results/prediction`, `results/paired_prediction`, `results/procedural_half_prediction`: 모든 실행의 시험 예측·검증 이력·가중치.
- `results/generation_comparison.jpg`: 같은 출처 조건을 정렬한 영상 비교.
- `results/mae_comparison.png`: 실제 시험 환자별 평균절대오차 그림.
- `results/environment.json`, `results/tests.xml`, `results/network`: 환경·테스트·통신 근거.

방법 근거: [PyTorch 공식 설치](https://pytorch.org/get-started/previous-versions/),
[소량 데이터의 미분 가능 증강](https://github.com/mit-han-lab/data-efficient-gans),
[자료 증강 GAN](https://arxiv.org/abs/1711.04340).

## 사진 명도 편향 억제 추가 실험

연구 단계·실전 미적용. 실제 본래 피부색·멜라닌 측정 라벨은 없다.
사진 명도 삼분위는 황달·조명을 포함하는 대리지표이며 피부색/인종 집단이 아니다.
RGB 공통 배수 변화만 다루므로 생물학적 멜라닌의 색 변화에 대한 공정성을 증명하지 않는다.

명도 증강, 명도 증강+출력 일관성, 픽셀 색상 비율 입력을 원본 및 촬영쌍 생성 100% 조건에 적용한다.
기존 저장 모델 6개와 새 모델 18개를 평가한다. 20회·3기관·초기값 17/29/43,
기존과 같은 실제 환자 수 가중치 및 환자 분할, 원본 검증 환자 평균절대오차로만 체크포인트 선택.
증강 범위 0.65~1, 시험 스트레스 0.55/0.75/0.90, 모든 RGB 채널 공통 배수로 비율 보존.
색상 비율 입력은 배수에 불변이지만 명도 질감까지 제거한다. 전체 성능과 집단 성능을 함께 보고한다.

프로젝트 디렉토리에서 실행:

```powershell
.\.venv\Scripts\python.exe -u run_skin_bias_study.py
.\.venv\Scripts\python.exe summarize_skin_bias.py
.\.venv\Scripts\python.exe write_report.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

`results/skin_bias/protocol.json`은 시험 평가 전에 기록한 고정 방법·손실 계수·분할 기준이다.
`results/skin_bias/summary.json`은 전체 8조건의 수치와 환자별 짝지은 5,000회 층화 재표집 구간,
`summary.csv`는 전체 지표, `subgroup_metrics.csv`는 8개 비교 조건 × 3개 명도 집단 지표다.
집단 격차와 최대 집단 오차는 세 초기값의 환자별 절대오차를 먼저 평균한 뒤 계산한다.
검사값 공통 구간 구성 보정은 세 명도 집단 모두 최소 3명이 있는 구간만 사용하며
적용 환자 수를 보고한다. 원래 전체 환자 결과도 모두 유지한다.
`report_section.html`은 `write_report.py`가 본 보고서에 삽입하므로 재생성해도 보존된다.
`archive/before_skin_bias`에 추가 전 보고서와 렌더링 코드를 보존했다.

## FedProx·SCAFFOLD·FedBN 비교 (2026-10-17)

연구 단계·실전 미적용. `federated_algorithms.py`가 근접 벌점, SCAFFOLD 보정 및 지역 배치 정규화 상태를 구현한다.
`run_algorithm_study.py`는 원본/촬영쌍 생성 100% × 명도 일관성 미적용/적용 네 조건을 모두 평가한다.
기존 자료 분할·기관 배정을 유지하고, 새로운 강한 기관 이질성은 만들지 않는다.

- FedProx: Adam 0.0001, 근접 벌점 강도 0.01/0.1/1.0. 파라미터 거리 제곱의 합을 사용한다.
- SCAFFOLD: 운동량 없는 확률적 경사하강 0.001/0.01/0.1, 실제 지역 단계 수를 사용하는 두 번째 보정 갱신식.
  손실 경사를 길이 1로 제한한 후 보정항을 더하는 변형이다. 같은 학습률의 연합 평균 대조도 평가한다.
- FedBN: Adam 0.0001, 배치 정규화의 모든 계수·실행 통계·정수 계수기를 기관별 보유.
  알려진 기관 모델로 평가하며 시험 사진으로 정규화 통계를 갱신하지 않는다.
- 대조군: 기존 통계 고정 연합 평균·Adam, 연합 평균·확률적 경사하강, 지역 배치 통계를 학습하고 전역 집계하는 연합 평균.
  마지막 대조의 실행 통계 평균은 모집단 평균·분산을 추정하는 절차가 아닌 모델 상태 평균 휴리스틱이다.

초기값 17에서 네 조건별 12개 설정을 모두 검증 점수로 비교하고, 방법별 설정을 고정해 초기값 29/43을 반복한다.
최적 경사하강 대조와 SCAFFOLD의 선택 학습률이 다르면 동일 학습률 대조를 추가 반복한다.
각 실행은 20회 통신·지역 1회 자료 통과·배치 32. 모든 학습이 끝나야 시험 예측을 시작한다.
중단 후 같은 명령으로 재개하면 완료된 실행을 건너뛴다. 도중 중단된 실행은 해당 20회를 처음부터 다시 수행한다.
`protocol.json`의 설정이 달라지는 재개는 거부한다. 시험 결과를 보고 설정을 바꾸지 않는다.

```powershell
.\.venv\Scripts\python.exe -u run_algorithm_study.py
.\.venv\Scripts\python.exe summarize_algorithm_study.py
.\.venv\Scripts\python.exe write_report.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

선택 설정·전체 탈락 설정의 검증 결과·초기값별 결과·시험 예측·명도 집단 지표·환자 재표집 구간은
`results/algorithm_study`에 저장한다. `complete.json`이 생성되기 전에는 결과 보고서를 만들지 않는다.
`summary.json`의 조건별/전체 최적 후보는 평균 검증 오차로 선택한 결과이며 시험 점수 순위가 아니다.
`archive/before_algorithm_comparison`에는 추가 전 보고서와 렌더링 코드를 보존했다.

독립된 네 조건을 동시에 계산할 여유가 있으면 `run_algorithm_parallel.py`를 대신 실행한다.
각 실험 안의 기관 3개는 여전히 순차 학습하고, 네 조건이 모두 완료되기 전에는 시험 평가를 하지 않는다.
두 실행 방식은 완료된 동일 이름의 검증 실행을 재사용한다. 이 계산에서는 초기에 순차 실행하다 GPU 여유가 생겨
독립 조건 동시 실행으로 전환했으며 `parallel_execution.json`과 `context_0.log`~`context_3.log`에 기록한다.
기관 환자 수는 실제로 174/173/172명이다. 최초 균등 가정의 검사가 SCAFFOLD 실행 전에 중단시켜
서버 보정 상태도 모델 집계와 같은 실제 환자 가중평균으로 수정했다. 초기 계획·중단 로그는 보존했다.
