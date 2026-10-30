# deFAI 0.2 사용자 안내

공통 연합학습 알고리즘과 네 질병 연구를 하나의 설치 패키지에서 제공합니다.
모든 연구는 **연구 단계·실전 미적용**입니다. 작은 합성 예제의 성공은 임상 성능,
실제 기관 간 네트워크 성능, 실제 정책효과를 입증하지 않습니다.

## 1. 설치와 첫 실행

Python 3.11 또는 3.12를 권장합니다. 기본 Python 패키지는 3.13도 지원하지만,
Windows에서 Flower의 Ray 기반 분산 시뮬레이션을 사용할 때는 3.11/3.12가 필요합니다.
`defai run`의 순차 실행은 Ray 없이도 실제 모델을 학습·평가합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install ".[all]"
defai doctor
defai list
defai demo --output results/demo01
```

배포된 wheel을 받은 경우 `python -m pip install "./defai-0.2.0-py3-none-any.whl[all]"`로
설치합니다. `defai demo`는 사진·공공 데이터·사전학습 가중치를 다운로드하지 않고
모든 기능 계열을 작은 합성 입력으로 실행합니다. 결과는 `demo.json`과 공통 학습의
`common/history.json`으로 남습니다. 같은 출력 경로를 덮어쓰지 않습니다.

CPU가 기본값입니다. CUDA용 PyTorch 설치는 [공식 호환 안내](https://pytorch.org/get-started/previous-versions/)에
따르고 `defai doctor --probe-gpu`로 확인한 뒤 `--device cuda`를 명시합니다.
특정 GPU 모델을 요구하지 않습니다.

| 설치 범위 | 포함 내용 |
|---|---|
| 기본 설치 | 기능 탐색, 작업 폴더 생성, NumPy 집계, 공공 데이터 참조·확인 |
| `[analysis]` | 감염세대간격 추정, 정책 통계, CSV/XLSX 처리 |
| `[learning]` | PyTorch 영상·개인별 학습 구성요소 |
| `[flower]` | 공통 15개 알고리즘과 실제 ClientApp 순차 실행 |
| `[research]` | 네 질병 연구의 Python 의존성, 그림·보고서·동형암호 |
| `[mobile]` | Android용 모델 집계 서버 |
| `[simulation]` | Flower/Ray 시뮬레이션 엔진 |
| `[all]` | 위 선택 기능 전체; Android SDK/JDK는 별도 설치 |

따라서 처음에는 `[all]`과 합성 예제로 실행 경로를 확인하고, 필요한 실제 자료를 준비하면 됩니다.
과거 연구 폴더의 requirements lock은 당시 환경의 기록입니다. 현재 환경에 중복 설치하지 마세요.

## 2. 공통 알고리즘: 같은 명령, 다른 설정

```powershell
defai run --strategy fedavg --dataset synthetic --clients 4 --rounds 3 --output results/fedavg01
defai run --strategy fedprox --clients 4 --rounds 3 --set "proximal-mu=0.1" --output results/fedprox01
defai run --strategy scaffold --clients 4 --rounds 3 --output results/scaffold01
defai run --strategy fedadam --clients 4 --rounds 3 --set "dp='fixed'" --output results/private01
defai run --strategy fedavg --set "compression='topk'" --set "error-feedback=true" --output results/compressed01
```

제공 전략은 `fedavg`, `fedprox`, `moon`, `scaffold`, `fednova`, `qfedavg`, `fedavgm`,
`fedadagrad`, `fedadam`, `fedyogi`, `fedmedian`, `fedtrimmedavg`, `krum`, `multikrum`, `bulyan`입니다.
서버 최적화, 악성 갱신 모의실험, 압축, 개인화 평가, 클라이언트 이탈, 차등 개인정보보호는
동일한 `--set KEY=TOML_VALUE` 형식으로 설정합니다. 전체 키는 작업 폴더의
`pyproject.toml`에 있습니다. 잘못된 키·형식·비정상 수치는 실행 전에 거부합니다.

강건 집계는 충분한 참여 수가 필요합니다. 예를 들어 `bulyan`, 악성 기관 수 가정 1개라면
최소 7개 응답이 필요하므로 `--clients 8 --set "fraction-train=1.0"`처럼 설정합니다.
차등 개인정보보호는 평균 기반 전략에서만 지원합니다. 기존 README의 알고리즘 설명과
개인정보보호 가정·한계는 그대로 적용됩니다.

`synthetic`은 다운로드 없는 기능 확인용입니다. 실제 공개 영상 벤치마크는
`mnist`, `cifar10`, `mnist-mirror`, `fashion-mnist-mirror`를 선택합니다.
그 성능은 질병 진단 성능과 구분해야 합니다.

`defai run`은 실제 Strategy와 ClientApp을 사용하되 한 프로세스에서 순차 실행합니다.
통신량 필드는 모델 페이로드의 계산값이며 실측 네트워크 전송량이 아닙니다.
기관별 상태는 라운드 사이에 유지됩니다. 결과에는 설정, 이력, 실행 시간, 실행 방식이 남습니다.
분산 실행이 필요하면 작업 폴더에서 기존 `flwr run .`과 Flower Simulation/Deployment Engine을 사용합니다.

## 3. 연구 작업 폴더와 전체 기능 목록

```powershell
defai init studies/session01
defai research list
defai research list --domain neonatal
defai research show neonatal.gan_research.prepare_data
defai research run neonatal.gan_research.prepare_data --workspace studies/session01 -- --help
```

`init`은 코드, 설정, 문서, 출처 레지스트리를 새 폴더에 복사하고 체크섬을 확인합니다.
설치 디렉토리에 사진·키·학습 결과를 쓰지 않습니다. 기존 폴더를 덮어쓰지 않습니다.
`research list`는 모든 연구 Python 파일을 script/library/test로 분류하고,
`show`는 원본 경로, 기능 설명, 공개 함수·클래스, 실행 인자를 보여줍니다.
라이브러리와 테스트 파일은 연구 스크립트로 잘못 실행하지 않습니다.

스크립트는 해당 프로젝트 폴더에서 별도 프로세스로 실행되므로 서로 다른 프로젝트의
`core`, `src`, `learning` 모듈이 충돌하지 않습니다. 원래 명령행 인자는 `--` 뒤에 전달합니다.
`--timeout 초`를 지정할 수 있고 실행 결과와 스크립트 해시는 `run_receipts/`에 남습니다.
오래된 시연·보고서 코드도 목록에 포함되지만, 그 안의 과거 수치가 새로 검증됐다는 뜻은 아닙니다.
자료·결과가 필요한 스크립트는 해당 입력을 먼저 생성해야 합니다.

## 4. 신생아 황달: 자료 → 준비 → 생성·학습 → 보고서

```powershell
defai data list
defai data fetch neojaundice --directory downloads
```

`NeoJaundice.zip`은 [Figshare 공식 v1](https://doi.org/10.6084/m9.figshare.22302559.v1)입니다.
크기와 공식 체크섬을 확인한 뒤 사용합니다. 다음 경로의 `01_...`은 `init`이 만든 실제 폴더 이름입니다.

```powershell
defai data prepare-neojaundice downloads/NeoJaundice.zip --directory studies/session01/disease_research/01_신생아황달_연합학습/NeoJaundice/data
defai research run neonatal.gan_research.prepare_data --workspace studies/session01
defai research run neonatal.gan_research.train_gan --workspace studies/session01 --device cpu -- --steps 1
defai research run neonatal.gan_research.run_algorithm_study --workspace studies/session01 --device cpu -- --rounds 1
```

단계별 입력을 확인합니다. `prepare_data`는 환자 단위로 분할하고 중복·모순·0 목표값을 격리합니다.
`train_gan`은 준비된 캐시로 생성 모델을 학습합니다. `train_paired_gan`은 촬영쌍 생성 가지입니다.
`run_federated_study`, `run_skin_bias_study`, `run_algorithm_study`는 각각 별도 비교 연구입니다.
`summarize_*`, `write_report`, `check_generation`은 그에 대응하는 결과·체크포인트를 읽습니다.
일부 전체 연구는 세 시드·여러 조건을 실행하므로 `demo`보다 오래 걸립니다.
원래 예측 대상과 환자 분할을 변경하지 않았으며, 밝기 대리 지표를 실제 피부색·멜라닌 측정으로 해석하지 않습니다.

Python에서는 `defai.research.neonatal`의 모델, 생성기, 집계와 알고리즘 보조 함수를 사용할 수 있습니다.
`MultiModalNet(pretrained=False)`는 사전학습 가중치 다운로드 없이 생성됩니다.

## 5. 정책효과·중증도 연구

```powershell
defai policy --input my-district-panels --output results/policy01.json
```

각 CSV/XLSX는 한 시도·시군구의 **완전한 일별 패널**이어야 합니다. 기본 달력은
2024-01-01~2024-04-09, 정책일은 2024-01-31입니다. `--start`, `--end`, `--policy-date`로 바꿀 수 있습니다.
필수 열은 `record_format=daily_counts_v1`, `date`, `cases`, `diagnosed_cases`, `시도`, `시군구`, `treated`입니다.
`treated`는 각 기관에 명시한 0/1이며 같은 시도의 기관끼리 일치해야 합니다.
0건인 날도 명시해야 하고, 미보고 날짜를 자동으로 0건으로 바꾸지 않습니다.

결과값은 `log(1+일별 관측 환자 수)`이고 시군구를 동등 가중합니다.
출력의 변환 비율을 원시 환자 수 감소율로 단정하지 않습니다. 정책 배정 단위의 광역 수가
부족하면 불확실성 추론을 보류합니다. 인과 해석에는 별도 가정과 검증이 필요합니다.

Python API는 `defai.research.policy`입니다. 계층형 평문·CKKS 서버/중계/기관 실행,
중증도·불균형·방어 실험은 `research list --domain policy`에 전부 포함됩니다.
계층형 실행은 공통 키·달력·기관 배정이 필요하며 키는 각 작업 폴더에서 생성합니다.
합성 자료 생성은 실제 공공 환자 자료의 확보와 구분합니다.

## 6. 감염세대간격 추정

```powershell
defai generation --input intervals.csv --output results/generation01.json
```

CSV 열은 `child_sigungu`(기관/지역의 정수 식별자), `T_days`(양수인 간격, 일)입니다.
기존 ABM 연구의 입력 필터 규칙은 연구 스크립트에 보존돼 있습니다. 결과에서 실제 사용 표본 수를 확인하세요.
출력은 Burr XII의 `c`, `k`, 일 단위 척도와 음의 로그우도입니다.
Python API는 `defai.research.generation`의 `build_hierarchy`, `federated_fit` 등입니다.

`abm_fl_simulation`의 E0→E1→E2→E3 실험은 각각 시뮬레이션, 정합성,
Paillier 암호화 집계, 연합 이득 비교입니다. 전체 ABM에는 별도의 인구·장소 입력이 필요합니다.
그 입력의 공공 출처는 아직 확정하지 않았습니다. 작은 키·고정 시드의 Paillier 자체검증은
수학적 구현 확인용이며 실서비스 암호 키 운용의 근거로 쓰지 않습니다.

## 7. 당화혈색소 개인별·모바일 연합학습

```powershell
defai research list --domain hba1c
defai data fetch public_inventory_05 --directory studies/session01/disease_research/04_당화혈색소_모바일연합학습/hba1c_fl_research
defai data fetch public_inventory_03 --directory studies/session01/disease_research/04_당화혈색소_모바일연합학습/hba1c_fl_research
```

현재 보정 입력은 NHANES 2017–2018과 Hall 2018입니다. 다른 수집 자료와 2021–2023 탐색 결과를
현재 보정에 모두 사용한 것으로 간주하지 않습니다. 파일별 URL·체크섬은 `references/`에 있습니다.
자료 사용조건은 각 원배포처에서 확인해야 합니다. 해시가 달라지면 다운로드를 실패 처리하고
기존 로컬 자료를 보존합니다.

`hba1c_fl_research/experiments/fl_comparison`의 `run_comparison.py`가
보정 → 원천 사람 분할 → 가상 종단 자료 → 검증 집합 기반 선택 → 보류 평가의 전체 연구 흐름입니다.
`--n`, `--seeds`, `--out`을 명시해 규모와 저장 위치를 조절합니다. 기본 20만 가상 사람·5시드는
작은 기능 확인용 설정이 아닙니다. 가상 인원은 실제 임상 대상자 수가 아닙니다.
`audit_worlds`, `verify_results`, `report_results`로 데이터·결과를 점검하고 보고합니다.
`gpu_execute` 같은 전용 GPU 도구는 `--device cuda`에서 사용합니다.

Python에서는 `defai.research.hba1c`의 `initial_parameters`, `local_fit`, `Learner`로
독립적인 기관별 갱신을 배치 처리할 수 있습니다. 모듈을 불러오는 것만으로
프로세스 스레드 수나 GPU 메모리 한도를 바꾸지 않습니다.

## 8. Android 앱과 서버

```powershell
defai mobile --output results/mobile01 --clients 2 --host 127.0.0.1 --port 8001
defai android --sdk C:/Users/YOUR_NAME/AppData/Local/Android/Sdk --output artifacts/defai-debug.apk
```

빌드에는 Android SDK와 JDK 17이 필요합니다. `--java-home`으로 JDK 위치를 지정할 수 있습니다.
빌드 도구는 코드가 한글 경로에 있어도 ASCII 임시 폴더에서 Gradle을 실행합니다.
검증한 Gradle 8.14.2 wrapper를 포함하며 앱 모듈은 `deFai_federated/defai/app`입니다.
루트 `src/`의 다른 Kotlin 가지는 과거 시제품으로 보존한 소스입니다.

앱에서 서버 호스트·포트를 입력합니다. Android 에뮬레이터에서 호스트 PC는 `10.0.2.2`이고,
실기기는 사용 환경에 맞는 접근 가능한 주소가 필요합니다.

서버는 NumPy 기반 선형 모델 파라미터를 검증·집계합니다. TensorFlow를 요구하지 않습니다.
각 클라이언트는 `/get_parameters`로 읽은 `round`를 `/submit_parameters`에 포함합니다.
같은 라운드의 중복·과거 제출은 거부하고, `--clients` 수가 모일 때 한 번 집계합니다.
`num_examples`를 보내면 그 수로 가중하며, 생략하면 동등 가중합니다.
부분 라운드와 모델은 `coordinator.json`에 원자적으로 저장되며 같은 설정으로 재시작할 수 있습니다.
같은 저장 폴더의 코디네이터는 한 서버 프로세스로 실행합니다. 서버는 직접 학습하거나
평가하지 않으므로 미측정 손실·오차를 0으로 보고하지 않습니다.

## 9. 성능·재현성·한계

`scripts/benchmark_aggregation.py`는 예전 배열 적층 방식과 현재 스트리밍 평균을
같은 합성 입력으로 비교합니다. 부동소수 결과 일치를 확인하고, 준비 실행 후 순서를 번갈아
시간 중앙값과 추적된 최대 추가 할당량을 측정합니다. 전체 프로세스 메모리나
임상 정확도, 실제 네트워크 속도에 대한 벤치마크는 아닙니다.

`catalog.json`은 현재 코드의 자원 목록과 SHA-256을 기록합니다.
`disease_research/source_manifest.json`, `PUBLICATION.md`, `VALIDATION.md`의 2026-10-29 기록은
최초 공개 스냅샷의 이력입니다. 0.2에서 개선한 소스의 현재 해시는 패키지 catalog가 기준입니다.

| 한 것 | 안 한 것 |
|---|---|
| 설치 가능한 패키지·통합 명령·Python API·전체 연구 파일 목록 제공 | 모든 과거 수치의 전체 재계산 |
| 합성 입력의 기능 연결·수치 정합성·오류 처리 검증 | 실제 임상·기관 간 배포 성능 검증 |
| Android 빌드 경로와 서버 프로토콜 개선 | 실기기·장시간 연결 운용의 완전한 검증 |
| 공공 데이터 출처·다운로드 체크섬 제공 | 전체 자료의 최신 이용조건 재심사·미확정 인구 입력의 출처 확정 |

따라서 제공 명령과 검증 결과는 소프트웨어를 일관되게 사용하는 토대이며, 연구 결과의
실전 적용에는 해당 데이터와 사용 환경의 별도 검증이 필요합니다.

용어: API는 Python에서 호출하는 함수·클래스 인터페이스, CLI는 명령행 인터페이스,
wheel은 Python 설치 파일, APK는 Android 설치 파일입니다. HbA1c는 당화혈색소이며,
값은 %, 값의 차이와 오차는 %p로 구분합니다.
