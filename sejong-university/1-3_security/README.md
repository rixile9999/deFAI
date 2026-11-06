# 1-3 보안 공격·방어 시뮬레이션

3차년도 보고서 가-(3) 보안 시뮬레이션 환경 고도화 항목의 코드입니다. 연합학습의 클라이언트, 엣지(어시스턴트 노드), 중앙 서버 구간에서 생기는 공격을 재현하고, 방어 기법을 켰을 때의 탐지 성능과 모델 성능, 네트워크 비용을 함께 측정합니다.

단일 서버 연합학습(KIST 시스템 포함)과 계층형 시뮬레이터(`hierarchical_fl`) 어디에든 `SecurityPlugin` 하나로 붙습니다. 방어는 Flower `Strategy`를 감싸는 `DefendedStrategy`이고 공격과 서명은 ClientApp 핸들러 앞뒤에서 동작하는 미들웨어여서, KIST 전략과 `defai.client_app.train`에도 그대로 붙습니다. `flwr run` 배포에서는 `fl_core.kist_apps`의 `sejong-plugins` 설정으로 켭니다([통합 가이드](../../docs/KIST_통합_가이드.md)).

패키지는 `security_sim`([`src/security_sim`](src/security_sim)), 테스트는 [`tests/modules/test_security.py`](../../tests/modules/test_security.py)입니다.

## 1~3차년도 공격과 방어

| 연차 | 공격 시나리오 | 대응 방안 | 구현 (공격 / 방어) |
|---|---|---|---|
| 1차 | 데이터 조작(클라이언트 데이터 노이즈) | 데이터 분포 분석으로 노이즈 구분 | `data_noise` / `data_dist` |
| 1차 | 모델 변조(파라미터 노이즈 삽입) | 파라미터 통계 분석 이상 패턴 감지 | `model_poison`, `sign_flip` / `stat` |
| 1차 | 중앙 서버 공격(가짜 클라이언트 다수 생성) | 유효 인증서 적용 | `sybil` / `cert` |
| 1차 | 보안 시나리오별 네트워크 비용 산출 | 통신 로그 수집·이상 징후 탐지 | 라운드별 `net_cost_mb`, `sec_overhead_kb`, 이벤트 로그 |
| 2차 | 데이터 중계 공격(어시스턴트 노드의 중계 데이터 변조) | 데이터 검증·HMAC-SHA256 서명 무결성 확인 | `relay_tamper`(클라이언트→엣지), `edge_relay`(엣지→중앙) / `hmac` |
| 2차 | 계층적 협력 공격(다수 어시스턴트 노드 협력) | K-Means 클러스터링 담합 탐지 | `hier_collusion` / `kmeans` |
| 2차 | 다단계 피싱(어시스턴트 노드 인증 정보 탈취) | 실시간 모니터링·경고, 인증 강화 | `phishing` / `mfa`, `monitor` |
| 2차 | 비정상 데이터 패턴 | One-Class SVM 이상치 탐지 | `ocsvm` |
| 3차 | 내부자 협력 공격(악의적 파라미터 공동 제출) | 내부자 활동 모니터링·로그 분석, 권한 최소화 | `insider_collusion`, `adaptive_collusion` / `monitor` |
| 3차 | 모델 역추적 공격(API 출력 수집으로 모델 추출·입력 복원) | 출력 제한, 랜덤 노이즈(차분 프라이버시), 질의 한도 | `inversion.py` / `top1`, `round`, `noise`, `limit` |

연차별 공격과 방어는 서로 대체하는 관계가 아니어서 모두 남겼습니다. 2차의 "다계층적 구조의 보안 시험 환경"은 2계층 방어(`two-tier`)로 구현했는데, 계층형 시뮬레이터에서 엣지 계층(클라이언트 업데이트)과 전역 계층(엣지 모델)에 같은 방어를 거는 방식입니다. 탐지하는 대신 피해를 줄이는 보조 방어로 `clip`(중앙값 노름 기준 업데이트 클리핑)도 있습니다.

## 모듈 구성

- `attacks.py`: ClientApp 미들웨어(모델 변조, 중계 변조, 서명), 데이터 변환 훅(노이즈, 레이블 뒤집기), 공모/시빌 공격 전략 래퍼, 엣지 모델 변조
- `keys.py`: 인증서(CA HMAC), 페이로드 HMAC-SHA256, 기기 비밀 기반 OTP(2차 인증). 응답 Message에 `auth` ConfigRecord로 붙입니다.
- `detectors.py`: robust z-score(노름, 방향), 데이터 분포 z-score, One-Class SVM(누적된 정상 이력으로 학습), K-Means 공모 군집, 내부자 모니터(중복, 기준 방향, 크기), 정밀도/재현율
- `defense.py`: `DefendedStrategy`가 임의의 Flower Strategy(KIST 전략 포함)를 감싸 인증, 통계 탐지, 거부 또는 클리핑, 내부 집계 순서로 처리하고 탐지 로그를 남깁니다.
- `plugin.py`: `SecurityPlugin`을 단일 서버(`flat`), 엣지(`edge`), 전역(`global`) 계층별로 적용합니다. 엣지 응답 서명, 엣지에서 중앙으로 가는 구간의 공격, 라운드별 네트워크 비용과 탐지 집계도 여기서 처리합니다.
- `inversion.py`: Flask 예측 API(`/api/v1/predict`)를 띄운 뒤, 공격자가 질의만으로 대리 모델을 추출(knockoff)하고 클래스별 입력을 역추적(MI-FACE)합니다. 방어마다 API 정확도, 추출 충실도, 복원 유사도, 복원 성공률을 비교합니다.

## 실행

```bash
python -m security_sim run --attack model_poison --defense stat,hmac
python -m security_sim run --mode hier --attack hier_collusion --defense kmeans --clients 9 --edges 3
python -m security_sim run --mode hier --attack phishing --defense mfa,monitor
python -m security_sim run --backend kist --attack sign_flip --defense stat
python -m security_sim matrix --groups y1,y2,y3
python -m security_sim inversion --queries 2000
```

파이썬에서는 다음과 같이 씁니다.

```python
from fl_core.backends import get_backend
from fl_core.plugins import run_flat
from security_sim import SecurityPlugin

plugin = SecurityPlugin({"attack": "insider_collusion", "ratio": 0.2, "defense": ["monitor", "hmac", "cert"]})
out = run_flat(get_backend("kist", {"dataset": "synthetic"}), [plugin], num_clients=10, rounds=5)
print(plugin.summary(None)["client_detection"], out["alerts"][:3])
```

계층형 시뮬레이터에서는 `HierarchicalServer({...}, plugins=[SecurityPlugin({...})]).run()`처럼 넘기며, 플러그인 설정은 Flask 엣지 프로세스에도 전달됩니다. 설정 키는 `attack`, `ratio`(악성 클라이언트 비율), `defense`(목록), `two-tier`, `scale`, `noise-std`, `param-std`, `relay-std`, `sybils`, `malicious-edges`, `phish-round`, `start-round`, `secret`, `seed`입니다.

`run`은 `results/security/<mode>_<attack>_<defense>.json`에 라운드별 정확도, 탐지 수, 네트워크 비용, 경고 로그, 탐지 정밀도/재현율을 저장합니다. `matrix`는 공격별로 무방어와 방어를 비교한 표를 `results/security/matrix.csv`에, `inversion`은 결과를 `results/security/inversion.json`에 씁니다.

## 측정 결과

`scripts/run_demo.py` 기본 설정(synthetic 데이터, 3라운드 안팎의 짧은 실행, CPU i9-11900F)에서 잰 값입니다. 짧은 실행이라 절대값보다는 같은 설정 안의 비교로 읽어야 합니다.

아래 표는 `python -m security_sim matrix` 결과입니다(클라이언트 9, 악성 비율 0.2, 3라운드, 계층형은 엣지 3). 정밀도와 재현율은 전체 라운드에서 거부된 주체를 기준으로 계산했습니다.

| 공격 | 방어 | 최종 정확도 (무방어 → 방어) | 탐지 정밀도 / 재현율 |
|---|---|---|---|
| 없음 (기준) | - | 0.6045 | - |
| data_noise | data_dist | 0.596 → 0.5965 | 0.67 / 1.0 |
| model_poison | stat | 0.107 → 0.6845 | 0.50 / 1.0 |
| sybil (가짜 3개) | cert | 0.107 → 0.600 | 1.0 / 1.0 |
| relay_tamper (클라이언트→엣지) | hmac | 0.3605 → 0.568 | 1.0 / 1.0 |
| edge_relay (엣지→중앙, 계층형) | hmac | 0.099 → 0.9225 | 1.0 / 1.0 (엣지) |
| hier_collusion (엣지 2개 공모, 계층형) | kmeans | 0.709 → 0.874 | 1.0 / 1.0 (엣지) |
| phishing (탈취 자격증명 위조, 계층형) | mfa+monitor | 0.1755 → 0.9015 | 1.0 / 1.0 (엣지) |
| insider_collusion | monitor | 0.1005 → 0.5675 | 1.0 / 1.0 |
| adaptive_collusion | monitor+stat | 0.2385 → 0.5655 | 1.0 / 1.0 |
| model_poison | ocsvm | 0.10 → 0.836 (4라운드, 개선된 OCSVM 별도 실행) | 0.50 / 1.0 |

이 데이터에서는 데이터 노이즈 공격이 정확도에 준 영향이 작았습니다. 통계 기반 방어(stat, ocsvm, data_dist)는 클라이언트 수가 적으면 정상 클라이언트도 일부 거부해서 정밀도가 0.5~0.67에 머뭅니다.

모델 역추적은 `python -m security_sim inversion --queries 1000`으로 측정했습니다. 공격자 질의는 1,000건, 피해 모델 정확도는 0.95입니다.

| 출력 방어 | API 정확도 | 응답 질의 | 추출 충실도 | 대리 모델 정확도 | 복원 유사도 | 복원 성공률 |
|---|---|---|---|---|---|---|
| none | 0.950 | 1000 | 0.767 | 0.773 | 0.229 | 1.0 |
| top1 (레이블만) | 0.950 | 1000 | 0.111 | 0.108 | 0.000 | 0.1 |
| round (소수 1자리) | 0.950 | 1000 | 0.798 | 0.809 | 0.184 | 0.7 |
| noise (Laplace, ε=10) | 0.872 | 1000 | 0.111 | 0.108 | 0.000 | 0.1 |
| limit (질의 한도 250) | 0.950 | 200 | 0.559 | 0.558 | 0.330 | 0.5 |
