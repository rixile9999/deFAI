# 2-3 MapReduce 기반 다계층 연합 분석

클라이언트, 그룹, 중앙 서버로 이루어진 계층 구조에서 분석 결과를 블록 단위 키값쌍으로 나누고, 표본 수로 가중해 집계합니다. 모델 업데이트는 Transformer 인코더로 압축한 잠재 표현과 주요 잔차만 보내고, 받는 쪽에서 복원한 업데이트를 실제 Flower 집계에 넣습니다.

3차년도 보고서 나-(3) 항목입니다. 코드는 `defai_research` 패키지([`src/defai_research`](src/defai_research))에 있고 테스트는 [`tests/research`](../../tests/research)에 있습니다. 단말 관계와 중복 제거는 [2-4 그래프 기반 중복 제거](../2-4_graph_deduplication/README.md)의 코드를 같은 패키지 안에서 함께 씁니다.

## 이전 차년도와의 관계

1·2차년도의 다계층 FedATC는 Transformer 압축으로 클라이언트 가중치를 압축해 통합하고, 데이터 크기와 엔트로피로 가중 평균하는 방식이었습니다. 이 가운데 클라이언트·그룹·중앙 계층 구조와 Transformer 압축 경로를 그대로 가져왔고, 그 위에 3차년도의 MapReduce 기반 다계층 연합 분석(블록 분할, 상위 계층을 키로 하는 키값쌍, 키별 평균)을 구현했습니다.

INT8만 쓰던 이전 실행기는 최종 방식과 겹쳐서 따로 두지 않았습니다. INT8은 잠재 표현과 잔차를 부호화할 때, 그리고 무손실 대조군의 입력 표현으로 남아 있습니다. 비교용으로 zlib/XOR 무손실 전송(`compact`, `legacy`)과 직접 가중 평균, 독립 shuffle 집계 검산도 유지했습니다.

## 구성

| 파일 | 내용 |
|---|---|
| `src/defai_research/aggregation.py` | 블록 MapReduce, 개수·합·제곱합 충분통계 집계 |
| `src/defai_research/compression/model.py` | Transformer 압축 모델, 학습·추론용 내보내기 |
| `src/defai_research/compression/transformer.py` | 잠재 표현·잔차 부호화와 수신 복원 |
| `src/defai_research/compression/runner.py` | 수신 배열을 그대로 쓰는 그룹·중앙 Flower 집계 |
| `src/defai_research/compression/calibration.py` | 학습 표본만으로 압축기 학습용 업데이트 수집 |
| `src/defai_research/audit.py` | 저장된 전송 장부, 제거 마스크, 모델 예측의 독립 검산 |
| `src/defai_research/cli.py` | `defai-research` 명령 |
| `scripts/verify_transformer_tcp.py` | 별도 프로세스가 TCP로 받은 업데이트를 복원해 Flower에 넣는지 검증 |

## 실행

```bash
defai-research verify-aggregation --full --output results/aggregation_check.json
defai-research collect-calibration --output results/calibration
defai-research train-codec --calibration results/calibration --output results/codec
defai-research benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --output results/digits_transformer
defai-research verify-run results/digits_transformer --output results/digits_transformer_check.json
python 2_analysis_speed/2-3_mapreduce_analysis/scripts/verify_transformer_tcp.py --artifact results/codec/inference/codec_model.bin --output results/transformer_tcp_check
```

기본 설정은 단말 12개, 그룹 4개, 80라운드이며 41라운드에 데이터 구성이 바뀝니다. Transformer는 d_model 8, 헤드 2개, 잠재 토큰 2개를 쓰고 잠재 표현은 INT8, 잔차는 상위 8개를 INT8로 보내며 블록 이득 보정을 합니다. 전체 전송량에는 양방향 모델 프레임, 그래프 제어 정보, 압축기 가중치 배포가 모두 들어갑니다.

처음 실행할 때는 `collect-calibration`과 `train-codec`으로 압축기를 먼저 학습해야 합니다. 동작만 빨리 확인하려면 다음처럼 규모를 줄입니다.

```bash
defai-research collect-calibration --rounds 4 --drift-round 3 --max-blocks-per-role 256 --output results/quick_calibration
defai-research train-codec --epochs 2 --calibration results/quick_calibration --output results/quick_codec
defai-research benchmark --rounds 4 --drift-round 3 --artifact results/quick_codec/inference/codec_model.bin --output results/quick_run
```

체크포인트는 검증 손실로 고릅니다. 시험 데이터 예측은 설정을 확정한 실행에서 `--evaluate-test`를 줄 때만 하고, 이미 있는 결과 폴더는 덮어쓰지 않습니다.

## KIST 시스템과의 관계

이 항목은 Flower의 `NumPyClient`와 `FedAvg.aggregate_fit`으로 자체 실험을 돌리는 분석 도구여서, KIST ServerApp의 전략으로 끼우는 형태는 아닙니다. KIST와 같은 Flower 1.42 환경에 함께 설치해 돌아가는 것은 확인했습니다(`tests/research` 37개 통과).

KIST에 붙일 수 있는 부분은 집계 함수입니다. `aggregation.py`의 블록 MapReduce와 충분통계 집계는 NumPy 배열만 받으므로 KIST 전략의 `aggregate_updates`를 덮어써서 쓸 수 있습니다. 방법은 [통합 가이드](../../docs/KIST_통합_가이드.md#5-kist-저장소를-수정해야-하는-부분)에 있습니다.

## 검증

- `tests/research/test_federated.py`: 그룹 크기가 다를 때 표본 수 가중, 충분통계와 모집단 통계 일치
- `tests/research/test_compression.py`: 어텐션 구조, 무손실 전송, 독립 수신기 일치, 손상 프레임 거부, 수신 복원 업데이트로 2단계 Flower 집계
- `tests/research/test_audit.py`, `test_source_layout.py`: 저장 결과 검산과 소스 목록
- 정리 이전 검증 기록(110개 벡터 집계 조건, 30개 충분통계 조건, 직접 평균 대비 최대 오차 3.33×10⁻¹⁶ 등): [docs/history/validation_mapreduce_graph_metrics_v1.0.1.md](../../docs/history/validation_mapreduce_graph_metrics_v1.0.1.md)
