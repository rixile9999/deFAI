# 2-4 그래프 기반 동적 클러스터링 중복 제거

로컬 단말과 군집 중심의 연결 관계를 GraphSAGE로 학습해 중복 데이터가 있을 법한 단말 후보를 고르고, 후보끼리 해시와 배열을 비교해 실제로 같은 사본만 지웁니다. 데이터가 바뀌면 후보 그래프를 다시 학습하고, 바뀌지 않았으면 이전 식별 결과를 캐시에서 꺼내 씁니다.

3차년도 보고서 나-(4) 항목입니다. 코드는 `defai_research.federated`([`src/defai_research/federated`](src/defai_research/federated))이고 테스트는 [`tests/research/test_federated.py`](../../tests/research/test_federated.py)입니다.

1차년도의 유사도 기반 중복 분석 데이터 압축(SID)은 특징 추출, 차원 축소, DBSCAN 군집, 유사도 측정으로 이루어졌고, 그중 유사도로 후보를 고르는 방향이 최신 경로에 반영되었습니다. 2차년도의 동적 클러스터링 중복 분석 제거(DCNI)에서는 새 데이터가 들어오면 노드 영역을 다시 계산했는데, 이 부분은 데이터 요약과 데이터량이 바뀔 때 후보 그래프를 갱신하는 방식으로 남았습니다. 최신 구현은 3차년도의 그래프 모델링으로, 로컬 단말 구조의 연관성을 반영하고 GraphSAGE(가중 이웃 샘플링, 풀링 집계, 정·부정 표본 손실)를 씁니다.

해시 스케치만으로 후보를 고르던 이전 실행기는 GraphSAGE 경로로 대체했고, 정적 그래프는 동적 갱신의 효과를 보기 위한 비교 조건으로 남겼습니다. 유사도는 후보를 고를 때만 쓰며, 실제 삭제는 SHA256과 특징·레이블 배열이 모두 같은 사본으로 한정합니다.

## 코드

- `graph.py`: GraphSAGE 연관성 학습, 후보 그래프, 정확 일치 검증(`GraphDeduplicator`)
- `cache.py`: 동적 갱신, 버전별 식별 정보와 제거 마스크 캐시(`GraphSAGECachedDeduplicator`)
- `snapshots.py`: 불변 학습 스냅샷, digest, 소유권 제어 메시지(`ImmutableTrainingSnapshot`)
- `data.py`: 고정된 학습·검증·시험 분할, 단말 데이터 변화 조건
- `learning.py`: 공유 Flower 클라이언트, 학습, 고유 표본 보존 검사
- `examples/compare_methods.py`: 기준 전송, compact 전송, 정적·동적 Transformer 방식 비교
- `examples/plot_results.ipynb`: 비교 결과 CSV 시각화

`ImmutableTrainingSnapshot.from_arrays`로 만든 스냅샷을 라운드마다 `GraphSAGECachedDeduplicator.apply`에 넘겨서 씁니다. 중복을 지우면 학습 스텝이 줄어들기 때문에 그에 맞춰 학습률을 보정합니다. `data.py`, `learning.py`, `snapshots.py`는 [2-3](../2-3_mapreduce_analysis/README.md)과 [3-2](../../3_heterogeneous_data/3-2_fl_metrics/README.md)에서도 함께 씁니다.

## 실행

압축기 파일은 [2-3의 학습 절차](../2-3_mapreduce_analysis/README.md#실행)를 따라 먼저 만들어 둡니다.

```bash
defai-research benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --output results/graph_dynamic
defai-research benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --static --output results/graph_static
python 2_analysis_speed/2-4_graph_deduplication/examples/compare_methods.py --artifact results/codec/inference/codec_model.bin --output results/comparison
```

반복 비교는 기본적으로 데이터셋 2개, 시드 12개, 방식 4개를 조합해 96회 실행합니다. 전송량과 함께 데이터가 바뀐 뒤의 중복 사본 제거율, 학습량, 분류 성능을 봅니다.

## KIST 시스템에 넣으려면

`GraphSAGECachedDeduplicator`는 클라이언트 데이터 배열을 받아 제거 마스크를 돌려주므로, KIST ClientApp이 학습하기 전에 로컬 데이터를 거르는 단계로 쓸 수 있습니다. 다만 그래프를 학습하려면 모든 단말의 데이터 요약이 있어야 해서, 실제 배포에서는 요약을 모으는 라운드를 따로 두어야 합니다. 자세한 방법은 [통합 가이드](../../docs/KIST_통합_가이드.md#5-kist-저장소를-수정해야-하는-부분)에 정리했습니다.

## 검증

`tests/research/test_federated.py`는 정확 일치 검사와 불변 저장, 변하지 않은 스냅샷의 캐시 재사용과 바뀐 스냅샷의 재학습, 정적 그래프 고정, 손상된 제어 프레임 거부를 확인합니다. 정리 이전의 검증 기록은 [docs/history/validation_mapreduce_graph_metrics_v1.0.1.md](../../docs/history/validation_mapreduce_graph_metrics_v1.0.1.md)에 있습니다.
