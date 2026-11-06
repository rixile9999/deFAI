# 2-1 연산 그래프 기반 모델 가지치기

모델을 `torch.fx`로 추적해 연산 그래프를 만들고, 채널 의존성 그룹마다 노드와 엣지의 경로 중요도를 따져 구조적 가지치기를 합니다. 손실 변화를 지켜보면서 임계값도 동적으로 조정합니다. 3차년도 보고서 나-(1) 항목이며 패키지는 `graph_pruning`([`src/graph_pruning`](src/graph_pruning)), 테스트는 [`tests/modules/test_graph_pruning.py`](../../tests/modules/test_graph_pruning.py)입니다.

쓰는 방법은 두 가지입니다. 중앙(서버)에서 단독으로 실행하거나, 연합학습 도중에 전역 모델을 가지치기하는 `GraphPruningStrategy`(Flower Strategy 래퍼)를 씁니다.

## 이전 차년도 방법과의 관계

3차년도의 연산 그래프 기반 가지치기(연산 그래프 생성, 노드·엣지 경로 중요도 평가, 임계값 미만 제거, 경험적 초기 임계값을 두고 손실 변화를 감시해 동적으로 조정)가 최신 구현이고, 아래 알고리즘 전체가 여기에 해당합니다.

1차년도 설계는 임계값 이하의 뉴런과 필터를 지우는 방식과 가중치 절대값 기준으로 지우는 방식이었습니다. 필터 단위 제거는 3차 구조적 가지치기로 대체했고, 가중치 절댓값 기준 균일 L1 필터 가지치기는 비교 기준(`uniform_l1`)으로 남겼습니다. 이 기준은 파라미터 수를 맞추려고 이분 탐색을 씁니다.

2차년도에는 BN 스케일 인자 L1 정규화(구조적 희소성), 기울기×활성 기반 중요도, 학습 중 임계값 조정을 구현했습니다. BN-L1 희소성 정규화는 로컬 학습 손실 훅(`--bn-l1`, `bn-l1`)으로 남아 있고, 기울기 기반 중요도는 1차 테일러 중요도(`--importance taylor`)로 쓸 수 있습니다. 임계값 조정만 3차의 동적 조정으로 대체했습니다.

## 알고리즘

1. 연산 그래프 생성(`graph.py`): `torch.fx` 추적과 형상 전파로 그래프를 만듭니다. Conv/Linear 출력이 채널 공간을 만들고 BN, 활성, 풀링, flatten은 그대로 통과합니다. 잔차 덧셈은 union-find로 공간을 합쳐 결합 그룹을 만들고, 입력과 출력, 그룹 합성곱, 공유 모듈은 고정합니다.
2. 경로 중요도: 채널 c마다 들어오는 가중치(생산자)와 나가는 가중치(소비자)의 정규화 제곱합에 BN |γ| 평균을 곱합니다. 그룹 평균이 1이 되도록 정규화한 뒤, 가지치기 전 모델의 척도에 고정(anchor)합니다.
3. 가지치기: 중요도가 τ 이상인 채널만 남기되 그룹마다 최소 max(4, ⌈0.1·C₀⌉)개는 유지합니다. Conv/BN/Linear를 물리적으로 다시 구성하고, 원래 채널 인덱스(`prune_meta`)를 누적해 기록합니다.
4. 임계값 동적 조정(`dynamic.py`): 중요도 분위수를 경험적 초기값 τ₀로 잡고 가지치기한 다음, 짧게 회복 학습을 하고 검증 손실 변화 d = (L − L_ref)/L_ref를 봅니다. d ≤ tol이면 수락하고 τ ← τ(1+η)로 올립니다. 그렇지 않으면 직전 모델로 되돌리고 η ← η/2로 줄입니다.
5. 연합학습 적용(`federated.py`): 전역 집계가 끝난 뒤 지정한 라운드마다 중요도와 임계값으로 채널 마스크를 갱신합니다. 그다음 라운드들의 연합 학습이 회복 학습 역할을 하고, 다음 시점에 손실 변화를 확인해 수락하거나 롤백하면서 η를 조정합니다. 마스크 방식이어서 KIST ClientApp 같은 기존 클라이언트도 같은 모델 구조로 학습하며, 끝난 뒤 `final_models()`로 물리적으로 줄인 모델을 얻습니다. 이 모델은 마스크 모델과 출력이 같습니다.

KIST 시스템에서는 `GraphPruningStrategy`가 KIST 전략을 감싸 전역 모델에 채널 마스크를 씌웁니다. KIST ClientApp은 원래 `Net` 구조 그대로 학습하고, 가지치기를 수락할지 되돌릴지는 검증 손실 증가율(`tol`)과 정확도 하락(`acc-tol`)을 함께 보고 정합니다.

## 실행

```bash
python -m graph_pruning graph --set model=resnet8
python -m graph_pruning central --set model=cnn_bn --bn-l1 1e-4 --iters 6
python -m graph_pruning central --importance taylor --target 0.7
python -m graph_pruning federated --clients 8 --rounds 8 --start 2 --every 2
python -m graph_pruning federated --backend kist
```

```python
from graph_pruning.federated import GraphPruningPlugin
from fl_core.plugins import run_flat
from fl_core.backends import get_backend

plugin = GraphPruningPlugin({"start": 2, "every": 2, "percentile": 30, "tol": 0.15})
out = run_flat(get_backend("kist", {"dataset": "synthetic"}), [plugin], num_clients=8, rounds=8)
masked, compact = plugin.strategy.final_models(out["result"].arrays)
```

계층형 시뮬레이터에서는 전역 계층(`tiers=["global"]`)에서 동작합니다.

결과는 `results/graph_pruning/`에 저장됩니다. `central_pruning.json`에는 반복마다 τ, η, 제거한 채널, 파라미터와 MACs 감소율, 검증 손실, 수락 여부가 남고 기준 모델, 그래프 가지치기, 균일 L1을 비교한 표도 들어갑니다. 그 밖에 구조 사양까지 담은 `pruned_model.pt`/`.json`, `federated_pruning.json`, `federated_pruned_model.pt`가 생깁니다.

## 측정 결과

`scripts/run_demo.py` 기본 설정(synthetic 데이터, 3라운드 안팎의 짧은 실행, CPU i9-11900F)에서 잰 값이므로 절대값보다는 같은 설정 안의 비교로 보아야 합니다.

중앙 실행은 `python -m graph_pruning central --set model=cnn_bn --bn-l1 1e-4 --iters 4`입니다.

| 모델 | 파라미터 | MACs | 지연시간(배치 1) | 테스트 정확도 |
|---|---|---|---|---|
| 기준 (BN-L1 학습) | 1,663,562 | 12.27 M | 1.41 ms | 0.973 |
| 연산 그래프 + 동적 임계값 | 454,880 (−72.7%) | 8.80 M (−28.3%) | 0.92 ms | 0.987 |
| 균일 L1 (같은 파라미터 수) | 444,402 | 3.51 M | 1.16 ms | 0.990 |

동적 조정 과정에서 τ가 0.27, 0.40, 0.61일 때는 수락했고, 0.91에서는 손실 증가율 0.29가 tol 0.15를 넘어 거부하고 되돌렸습니다.

연합 실행(KIST 호환 마스크 방식)은 `python -m graph_pruning federated --rounds 5 --clients 6`입니다. 2라운드에 183채널을 가지치기했고 4라운드에 손실을 확인한 뒤 수락했습니다(손실 변화 −79%). 최종 축소 모델은 파라미터 103,588개(−93.8%)이고 지연시간은 0.92 ms에서 0.40 ms로 줄었습니다. 마스크 모델과 축소 모델의 정확도는 둘 다 0.917입니다.
