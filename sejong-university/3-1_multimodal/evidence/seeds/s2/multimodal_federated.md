# 트랜스포머 기반 멀티모달 학습 모델 실험 결과

설정: `{"part": "federated", "models": ["image_only", "text_only", "audio_only", "y1_joint_embedding", "y2_attention_fusion", "y3_mm_transformer", "y3_no_integration", "y3_no_decoder", "y3_mean_fusion"], "fed_model": "y3_mm_transformer", "fed_configs": ["fedavg_flat", "fedatc", "fedatc_compress"], "n_train": 6000, "n_test": 2000, "epochs": 15, "lr": 0.002, "batch_size": 64, "n_clients": 9, "n_per_client": 400, "alpha": 0.5, "rounds": 50, "local_epochs": 2, "fed_lr": 0.001, "seed": 2, "device": "cuda", "out": "results/seeds/s2", "quick": false, "verbose": true}`

## 2. 연합 학습: FedAvg vs FedATC (다계층) vs FedATC + Transformer Compression

| 방식 | 정확도 | F1 | 결측30% | 그룹0 (이미지+텍스트) | 그룹1 (텍스트+오디오) | 그룹2 (전체) | 서버 수신 MB | 업데이트 압축률 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fedavg_flat | 0.4285 | 0.3884 | 0.3490 | 0.2585 | 0.3950 | 0.4285 | 827.67 | - |
| fedatc | 0.6105 | 0.5710 | 0.4835 | 0.4190 | 0.3910 | 0.6105 | 275.89 | 1.0x |
| fedatc_compress | 0.6795 | 0.6765 | 0.5140 | 0.4960 | 0.4145 | 0.6795 | 42.32 | 6.5x |

### FedATC 다계층 연합 분석 (Transformer Compression 된 분석 결과 → 서버 프로토타입 분류)

| 항목 | 값 |
|---|---:|
| 프로토타입 분류 정확도 (원본 분석 결과) | 0.5670 |
| 프로토타입 분류 정확도 (압축→복원 분석 결과) | 0.5555 |
| 분석 결과 누적 압축률 (재학습 라운드의 원본 전송 포함) | 3.96x |
| 압축 라운드 평균 상대 복원 오차 | 0.577 |
