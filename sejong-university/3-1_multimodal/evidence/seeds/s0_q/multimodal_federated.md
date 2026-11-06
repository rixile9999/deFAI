# 트랜스포머 기반 멀티모달 학습 모델 실험 결과

설정: `{"part": "federated", "models": ["image_only", "text_only", "audio_only", "y1_joint_embedding", "y2_attention_fusion", "y3_mm_transformer", "y3_no_integration", "y3_no_decoder", "y3_mean_fusion"], "fed_model": "y3_mm_transformer", "fed_configs": ["qfedavg_flat"], "n_train": 6000, "n_test": 2000, "epochs": 15, "lr": 0.002, "batch_size": 64, "n_clients": 9, "n_per_client": 400, "alpha": 0.5, "rounds": 50, "local_epochs": 2, "fed_lr": 0.001, "q": 0.2, "qffl_lr": 10.0, "seed": 0, "device": "cuda", "out": "results/seeds/s0_q", "quick": false, "verbose": true}`

## 2. 연합 학습: FedAvg vs FedATC (다계층) vs FedATC + Transformer Compression

| 방식 | 정확도 | F1 | 결측30% | 그룹0 (이미지+텍스트) | 그룹1 (텍스트+오디오) | 그룹2 (전체) | 서버 수신 MB | 업데이트 압축률 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qfedavg_flat | 0.6615 | 0.6612 | 0.5035 | 0.4205 | 0.3805 | 0.6615 | 827.67 | - |
