# 트랜스포머 기반 멀티모달 학습 모델 실험 결과

설정: `{"part": "federated", "models": ["image_only", "text_only", "audio_only", "y1_joint_embedding", "y2_attention_fusion", "y3_mm_transformer", "y3_no_integration", "y3_no_decoder", "y3_mean_fusion"], "fed_model": "y3_mm_transformer", "fed_configs": ["central_ref", "fedavg_flat", "fedatc", "fedatc_compress"], "n_train": 6000, "n_test": 2000, "epochs": 20, "lr": 0.002, "batch_size": 64, "n_clients": 9, "n_per_client": 400, "alpha": 0.5, "rounds": 20, "local_epochs": 2, "fed_lr": 0.001, "seed": 0, "device": "cuda", "out": "results", "quick": false, "verbose": true}`

## 2. 연합 학습: FedAvg vs FedATC (다계층) vs FedATC + Transformer Compression

| 방식 | 정확도 | F1 | 결측30% | 그룹0 (이미지+텍스트) | 그룹1 (텍스트+오디오) | 그룹2 (전체) | 서버 수신 MB | 업데이트 압축률 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| central_ref | 0.8225 | 0.8220 | 0.6185 | 0.5770 | 0.4265 | 0.8225 | - | - |
| fedavg_flat | 0.6310 | 0.6253 | 0.5010 | 0.4570 | 0.4005 | 0.6310 | 331.07 | - |
| fedatc | 0.5070 | 0.4557 | 0.4165 | 0.3640 | 0.3840 | 0.5070 | 110.36 | 1.0x |
| fedatc_compress | 0.7110 | 0.7093 | 0.5490 | 0.4925 | 0.4205 | 0.7110 | 16.93 | 6.5x |

### FedATC 다계층 연합 분석 (Transformer Compression 된 분석 결과 → 서버 프로토타입 분류)

| 항목 | 값 |
|---|---:|
| 프로토타입 분류 정확도 (원본 분석 결과) | 0.4720 |
| 프로토타입 분류 정확도 (압축→복원 분석 결과) | 0.4595 |
| 분석 결과 누적 압축률 (재학습 라운드의 원본 전송 포함) | 3.96x |
| 압축 라운드 평균 상대 복원 오차 | 0.526 |
