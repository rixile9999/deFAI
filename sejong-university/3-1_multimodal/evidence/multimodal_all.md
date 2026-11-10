# 트랜스포머 기반 멀티모달 학습 모델 실험 결과

설정: `{"part": "all", "models": ["image_only", "text_only", "audio_only", "y1_joint_embedding", "y2_attention_fusion", "y3_mm_transformer", "y3_no_integration", "y3_no_decoder", "y3_mean_fusion"], "fed_model": "y3_mm_transformer", "fed_configs": ["central_ref", "fedavg_flat", "fedatc", "fedatc_compress"], "n_train": 6000, "n_test": 2000, "epochs": 20, "lr": 0.002, "batch_size": 64, "n_clients": 9, "n_per_client": 400, "alpha": 0.5, "rounds": 20, "local_epochs": 2, "fed_lr": 0.001, "seed": 0, "device": "cuda", "out": "results", "quick": false, "verbose": true}`

## 1. 중앙 학습: 1·2·3차년도 모델 및 절제 실험

| 모델 | 파라미터 | 정확도 | F1 | 결측30% | 이미지 결측 | 텍스트 결측 | 오디오 결측 | 이미지→텍스트 검색 R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 480234 | 0.4765 | 0.4720 | 0.4765 | 0.4765 | 0.4765 | 0.4765 | 0.255 |
| text_only | 480234 | 0.3415 | 0.3413 | 0.3415 | 0.3415 | 0.3415 | 0.3415 | 0.144 |
| audio_only | 480234 | 0.3070 | 0.3064 | 0.3070 | 0.3070 | 0.3070 | 0.3070 | 0.155 |
| y1_joint_embedding | 292327 | 0.8665 | 0.8663 | 0.6670 | 0.4620 | 0.8540 | 0.5955 | 0.363 |
| y2_attention_fusion | 347239 | 0.8735 | 0.8738 | 0.6710 | 0.5125 | 0.8635 | 0.6025 | 0.302 |
| y3_mm_transformer | 480234 | 0.8885 | 0.8888 | 0.6760 | 0.5125 | 0.8700 | 0.6075 | 0.361 |
| y3_no_integration | 480234 | 0.8840 | 0.8839 | 0.6700 | 0.4755 | 0.8655 | 0.6190 | 0.250 |
| y3_no_decoder | 413354 | 0.8825 | 0.8826 | 0.6770 | 0.4940 | 0.8630 | 0.6135 | 0.312 |
| y3_mean_fusion | 430698 | 0.8890 | 0.8892 | 0.6885 | 0.5155 | 0.8700 | 0.6090 | 0.282 |

## 2. 연합 학습

Flower 기반으로 다시 실행한 결과는 `multimodal_federated.md` 참조 (이 파일의 이전 연합 학습 결과는 Flower 전환 전 구현이라 삭제).
