# 연합학습 과제 보고서

## 1. 개요
- 과제 목표:
- 사용 데이터셋 / 모델:
- 프레임워크: Flower

## 2. 실험 설정
| 항목 | 값 |
|---|---|
| SuperNodes (전체 / 라운드당 비율) | 10 / 0.5 |
| Rounds | 10 |
| Local epochs | 1 |
| Batch size / LR | 32 / 0.01 |
| Strategy | FedAvg / FedProx (μ) / SCAFFOLD / MOON / FedAvgM / FedAdam ... |
| Partition | IID / Dirichlet(α) |
| Seeds | |
| Attack (type / # malicious / scale) | none |
| Defence | none / FedMedian / FedTrimmedAvg / Krum / Bulyan |
| Differential privacy (mode / z / C / δ → ε) | none |

## 3. 결과
<!-- python scripts/plot.py results/*/history.json -o docs/acc.png -->
<!-- python scripts/compare.py results/*/history.json   (mean ± std over seeds) -->

## 4. 분석 및 고찰

## 5. 참고문헌
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020.
- Karimireddy et al., *SCAFFOLD*, ICML 2020. / Li et al., *Model-Contrastive Federated Learning* (MOON), CVPR 2021.
- Reddi et al., *Adaptive Federated Optimization*, ICLR 2021. / Hsu et al., *Measuring the Effects of Non-Identical Data Distribution* (FedAvgM), 2019.
- Yin et al., *Byzantine-Robust Distributed Learning*, ICML 2018. / Blanchard et al., *Machine Learning with Adversaries* (Krum), NeurIPS 2017. / El Mhamdi et al., *The Hidden Vulnerability of Distributed Learning in Byzantium* (Bulyan), ICML 2018.
- McMahan et al., *Learning Differentially Private Recurrent Language Models*, ICLR 2018. / Andrew et al., *Differentially Private Learning with Adaptive Clipping*, NeurIPS 2021.
- Beutel et al., *Flower: A Friendly Federated Learning Research Framework*, arXiv:2007.14390, 2020.
