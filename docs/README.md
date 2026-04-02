# Documentation

## Published Paper

**RT-MLIDS: Real-Time Machine Learning Intrusion Detection System**

> Brighouse, I. (2024). *RT-MLIDS: A Real-Time Machine Learning Intrusion Detection System using Stacked Ensemble Learning with Adversarial Robustness Evaluation*. University of East London.

📄 **[Read the full paper on ResearchGate](https://doi.org/10.13140/RG.2.2.16609.47201/1)**

```
DOI: 10.13140/RG.2.2.16609.47201/1
```

---

## Key Results

### Classification Performance (NSL-KDD KDDTest+)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 77.89% | 96.65% | 63.35% | 76.54% |
| XGBoost | 80.41% | 96.83% | 67.81% | 79.76% |
| SVM | 79.02% | 97.31% | 64.94% | 77.90% |
| **Stacked Ensemble** | **80.28%** | **96.85%** | **67.56%** | **79.60%** |

### Throughput Benchmark

| Model | Latency (μs) | Throughput (flows/sec) |
|-------|-------------|------------------------|
| XGBoost | 2.62 | 382,013 |
| Random Forest | 148.70 | 6,725 |
| Stacked Ensemble | 191.87 | 5,212 |
| SVM | 1,072.61 | 932 |

### Adversarial Robustness (TABLE 3)

| Attack | F1-Score | Δ vs Clean |
|--------|----------|------------|
| Clean baseline | 0.8000 | — |
| HopSkipJump | 0.2000 | −75.00% |
| ZooAttack | 0.6500 | −18.75% |

---

## Methodology

- **Dataset**: NSL-KDD (KDDTrain+ → KDDTest+, harder complete test set)
- **Feature selection**: Mutual Information Gain (MIG), 53 → 30 features
- **Class imbalance**: SMOTE oversampling (U2R class: 52 training samples)
- **Architecture**: RF(500) + XGBoost(300) base learners, Logistic Regression meta-learner, 5-fold CV
- **Adversarial evaluation**: HopSkipJump and ZooAttack black-box evasion attacks
- **Interpretability**: SHAP analysis — top feature `src_bytes` (mean |SHAP| = 4.345)
- **Deployment**: Apache Kafka streaming pipeline, sliding buffer of 512 flows
