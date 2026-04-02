# RT-MLIDS
### Real-Time Ensemble Machine Learning Framework for Network Intrusion Detection with Adversarial Robustness Evaluation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Institution](https://img.shields.io/badge/UEL-Cybersecurity%20%26%20Networks-003087?style=flat-square)](https://uel.ac.uk)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Paper-00CCBB?style=flat-square&logo=researchgate&logoColor=white)](docs/RT_MLIDS_Final_ResearchGate.pdf)

---

RT-MLIDS is a real-time network intrusion detection framework that combines **Random Forest** and **XGBoost** in a stacked ensemble architecture, integrated within a **streaming pipeline** built on Apache Kafka. It is designed for deployment in enterprise environments where high throughput, low latency, and adversarial robustness are required.

A key contribution of this work is the **adversarial robustness evaluation** using HopSkipJump and ZooAttack black-box evasion strategies — a dimension largely absent from comparable ML-IDS literature. Results reveal a critical 75% accuracy drop under HopSkipJump, with direct implications for production IDS deployment.

Evaluated on the **NSL-KDD** benchmark (complete KDDTest+ set, not the simplified KDDTest-21 used by most papers).

> **Published:** ResearchGate · Ian Alexander Brighouse Quintana · University of East London, Department of Cybersecurity and Networks

---

## Performance

### Classification (NSL-KDD KDDTest+ — Complete Benchmark)

| Model | Accuracy | Precision | F1-Score |
|---|---|---|---|
| **RT-MLIDS (Stacked Ensemble)** | **80.28%** | **96.85%** | **79.60%** |
| XGBoost | 79.60% | 96.61% | 79.76% |
| Random Forest | — | — | — |

> **Note on methodology:** Most published works reporting 99%+ on NSL-KDD use the simplified KDDTest-21 subset. RT-MLIDS is evaluated on the complete KDDTest+, which includes harder attack variants absent from training — producing more conservative and reproducible real-world estimates.

RT-MLIDS achieves the highest **precision (96.85%)** of all evaluated models, directly minimising false-positive-driven alert fatigue in operational SOC environments.

### Latency and Throughput (Batch Size = 512)

| Model | Latency/Flow | Throughput |
|---|---|---|
| **XGBoost** | **2.62 µs** | **382,013 flows/sec** |
| RT-MLIDS (Stacked) | 191.87 µs | 5,212 flows/sec |
| SVM | 1,072 µs | — |

XGBoost achieves **382,013 flows/second** — fully viable for real-time deployment. The stacked ensemble incurs modest additional latency due to the meta-learner pass, but remains within operational IDS requirements.

### Adversarial Robustness (Black-Box Evasion)

| Attack | Clean Accuracy | Accuracy Under Attack | Robustness Drop |
|---|---|---|---|
| **HopSkipJump** | 80.00% | 20.00% | **−75.00%** |
| **ZooAttack** | 80.00% | 65.00% | **−18.75%** |

> **Principal finding:** Ensemble ML-IDS are highly effective against passive attackers but critically vulnerable to black-box adversarial evasion. Sophisticated threat actors with black-box API access can craft evasion payloads that reduce detection accuracy from 80% to 20%. **Production IDS deployment must incorporate adversarial defenses.**

---

## Architecture

```
Raw Network Traffic
        │
        ▼
┌───────────────────┐
│   Apache Kafka    │  ← Decoupled packet capture & ingestion
│  Streaming Layer  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  CICFlowMeter     │  ← 41 features per flow (TCP flags, byte
│  Feature Extract  │    volumes, connection duration, etc.)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  MIG Selection    │  ← Mutual Information Gain → top-30 features
│  + SMOTE Balance  │    SMOTE for U2R class (0.04% of training data)
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────┐
│          Ensemble Layer 1          │
│  ┌───────────────┐  ┌────────────┐ │
│  │ Random Forest │  │  XGBoost   │ │  ← 5-fold CV meta-features
│  │  500 trees    │  │ 300 est.   │ │    (prevents data leakage)
│  └──────┬────────┘  └─────┬──────┘ │
└─────────┼─────────────────┼────────┘
          │                 │
          ▼                 ▼
┌────────────────────────────────────┐
│         Meta-Layer                 │
│   Logistic Regression              │  ← P(attack|x) = sigmoid(w1*p_RF + w2*p_XGB + b)
└────────────────┬───────────────────┘
                 │
                 ▼
         Alert if P(attack|x) >= 0.85
                 │
                 ▼
┌───────────────────┐
│  SHAP             │  ← Post-hoc interpretability
│  Interpretability │    Top feature: src_bytes (SHAP = 4.345)
└───────────────────┘
```

---

## Attack Categories Detected

| Category | Description | NSL-KDD Training Samples |
|---|---|---|
| **DoS** | Denial of Service / DDoS floods | 45,927 |
| **Probe** | Reconnaissance / port scanning | 11,656 |
| **R2L** | Remote to Local exploitation | 995 |
| **U2R** | User to Root privilege escalation | **52** |
| **Normal** | Legitimate network traffic | 67,343 |

U2R (52 samples, 0.04% of training data) is the hardest class — SMOTE oversampling is essential for the model to learn any meaningful U2R detection.

---

## SHAP Interpretability

Top features by mean absolute SHAP value (XGBoost):

| Rank | Feature | SHAP Value | Interpretation |
|---|---|---|---|
| 1 | `src_bytes` | 4.345 | Source data volume — dominant DoS indicator |
| 2 | `dst_host_srv_count` | 1.380 | Service connection count — probe signature |
| 3 | `count` | 1.174 | Connection frequency |
| 4 | `service` | 0.891 | Targeted service type |
| 5 | `dst_host_same_srv_rate` | 0.743 | Same-service connection rate |

---

## Installation

```bash
git clone https://github.com/brigghouse/RT-MLIDS.git
cd RT-MLIDS
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, Apache Kafka 3.4+

---

## Quick Start

```bash
python src/evaluate.py --dataset nsl-kdd --data-path data/NSL_KDD/ --save-model
```

---

## Project Structure

```
RT-MLIDS/
├── src/
│   ├── pipeline/
│   │   ├── stream_processor.py
│   │   └── alert_engine.py
│   ├── models/
│   │   └── ensemble.py
│   ├── preprocessing/
│   │   ├── feature_selection.py
│   │   └── smote_balancer.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_adversarial_eval.ipynb
├── data/README.md
├── docs/RT_MLIDS_Final_ResearchGate.pdf
├── tests/test_ensemble.py
├── .github/workflows/ci.yml
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Datasets

- **NSL-KDD** — [Download](https://www.unb.ca/cic/datasets/nsl.html) · Canadian Institute for Cybersecurity
- **CIC-IDS-2018** — [Download](https://www.unb.ca/cic/datasets/ids-2018.html)

---

## Research Paper

Full paper available in [`docs/RT_MLIDS_Final_ResearchGate.pdf`](docs/RT_MLIDS_Final_ResearchGate.pdf).

```bibtex
@article{brighouse2026rtmlids,
  title   = {RT-MLIDS: A Real-Time Ensemble Machine Learning Framework for Network Intrusion Detection with Adversarial Robustness Evaluation},
  author  = {Brighouse Quintana, Ian Alexander},
  school  = {University of East London, Department of Cybersecurity and Networks},
  year    = {2026},
  url     = {https://www.researchgate.net}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Ian Alexander Brighouse Quintana · University of East London<br>
Department of Cybersecurity and Networks
</div>
