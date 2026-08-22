# Aircraft-Engine-RUL-Prediction

**Condition-Aware Dual-Stream Transformer for Remaining Useful Life (RUL) Prediction of Aircraft Engines**

A deep learning framework for **Remaining Useful Life (RUL) prediction and degradation-state classification** using the NASA C-MAPSS turbofan engine degradation dataset. The proposed model combines operating-condition awareness with complementary temporal modeling through a **Multi-Scale Temporal Convolutional Network (TCN), Conditioned BiGRU, and Transformer Encoder**.

---

## Project Overview

Predictive Maintenance (PdM) aims to estimate the condition and future failure of industrial assets before unexpected breakdowns occur. This project addresses this problem by predicting the Remaining Useful Life of turbofan engines from sequential sensor measurements.

The proposed **Condition-Aware Dual-Stream Transformer** uses a multi-task learning framework to jointly perform:

* **Binary degradation-state classification**
* **Continuous RUL regression**
* **Predictive uncertainty estimation using Monte Carlo Dropout**

The model is evaluated across all four NASA C-MAPSS benchmark datasets: **FD001, FD002, FD003, and FD004**.

---

## Proposed Architecture

The architecture processes multivariate engine sensor sequences through a condition-aware dual-stream pipeline:

```text
Input Sensor Sequence
        │
        ▼
Variable Selection Network
   (Sigmoid Gating)
        │
        ├──────────────────┐
        ▼                  ▼
 Multi-Scale TCN     Conditioned BiGRU
   Stream A             Stream B
        │                  │
        └────────┬─────────┘
                 ▼
      Cross-Stream Gated Fusion
                 │
                 ▼
       Gated Residual Network
                 │
                 ▼
      Positional Encoding
                 │
                 ▼
       Transformer Encoder
                 │
                 ▼
     Adaptive Attention Pooling
                 │
                 ▼
       Squeeze-and-Excitation
                 │
          ┌──────┴──────┐
          ▼             ▼
   Classification      RUL
       Head            Head
```

### Main Components

* **Variable Selection Network (VSN)**
  Learns independent feature gates using Sigmoid activation to emphasize informative sensor variables.

* **Multi-Scale TCN**
  Uses parallel dilated convolutions with dilation rates of **1, 2, 4, and 8** to capture local temporal degradation patterns.

* **Conditioned BiGRU**
  Models cumulative engine degradation while incorporating learned **operating-regime embeddings**.

* **Cross-Stream Gated Fusion**
  Learns how to combine local temporal information from the TCN with longer-term degradation information from the BiGRU.

* **Transformer Encoder**
  Applies self-attention to the fused temporal representation using sinusoidal positional encoding.

* **Adaptive Attention Pooling**
  Combines attention-based pooling with mean and maximum temporal representations.

* **Squeeze-and-Excitation (SE)**
  Performs channel-wise recalibration of the pooled representation.

* **Dual Prediction Heads**
  Produces both degradation classification and continuous RUL predictions.

---

## Data Preprocessing

The preprocessing pipeline is designed to account for differences in engine operating conditions.

```text
Raw CMAPSS Data
      │
      ▼
RUL Calculation
      │
      ▼
Degradation Classification
      │
      ▼
Operating-Regime Clustering
      │
      ▼
Condition-Aware Normalization
      │
      ▼
EMA Sensor Denoising
      │
      ▼
Sliding-Window Sequences
      │
      ▼
Model Input
```

### Key preprocessing steps

* RUL values are calculated from the final failure cycle.
* RUL is capped at **125 cycles**.
* Engines with **RUL ≤ 50** are classified as degraded.
* Operating conditions are clustered using **K-Means**.
* FD002 and FD004 contain multiple operating regimes that are explicitly handled.
* Sensor values are normalized on a **per-regime basis**.
* Operational settings are normalized globally.
* Exponential Moving Average (EMA) denoising with a span of **5** is applied after normalization.
* Sequential samples are generated using a **40-cycle sliding window**.

---

## Dataset

The project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.

| Dataset | Operating Regimes | Fault Modes | Training Engines | Test Engines |
| ------- | ----------------: | ----------: | ---------------: | -----------: |
| FD001   |                 1 |           1 |              100 |          100 |
| FD002   |                 6 |           1 |              260 |          259 |
| FD003   |                 1 |           2 |              100 |          100 |
| FD004   |                 6 |           2 |              249 |          248 |

Each dataset contains:

* **21 sensor measurements**
* **3 operational settings**
* Engine operating cycles
* Engine/unit identifiers

Dataset source:

[https://www.kaggle.com/datasets/fareselgohary003/nasa-cmapss-turbofan-engine-rul-dataset](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation)

---

## Training Configuration

The current implementation uses the following primary configuration:

| Parameter               |      Value |
| ----------------------- | ---------: |
| Sequence Length         |         40 |
| Batch Size              |         64 |
| Learning Rate           |   5 × 10⁻⁴ |
| Maximum Epochs          |        150 |
| Early Stopping Patience |         45 |
| RUL Cap                 |        125 |
| Operating Regimes       |          6 |
| Random Seeds            | 42, 43, 44 |

The implementation is written in **PyTorch** and supports hardware acceleration through CUDA when available.

---

## Loss Functions

The model uses a joint multi-task objective:

```text
L_total = L_cls + α*L_reg
```

where:

* **L_cls** is the Focal Loss for degradation-state classification.
* **L_reg** is the NASA-inspired asymmetric regression loss for RUL prediction.

The asymmetric regression objective gives greater penalty to prediction errors that can have greater operational consequences, particularly late RUL predictions.

---

## Evaluation

The framework evaluates both classification and RUL regression performance.

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1-Score

### RUL Prediction Metrics

* RMSE
* NASA Score
* NASA Score per Engine

### Uncertainty

**Monte Carlo Dropout** is used during inference to obtain stochastic predictions and estimate predictive uncertainty.

Mean classification entropy is also calculated as an additional uncertainty indicator.

---

## Benchmarking

The complete benchmark evaluates:

```text
FD001
FD002
FD003
FD004
```

across three random seeds:

```text
Seed 42
Seed 43
Seed 44
```

The final benchmark reports results as:

```text
Mean ± Standard Deviation
```

for each dataset.

---

## Project Structure

```text
Aircraft-Engine-RUL-Prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   └── raw/
│       └── .gitkeep
│
├── notebooks/
│   └── Code.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── preprocessing.py
│   ├── losses.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── docs/
    └── architecture.png
```

---

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Research Focus

The project focuses on:

* Condition-aware RUL prediction
* Multivariate time-series modeling
* Operating-regime awareness
* Multi-task degradation prediction
* Multi-scale temporal feature extraction
* Transformer-based temporal modeling
* Predictive uncertainty estimation
* Predictive maintenance analytics

---

## Objective

The overall objective is to develop a **condition-aware and uncertainty-aware deep learning framework** capable of modeling turbofan engine degradation and estimating Remaining Useful Life from sequential sensor data.

The framework is designed to support predictive maintenance by providing both an estimated degradation state and continuous RUL prediction for aircraft engines.
