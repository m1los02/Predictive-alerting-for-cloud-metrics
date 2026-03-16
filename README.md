# Cloud Incident Predictor

Predictive alerting for cloud infrastructure metrics using sliding-window binary classification.

---

## Overview

This project implements a machine-learning pipeline that predicts whether a server incident will occur within the next **H** time steps, given the most recent **W** steps of multivariate server metrics.  The goal is to provide actionable early warnings to on-call engineers before an incident fully materialises.

The pipeline covers the complete ML lifecycle: data loading, exploratory analysis, feature engineering, model training (baselines → Logistic Regression → Random Forest → XGBoost → BiLSTM), threshold selection, and evaluation.

## Repository Structure

```
cloud-incident-predictor/
├── src/
│   ├── data/
│   │   ├── loader.py                ← SMD loader + MachineData class
│   │   └── windows.py               ← sliding-window construction + labeling
│   ├── features/
│   │   └── engineer.py              ← feature extraction + FeaturePipeline
│   ├── models/
│   │   ├── baseline.py              ← MajorityClass + Persistence baselines
│   │   ├── classical.py             ← LR, RF, XGBoost + Optuna tuning
│   │   └── lstm.py                  ← PyTorch BiLSTM with attention pooling
│   └── evaluation/
│       ├── metrics.py               ← AUROC, AUPRC, F1, lead-time
│       └── threshold.py             ← threshold sweep + optimisation
├── configs/
│   └── config.yaml                  ← all hyperparameters and paths
├── train.py                         ← end-to-end training pipeline
├── evaluate.py                      ← plots + markdown report generation
├── requirements.txt
└── README.md                     
```

---

## Problem Formulation

### Sliding-Window Classification

Given a multivariate time series **S** ∈ ℝ^(T×F) (T timesteps, F metrics) and a binary incident label vector **L** ∈ {0,1}^T, we construct a supervised dataset:

```
For each valid end-index t ∈ [W−1, T−H):
  X[i]  =  S[t−W+1 : t+1]       ∈ ℝ^(W×F)   (look-back window)
  y[i]  =  max(L[t+1 : t+H+1])  ∈ {0, 1}     (predictive label)
```

- **W** = 60 steps (60 minutes of history)
- **H** = 10 steps (predict incidents within the next 10 minutes)
- **y = 1** means an incident will occur within the next H steps

This formulation converts the time-series problem into standard binary classification, enabling the use of any supervised learning algorithm.

### Why This Framing Works

The H-step horizon is the key design choice.  Setting H too small (H=1) makes labels nearly impossible to predict; too large (H=120) causes many false positives since many incidents do not materialise.  H=10 balances predictability with actionability.

---

## Dataset: Server Machine Dataset (SMD)

**Source:** [OmniAnomaly (KDD 2019)](https://github.com/NetManAIOps/OmniAnomaly)

| Property | Value |
|----------|-------|
| Machines | 28 server machines |
| Metrics per machine | 38 (CPU, memory, disk, network I/O, …) |
| Timestep granularity | 1 minute |
| Train length per machine | ~7,200 steps (5 days) |
| Test length per machine  | ~7,200 steps (5 days) |
| Positive-class rate (test) | ~4–12% depending on machine |

**Labeling convention in SMD:** The `train/` split is considered anomaly-free (all labels = 0), while `test/` has hand-annotated incident windows.  We concatenate both splits per machine and use zero labels for the train portion, which maximises training data while respecting the labeling convention.

**Machine-level split** (not window-level): we assign whole machines to train / val / test sets.  This completely prevents temporal data leakage — no future timestep from any machine ever appears in the training set.

---

## Feature Engineering

Raw windows of shape (W, F) are fed to the LSTM directly.  For tree-based models and logistic regression, we extract summary statistics into a flat feature vector:

| Group | Description | Size |
|-------|-------------|------|
| Rolling statistics | mean, std, min, max, median over last 15, 30, 60 steps | 5 × 3 × 38 = 570 |
| First-difference stats | mean Δ, std Δ, mean \|Δ\| across whole window | 3 × 38 = 114 |
| Linear trend slope | signed slope of linear regression across window | 38 |
| Zero-crossing rate | sign-change frequency of Δx | 38 |
| Peak fraction | fraction of steps above the 90th-percentile value | 38 |
| **Total** | | **798** |

Multiple sub-window sizes (15, 30, 60 steps) are used deliberately to give tree models access to both short-term spikes and long-term drift patterns.

All features are z-scored using statistics computed on training data only (via `FeaturePipeline`), preventing leakage.

---

## Models

### Baselines

| Model | Description |
|-------|-------------|
| **Majority Class** | Always predicts class 0. Reveals that accuracy is the wrong metric. |
| **Persistence** | Score = normalised max of last 5 steps. Fires when metrics are already elevated. |

### Classical Models (trained on engineered features)

| Model | Class Imbalance Handling | Notes |
|-------|--------------------------|-------|
| **Logistic Regression** | `class_weight="balanced"` | Strong L2 regularisation (C=0.1) for ~800-dim feature space |
| **Random Forest** | `class_weight="balanced"` | `min_samples_leaf=5` prevents overfitting on rare positives |
| **XGBoost** | `scale_pos_weight = n_neg/n_pos` | Hyperparameters tuned via Optuna (50 trials, maximising val AUPRC) |

### Deep Learning

| Model | Architecture | Notes |
|-------|-------------|-------|
| **BiLSTM** | Input → LayerNorm → BiLSTM(128×2 layers) → Attention Pool → FC(64) → logit | Operates on raw z-scored sequences; no hand-crafted features needed |

The LSTM uses `BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos`, AdamW optimiser, cosine annealing LR schedule, gradient clipping (norm ≤ 1.0), and AMP for GPU speedup.  Early stopping monitors val AUPRC with patience=10.

---

## Evaluation Methodology

### Metrics

| Metric | Why It's Used |
|--------|---------------|
| **AUPRC** | Primary metric. Baseline = positive-class rate. Unaffected by large negative class. |
| **AUROC** | Useful for ranking; can be misleadingly high under severe imbalance. |
| **F1** | Harmonic mean of precision and recall at the chosen threshold. |
| **Precision** | Fraction of alerts that are real incidents (low = alarm fatigue). |
| **Recall** | Fraction of incidents that trigger an alert (low = missed incidents). |
| **Alert Lead Time** | How many steps before incident onset the model first fires. Operational value. |

### Threshold Selection

The alert threshold τ is a hyperparameter, not a model property.  We select τ on the **validation set** to maximise F1, then hold it fixed for test evaluation.  This is critical — tuning τ on the test set would be a form of test-set leakage.

The `--metric` option to `find_best_threshold()` allows choosing between:
- `f1` (β=1): balanced precision/recall
- `f1_beta_2` (β=2): recall-heavy (catch more incidents, accept more false alarms)
- `f1_beta_0.5` (β=0.5): precision-heavy (fewer false alarms, miss more incidents)

### Data Split

| Split | Role | Fraction |
|-------|------|----------|
| Train | Model training | ~70% of machines |
| Val | Threshold selection + early stopping | ~15% of machines |
| Test | Final evaluation (never used during training) | ~15% of machines |

---

## Results

<!-- Run `python train.py && python evaluate.py` to populate this table. -->

| Model | Split | AUROC | AUPRC ★ | F1 | Precision | Recall | Threshold |
|-------|-------|------:|--------:|---:|----------:|-------:|----------:|
| majority_class | val | 0.500 | 0.027 | 0.053 | 0.027 | 1.000 | 0.022 |
| | test | 0.500 | 0.022 | 0.043 | 0.022 | 1.000 | 0.022 |
| persistence | val | 0.748 | 0.061 | 0.134 | 0.074 | 0.746 | 0.989 |
| | test | 0.492 | 0.033 | 0.022 | 0.012 | 0.191 | 0.989 |
| logistic_regression | val | 0.321 | 0.049 | 0.076 | 0.120 | 0.056 | 1.000 |
| | test | 0.682 | 0.036 | 0.070 | 0.047 | 0.137 | 1.000 |
| random_forest | val | 0.755 | 0.108 | 0.172 | 0.130 | 0.255 | 0.443 |
| | test | 0.639 | 0.170 | 0.204 | 0.194 | 0.216 | 0.443 |
| xgboost | val | — | 0.231* | — | — | — | 0.932 |
| | test | 0.693 | 0.187 | 0.142 | 0.538 | 0.082 | 0.932 |
| lstm | val | — | — | — | — | — | — |
| | test | — | — | — | — | — | — |

---

## Limitations

1. **Fixed window size W and horizon H**: In practice, different incident types may require different look-back horizons.  A multi-scale approach (multiple W values) could improve recall.

2. **No cross-metric attention**: The feature engineering treats each metric independently.  Real incidents often involve correlated failures (e.g., high CPU causes high memory).  Graph neural networks or cross-attention could capture these dependencies.

3. **Machine-level split may underestimate generalisation gap**: In production, a model trained on machines 1–20 must generalise to entirely new machines.  A more rigorous evaluation would test on a machine group with no machines seen during training.

4. **Label quality**: SMD labels are binary per-timestep, but real incident severity is continuous.  The labeling boundary can be noisy (e.g., a minor blip that was labelled as an incident).

5. **Concept drift**: Server load patterns change over time (new deployments, traffic seasonality).  A deployed model would require periodic retraining or online adaptation.
