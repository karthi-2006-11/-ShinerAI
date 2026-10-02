# ShinerAI: Phase 4 Technical Completion Report
## Model Explainability and Flask Backend REST API

**Project:** ShinerAI  
**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Scope:** Phase 4 — Model Explainability (SHAP TreeExplainer) & Flask Backend API Serving  
**Date:** October 2026  
**Status:** Validated, Tested, and Frozen  

---

## 1. Executive Summary

Phase 4 bridges the trained Machine Learning models from Phase 3 into an operational, interpretable software service ready for downstream user interfaces. This phase delivered two core technical capabilities without modifying any raw data, labels, splits, or model weights:

1. **Model Explainability Engine:** Built with SHAP (*SHapley Additive exPlanations*) TreeExplainer, quantifying global feature importance across 1,000 held-out test set instances and generating local feature attributions for individual pond risk scores.
2. **Modular Flask REST API:** A production-grade web service serving the models with configurable selection (`MODEL_ARTIFACT_PATH`), comprehensive input validation, boundary condition enforcement ($\text{DO} \ge 3.0\text{ mg/L}$), and four REST endpoints (`/health`, `/model-info`, `/predict`, `/explain`).

The implementation is verified with **59 automated unit and integration tests** passing with 100% success rate (32 legacy tests from Phases 1–3 + 27 new Phase 4 tests).

---

## 2. Model Explainability Pipeline

### 2.1 Methodology & Artifact Provenance
Explainability was conducted on the two top-performing tree-based models on **Config C (DO History Only, 11 features)**:
- **XGBoost Config C** (Test PR-AUC: 0.7574, Recall: 79.75%, Precision: 51.86%, Specificity: 90.46%)
- **Random Forest Config C** (Test PR-AUC: 0.7471, Recall: 75.61%, Precision: 58.59%, Specificity: 93.11%)

> [!NOTE]
> **Model Artifact Provenance:**  
> Config C model artifacts were reproduced using the frozen Phase 3 training procedure solely to create dedicated explainability/API artifacts; no model architecture, dataset, split, feature set, or training procedure was changed.

Using `shap.TreeExplainer`, feature attributions are computed on the probability scale (Random Forest) and log-odds margin (XGBoost), measuring the exact contribution of each predictor toward or against the `AT_RISK` event.

### 2.2 Global Feature Importance Ranking
Evaluated across 1,000 stratified observations from the held-out temporal test set:

| Rank | Feature | XGBoost Mean \|SHAP\| | Random Forest Mean \|SHAP\| | Operational Role |
|:---:|:---|:---:|:---:|:---|
| 1 | `current_do` | **1.3896** | **0.1049** | Primary baseline level at prediction time $T$. |
| 2 | `minute_of_day` | **0.5388** | 0.0485 | Encodes time-of-day patterns observed in the dataset. |
| 3 | `hour_of_day` | 0.2549 | 0.0340 | Coarse temporal diurnal indicator. |
| 4 | `do_t_minus_30` | 0.2388 | 0.0434 | Short-term historical lag ($T-30\text{ min}$). |
| 5 | `do_t_minus_15` | 0.2385 | **0.0758** | Represents DO 15 min prior; with current DO, contributes trajectory information. |
| 6 | `do_t_minus_120` | 0.1562 | 0.0155 | 2-hour window boundary anchor ($T-120\text{ min}$). |
| 7 | `do_t_minus_45` | 0.1555 | 0.0317 | Intermediate trend lag ($T-45\text{ min}$). |
| 8 | `do_t_minus_90` | 0.1237 | 0.0144 | Historical trajectory lag ($T-90\text{ min}$). |
| 9 | `do_t_minus_105` | 0.1117 | 0.0158 | Historical trajectory lag ($T-105\text{ min}$). |
| 10 | `do_t_minus_60` | 0.1058 | 0.0257 | Midpoint trajectory lag ($T-60\text{ min}$). |
| 11 | `do_t_minus_75` | 0.0986 | 0.0212 | Intermediate trajectory lag ($T-75\text{ min}$). |

#### Key Global Observations:
1. `current_do` is by far the single most influential predictor in both architectures, establishing the proximity of the pond to the hypoxic cutoff ($3.0\text{ mg/L}$).
2. Diurnal timing (`minute_of_day` and `hour_of_day`) represents the second most critical axis, encoding time-of-day patterns observed in the dataset.
3. Recent history (`do_t_minus_15` and `do_t_minus_30`) represents DO levels 15 and 30 minutes before prediction and, together with current DO, contributes information about the recent temporal trajectory.

### 2.3 Representative Case Studies

#### SAFE Case Study — Rising DO During Daytime (Ground Truth: SAFE)
- **Pond:** `ara2_0677080b` | **Timestamp:** `2026-01-26 10:15:00`
- **Current DO:** `5.40 mg/L` | **2-Hour Prior DO:** `2.92 mg/L`
- **Model Output:**
  - XGBoost: Risk Probability = **6.67%** (`SAFE`)
  - Random Forest: Risk Probability = **5.17%** (`SAFE`)
- **Deconstruction:**
  Although DO had been at $2.92\text{ mg/L}$ two hours prior, readings increased monotonically ($3.23 \rightarrow 3.72 \rightarrow 4.21 \rightarrow 5.40\text{ mg/L}$) during mid-morning (`10:15 AM`). SHAP attributions for `minute_of_day` ($-1.1596$), `current_do` ($-0.6279$), and `hour_of_day` ($-0.3339$) pushed heavily toward `SAFE`, preventing a false alarm despite prior low oxygen.

#### AT_RISK Case Study — Declining DO During Nighttime (Ground Truth: AT_RISK)
- **Pond:** `ara2_0677080b` | **Timestamp:** `2026-01-26 03:30:00`
- **Current DO:** `3.84 mg/L` | **2-Hour Prior DO:** `5.02 mg/L`
- **Model Output:**
  - XGBoost: Risk Probability = **93.41%** (`AT_RISK`)
  - Random Forest: Risk Probability = **91.67%** (`AT_RISK`)
- **Deconstruction:**
  Although current DO remained above the critical threshold at $3.84\text{ mg/L}$, the trajectory showed a sustained drop from $5.02\text{ mg/L}$ during nighttime (`03:30 AM`). SHAP attributions for `current_do` ($+2.0758$), `minute_of_day` ($+0.3809$), and `do_t_minus_15` ($+0.1468$) pushed strongly toward `AT_RISK`, successfully triggering the early warning 2 hours before hypoxia.

---

## 3. Scientific Framing & Causal Guardrail

To maintain strict scientific integrity, the project adheres to the following principles:

> [!IMPORTANT]
> 1. **Statistical Association vs. Biological Causality:**  
>    SHAP values quantify empirical mathematical relationships between features and model outputs within this specific dataset. They **do not prove physiological causality**. High importance indicates predictive correlation, not direct biological causation.
> 2. **Discrete Lags vs. Explicit Derivatives:**  
>    Inputs are 8 discrete historical lags ($t-15\text{m} \dots t-120\text{m}$); no continuous derivatives or differential equations were computed.
> 3. **Low-DO Warning vs. Fish Mortality:**  
>    ShinerAI forecasts water oxygen depletion events ($\text{DO} < 3.0\text{ mg/L}$ within the next 2 hours). It **does not predict fish mortality, disease, or biological stress levels directly**.

---

## 4. Flask Backend API Architecture

### 4.1 Modular Layout
The API is implemented in `backend/` using a beginner-friendly, modular structure:
```
backend/
├── __init__.py         # Package entry and app exposure
├── app.py              # Application factory (create_app), error handling, routes
├── config.py           # Configuration (MODEL_ARTIFACT_PATH, feature schemas, thresholds)
├── validation.py       # Strict payload validation, ISO timestamp parsing, boundary checks
├── model_service.py    # Read-only model artifact loading, formatting, inference
└── explainability.py   # On-demand local SHAP computation
```

### 4.2 Endpoint Summary
- `GET /`: Discovery index with active model details.
- `GET /health`: Server health check, active artifact name, threshold, UTC timestamp.
- `GET /model-info`: Comprehensive model metadata, 11 expected features, held-out test metrics, operational trade-off guidance, and scientific scope.
- `POST /predict`: Real-time risk scoring returning probability, predicted label (`SAFE` or `AT_RISK`), binary flag, and operator alert text.
- `POST /explain`: Real-time risk scoring plus local SHAP feature contributions, base value, top risk drivers, and top safe drivers.

### 4.3 Validation & Boundary Enforcement
1. **Operational Cutoff Enforcement:** If `current_do < 3.0 mg/L`, the API rejects the request with HTTP `400 Bad Request` and `status: ALREADY_LOW_DO`.
2. **Canonical Input Naming:** The canonical public input standard is `do_t_minus_15` through `do_t_minus_120`. Alternative aliases (e.g. `do_t-15m`) are supported internally for client compatibility, but `do_t_minus_X` is the canonical standard.
3. **Automatic Temporal Derivation:** `hour_of_day` and `minute_of_day` are automatically extracted from `prediction_timestamp`.
4. **Data Sanity Checks:** Negative sensor values, non-numeric values, missing fields, or empty strings return structured 400 JSON errors.

---

## 5. Model Selection & Operational Trade-offs

The backend supports configurable model selection via the environment variable `MODEL_ARTIFACT_PATH`. Both Config C models deliver competitive, practical performance:

| Operating Priority / Metric | XGBoost Config C | Random Forest Config C | Operational Guidance |
|:---|:---:|:---:|:---|
| **Highest PR-AUC among tested models** | **0.7574** | 0.7471 | XGBoost Config C achieves the highest PR-AUC across all evaluated models. |
| **Recall (Sensitivity)** | **79.75%** (752 / 943) | 75.61% (713 / 943) | Highest recall among Config C tree models. |
| **Specificity** | 90.46% | **93.11%** | Random Forest reduces false alarms. |
| **False Positives (FP)** | 698 | **504** | Random Forest produces 194 fewer false positives than XGBoost Config C at the default threshold; this could reduce unnecessary interventions in a deployment where alerts trigger aeration. |
| **Default Selection** | **Default (`MODEL_ARTIFACT_PATH`)** | Alternative | Model choice depends on farm operating priorities. |

*Framing note: Neither model is universally superior. XGBoost Config C achieves the highest PR-AUC among tested models (0.7574) and highest recall among Config C tree models (79.75%), making it preferable when prioritizing detection of low-DO events within the DO-only feature space. Random Forest Config C achieves higher Specificity (93.11%), producing 194 fewer false positives than XGBoost Config C at the default threshold; this could reduce unnecessary interventions in a deployment where alerts trigger aeration. (Note: In Phase 3, Logistic Regression Config B demonstrated 89.93% recall across all 29 features, but at the cost of 2,158 false alarms; among tree-based models on Config C, XGBoost achieves the highest recall).*

---

## 6. Verification and Test Results

### 6.1 Artifact Reproducibility Verification
All reported Phase 3 evaluation metrics and thresholded confusion-matrix counts were reproduced exactly by the Phase 4 Config C artifacts on the same temporal holdout set.

| Metric / Count | Phase 3 Reported ([`model_comparison.csv`](file:///d:/FISH/results/reports/model_comparison.csv)) | Phase 4 XGBoost Config C | Phase 4 Random Forest Config C | Match Status |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC (Average Precision)** | 0.7574 (XGB) / 0.7471 (RF) | 0.7574 | 0.7471 | Exact match |
| **ROC-AUC** | 0.9162 (XGB) / 0.9144 (RF) | 0.9162 | 0.9144 | Exact match |
| **Precision** | 51.86% (XGB) / 58.59% (RF) | 51.86% | 58.59% | Exact match |
| **Recall (Sensitivity)** | 79.75% (XGB) / 75.61% (RF) | 79.75% | 75.61% | Exact match |
| **F1-Score** | 0.6285 (XGB) / 0.6602 (RF) | 0.6285 | 0.6602 | Exact match |
| **Specificity** | 90.46% (XGB) / 93.11% (RF) | 90.46% | 93.11% | Exact match |
| **Accuracy** | 89.24% (XGB) / 91.11% (RF) | 89.24% | 91.11% | Exact match |
| **True Positives (TP)** | 752 (XGB) / 713 (RF) | 752 | 713 | Exact match |
| **False Positives (FP)** | 698 (XGB) / 504 (RF) | 698 | 504 | Exact match |
| **True Negatives (TN)** | 6,620 (XGB) / 6,814 (RF) | 6,620 | 6,814 | Exact match |
| **False Negatives (FN)** | 191 (XGB) / 230 (RF) | 191 | 230 | Exact match |

> **Reproducibility Note:**  
> No difference was observed in the verified evaluation metrics or thresholded classification counts. Note that this verification validates the reported metrics and confusion-matrix counts on the temporal test set rather than full per-row prediction arrays.

### 6.2 Automated Test Suite Execution
The entire automated test suite was executed in the project virtual environment:
```
pytest -v
```
**Results:**
- Total Tests: **59 passed** (0 failures, 0 errors).
- Coverage:
  - 32 legacy tests for Phase 1 (data pipeline, QC parsing), Phase 2 (cleaning, leak-free targets), and Phase 3 (model evaluation, temporal purge gap).
  - 27 new tests for Phase 4 covering all 5 REST endpoints, model configurability, boundary conditions, error handling, SHAP explanation structures, and artifact presence.

---

## 7. Artifact Manifest (Phase 4)

### 7.1 Visualizations (`results/figures/explainability/`)
- `global_feature_importance_xgb_config_c.png`
- `global_feature_importance_rf_config_c.png`
- `local_explanation_safe_xgb_config_c.png`
- `local_explanation_at_risk_xgb_config_c.png`
- `local_explanation_safe_rf_config_c.png`
- `local_explanation_at_risk_rf_config_c.png`

### 7.2 Reports & Summaries (`results/reports/explainability/`)
- `global_importance_xgb_config_c.csv`
- `global_importance_rf_config_c.csv`
- `local_explanations_summary.json`
- `explainability_report.md`

### 7.3 Model Artifacts (`models/`)
- `xgboost_config_c.joblib` (Default serving artifact)
- `random_forest_config_c.joblib` (Alternative serving artifact)
- `model_metadata.json` (Updated with Config C parameters and metrics)
