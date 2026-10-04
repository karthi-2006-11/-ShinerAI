# ShinerAI: Independent Full Reproduction Report

**Status:** Certified & Reproducible  
**Date:** October 2026  
**Environment:** Python 3.12 (Virtual Environment `.venv`)  
**Pipeline Execution Time:** 0.86 seconds  

---

## 1. Executive Summary

This report documents the independent, full end-to-end reproduction of the **ShinerAI** research pipeline. All data invariants, temporal splits, model artifacts, and evaluation metrics were re-derived and programmatically validated against authoritative project records without error.

**Overall Certification Status:** **PASS (100% Deterministic Match)**

---

## 2. Dataset Preconditions & Row Invariants

The raw continuous water quality dataset (`data/raw/csv/`) comprises 17 commercial ponds. Data cleaning and time-series segmentation produce the processed dataset `data/processed/ml_ready_dataset.csv`.

| Invariant / Check | Expected Specification | Reproduced Value | Verification Status |
| :--- | :---: | :---: | :---: |
| **Total Rows** | 41,277 | 41,277 | **PASS** |
| **Total Features / Columns** | 37 | 37 | **PASS** |
| **Monitored Ponds** | 17 | 17 | **PASS** |
| **Healthy Baseline DO Invariant** | $\text{DO} \ge 3.00\text{ mg/L}$ | $\min(\text{DO}) = 3.00$ | **PASS** |
| **SAFE Class Count (`target = 0`)** | 36,101 (87.46%) | 36,101 | **PASS** |
| **AT_RISK Class Count (`target = 1`)** | 5,176 (12.54%) | 5,176 | **PASS** |
| **Missing Values in Predictors** | 0 NaN | 0 NaN | **PASS** |

---

## 3. Temporal Split & Purge Gap Verification

To eliminate temporal leakage, an 80/20 chronological partition per pond is enforced with a mandatory **2.0-hour purge gap**:

| Partition | Row Count | Target = 1 | Target = 0 | Positive Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Training Split** | 32,908 | 4,233 | 28,675 | 12.86% |
| **Purged Boundary Gap** | 108 | 0 | 108 | 0.00% |
| **Held-Out Test Split** | 8,261 | 943 | 7,318 | 11.42% |
| **Total Rows Conserved** | **41,277** | **5,176** | **36,101** | **100.00%** |

- **Purge Gap Verification:** Every pond exhibits $\Delta t \ge 2.0$ hours between the latest training timestamp and the earliest test timestamp. Minimum observed gap: **2.00 hours**; maximum observed gap: **12.25 hours**. Zero temporal leakage confirmed.

---

## 4. Active Production Model Reproduction

- **Model Artifact:** `models/xgboost_config_c.joblib`
- **Predictor Set:** Config C (11 features: `current_do`, `hour_of_day`, `minute_of_day`, and 8 lags $t-15\text{m}$ to $t-120\text{m}$)
- **Decision Threshold:** $\tau = 0.50$ (uncalibrated)

### Strict Metric Comparison Table

| Metric | Authoritative Benchmark | Independently Reproduced | Absolute Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | **0.7574** | **0.7574** | 0.0000 | **MATCH** |
| **ROC-AUC** | **0.9162** | **0.9162** | 0.0000 | **MATCH** |
| **F1-Score** | **0.6285** | **0.6285** | 0.0000 | **MATCH** |
| **Recall (Sensitivity)** | **0.7975** | **0.7975** | 0.0000 | **MATCH** |
| **Precision (PPV)** | **0.5186** | **0.5186** | 0.0000 | **MATCH** |
| **Specificity (TNR)** | **0.9046** | **0.9046** | 0.0000 | **MATCH** |
| **Accuracy** | **0.8924** | **0.8924** | 0.0000 | **MATCH** |
| **True Positives (TP)** | **752** | **752** | 0 | **MATCH** |
| **False Positives (FP)** | **698** | **698** | 0 | **MATCH** |
| **True Negatives (TN)** | **6,620** | **6,620** | 0 | **MATCH** |
| **False Negatives (FN)** | **191** | **191** | 0 | **MATCH** |

---

## 5. Domain Baselines Verification

| Baseline | PR-AUC | ROC-AUC | F1 | Recall | Precision | Specificity | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.1142 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8858 |
| **Strict Persistence Baseline** | 0.1142 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8858 |
| **Linear Trend Baseline (120 min)** | 0.4656 | 0.8870 | 0.5834 | 0.6267 | 0.5457 | 0.9328 | 0.8978 |
| **Current-DO Heuristic Baseline** | 0.6149 | 0.9024 | 0.4516 | 0.8961 | 0.3019 | 0.7330 | 0.7516 |
| **XGBoost Config C (Champion)** | **0.7574** | **0.9162** | **0.6285** | **0.7975** | **0.5186** | **0.9046** | **0.8924** |

- **Observed Lift:**
  - $+0.6432$ PR-AUC over Majority / Persistence Baselines.
  - $+0.2918$ PR-AUC over Linear Trend Extrapolation.
  - $+0.1425$ PR-AUC over Current-DO Static Threshold.

---

## 6. Operational Event-Level Reproduction

- **Contiguous Hypoxia Episodes:** 136 episodes identified across test holdout.
- **Event-Level Detection Rate:** **91.18%** (124 of 136 episodes detected).
- **Advance Warning Lead Time:**
  - Mean: **101.7 minutes**
  - Median: **120.0 minutes**
  - Min: 15.0 minutes
  - Max: 345.0 minutes
- **Operator Alarm Burden:**
  - Daily Alert Rate: **4.84 alerts/pond/day**
  - False Episode Rate: **1.62 episodes/pond/day**
  - Chattering Percentage: **32.5%**
- **Hysteresis Mitigation ($k=2$):**
  - False Positive Intervals drop from 698 to 449 (**-27.2%**).
  - Specificity increases from 90.5% to 93.9%.
  - Recall maintained at 73.4%.

---

## 7. Calibration & Cost-Sensitive Reproduction

- **Brier Score (Uncalibrated):** 0.0863
- **Brier Score (Platt Scaled):** 0.0503 (-41.7%)
- **Brier Score (Isotonic):** 0.0503 (-41.7%)
- **Empirical Cost Optimization (5:1 penalty ratio on train split):**
  - Minimum cost threshold: $\tau^* = 0.50$
  - Operating point matches default $\tau = 0.50$ (Global empirical cost minimum).

---

## 8. External Validation Reproduction Summary

1. **Oman Nile Tilapia Dataset (*Sensors* 2026):**
   - Clean QC Telemetry ($DO > 0$): **0 False Positives, 100.0% Specificity** (3,808 of 3,808 intervals correctly classified SAFE).
   - Unfiltered Raw Telemetry: **5 False Positives, 99.87% Specificity** (caused by 2 hardware dropout $0.0\text{ mg/L}$ zero-sensor spikes).
   - Reconciliation certified: Discrepancy fully attributed to upstream hardware dropouts.
2. **Andhra Pradesh Dataset (*WQRJ* 2026):**
   - Audit completed: Temporal cadence mismatch (20-min sampling vs 15-min feature contract).
   - Methodological rejection certified: Zero synthetic interpolation used, preserving scientific integrity.

---

## 9. Conclusion

The ShinerAI pipeline is 100% reproducible. All code paths, serialized models, temporal splits, and metric calculation routines yield exact, deterministic results matching all submitted documentation.
