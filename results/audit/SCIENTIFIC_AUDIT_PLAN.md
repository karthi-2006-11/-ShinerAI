# ShinerAI: Scientific Audit & Evidence Verification Plan

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Aquaculture  
**Audit Date:** October 3, 2026  
**Auditor:** Scientific Audit Agent (Independent AI & Machine Learning Research Verification)  
**Corpus / Repository:** `karthi-2006-11/-ShinerAI` (`d:\FISH`)  
**Active Production Model:** `models/xgboost_config_c.joblib`  
**Operational Target:** Binary prediction of low DO event ($\text{DO} < 3.0\text{ mg/L}$) within the next 2-hour window, given current $\text{DO} \ge 3.0\text{ mg/L}$ and 2-hour observation history.

---

## 1. Executive Summary & Audit Mandate

This document establishes the formal protocol for the comprehensive scientific audit of the ShinerAI project. The audit was commissioned to address mentor and peer-review feedback, resolve reported metric discrepancies, verify zero-leakage temporal validation protocols, confirm mathematical ground-truth label definitions, and eliminate overclaims across documentation and research code.

### Core Audit Principles
1. **Evidence Integrity:** No fabricated numbers, no cherry-picking, and no silent re-training to make results look better.
2. **Reproducibility Guarantee:** Every claim, metric, and figure must be mathematically reproducible from the raw sensor data through verified deterministic pipelines.
3. **Scientific Humility:** Distinguish statistical associations from biological causality; distinguish development workstation benchmarks from embedded edge benchmarks; describe model configurations as empirical ablation studies rather than novel deep architectures.
4. **Operational Clarity:** State unequivocally that the $3.0\text{ mg/L}$ threshold is an operational project threshold chosen for early risk mitigation, not an all-species universal biological constant.

---

## 2. Inventory of Inspected Project Assets

| Asset Category | File Path | Status / Checksum Verification |
| :--- | :--- | :--- |
| **Raw Datasets** | `data/raw/` (17 pond CSVs) | Verified 17 ponds, raw sensor telemetry |
| **Processed Dataset** | `data/processed/ml_ready_dataset.csv` | 41,277 rows, 37 columns, 0 NaNs, target $\in \{0, 1\}$ |
| **Cleaning Pipeline** | `src/cleaning.py` | 512 colliding records (258 excess rows dropped), 629 sensor zeros |
| **Feature Pipeline** | `src/model_utils.py` | Config A (5), Config B (29), Config C (11), 8 excluded columns |
| **Training Pipeline** | `src/train.py` | Per-pond 80/20 chronological split with 2.0-hour purge gap |
| **Evaluation Pipeline**| `src/evaluate.py` | Holdout evaluation, GroupKFold cross-validation, confusion matrices |
| **Explainability** | `src/explainability_pipeline.py` | TreeExplainer, SHAP values, feature importance, local force plots |
| **Active Deployed Model**| `models/xgboost_config_c.joblib` | MD5 / Joblib verified, 11 features, default inference model |
| **Alternative Model** | `models/random_forest_config_c.joblib` | 11 features, high-specificity configuration |
| **Legacy Models** | `models/xgboost.joblib`, `models/random_forest.joblib` | Config B (29 features) models preserved for traceability |
| **Master Evaluation** | `results/reports/MASTER_MODEL_EVALUATION.csv` | Master benchmark table across 11 configurations |
| **Research Notebook** | `notebooks/ShinerAI_Complete_ML_Pipeline.ipynb` | 64 cells, end-to-end self-contained executable notebook |
| **Test Suite** | `tests/` (7 test files) | 78 tests passing prior to audit |

---

## 3. Scope of Audit Investigations

### 3.1 Investigation 1: Metric Discrepancy Reconciliation
- **Objective:** Deep-dive into differences between earlier Phase 3 draft reports (Table A) and the master evaluation table / notebook (Table B).
- **Hypotheses Tested:**
  1. *Split Variation:* Were models evaluated on different train/test splits (e.g. 70/30 vs 80/20, or global vs per-pond)?
  2. *Purge Gap:* Did earlier drafts omit the 2-hour purge gap between train and test sets?
  3. *Threshold Variation:* Were different decision thresholds applied (e.g., threshold tuning vs default 0.50)?
  4. *Feature Set Discrepancies:* Were snapshot features in Config A defined with or without cyclical time encoding?
  5. *Scaler Application:* Was standard scaling applied consistently across Logistic Regression runs?
- **Deliverable:** `results/audit/METRIC_RECONCILIATION.csv`

### 3.2 Investigation 2: Active Model Reproduction
- **Objective:** Load the serialized active model `models/xgboost_config_c.joblib`, re-evaluate on the exact held-out test partition (8,261 rows), and compare all metrics to documented values.
- **Deliverable:** `results/audit/ACTIVE_MODEL_REPRODUCTION.md`

### 3.3 Investigation 3: Temporal Split & Purge Gap Audit
- **Objective:** Mathematically verify that no target window from the training partition extends into the test partition.
- **Audit Steps:**
  1. Check per-pond chronological sorting.
  2. Compute $t_{\text{cutoff}}$ at the 80th percentile for each pond.
  3. Verify that all training rows satisfy $\text{timestamp} < t_{\text{cutoff}} - 2.0\text{ hours}$.
  4. Count purged rows per pond (total 108 rows).
  5. Confirm test partition contains strictly $\text{timestamp} \ge t_{\text{cutoff}}$.
- **Deliverable:** `results/audit/SPLIT_AUDIT.md`

### 3.4 Investigation 4: Feature Leakage & Column Exclusion Audit
- **Objective:** Audit all 37 columns in `ml_ready_dataset.csv`.
- **Audit Steps:**
  1. Confirm Config A contains exactly 5 features.
  2. Confirm Config B contains exactly 29 features.
  3. Confirm Config C contains exactly 11 features.
  4. Verify that future DO readings (`do_t_plus_*`), redundant columns (`do_t`, `ph_t`, `temp_t`), metadata (`pond_id`, `prediction_timestamp`), and target columns (`target`, `target_name`, `data_quality_status`) are strictly excluded from model feature inputs.
- **Deliverable:** `results/audit/FEATURE_LEAKAGE_AUDIT.md`

### 3.5 Investigation 5: Ground-Truth Label Integrity Audit
- **Objective:** Independently recalculate the binary target label from the future 2-hour time series for all 41,277 rows and check for 0 discrepancies.
- **Mathematical Definition:**
  $$\text{target} = \mathbb{I}\left(\min_{m \in \{15, 30, \dots, 120\}} \text{do}_{t+m} < 3.0\text{ mg/L}\right)$$
  Conditioned on:
  $$\text{current\_do} = \text{do}_t \ge 3.0\text{ mg/L}$$
- **Deliverable:** `results/audit/LABEL_AUDIT.md`

### 3.6 Investigation 6: Data Cleaning Accounting Audit
- **Objective:** Reconcile exact accounting of raw records to ML-ready rows:
  - Total raw continuous readings: 72,750 across 17 ponds.
  - Colliding timestamps: 512 records (258 excess duplicate rows dropped).
  - Unphysical sensor zeros: 629 zero readings cleaned.
  - Already-low DO exclusions ($\text{DO}_t < 3.0\text{ mg/L}$): 15,234 rows.
  - Boundary exclusions (insufficient 2h past or 2h future): 8,700 short past, 6,398 short future.
  - Final ML-ready dataset: exactly 41,277 rows.
- **Deliverable:** `results/audit/DATA_CLEANING_AUDIT.md`

### 3.7 Investigation 7: Domain Contribution Verification
- **Objective:** Quantify and rigorously contextualize the $+0.1425$ PR-AUC lift of DO temporal history over static thresholding.
- **Deliverable:** `results/audit/DOMAIN_CONTRIBUTION_AUDIT.md`

### 3.8 Investigation 8: Explainable AI & SHAP Soundness Audit
- **Objective:** Audit TreeExplainer usage, verify mathematical convergence ($\sum \phi_i = f(x) - \mathbb{E}[f(x)]$), and ensure no claims of biological causality are made.
- **Deliverable:** `results/audit/SHAP_AUDIT.md`

### 3.9 Investigation 9: Error Analysis Audit
- **Objective:** Conduct exhaustive false positive (698) and false negative (191) profile analysis.
- **Deliverable:** `results/audit/ERROR_ANALYSIS_AUDIT.md`

### 3.10 Investigation 10: Hardware & Wall-Time Audit
- **Objective:** Measure real pipeline wall-time and inference latency on development hardware. Remove unverified edge hardware claims (ESP32 / Raspberry Pi).
- **Deliverable:** `results/audit/WALL_TIME_AUDIT.md`

### 3.11 Investigation 11: Master Notebook Reproducibility Audit
- **Objective:** Execute and verify that `notebooks/ShinerAI_Complete_ML_Pipeline.ipynb` runs deterministically and reproduces all figures and reports.
- **Deliverable:** `results/audit/NOTEBOOK_REPRODUCIBILITY.md`

### 3.12 Investigation 12: Evidence Traceability Mapping
- **Objective:** Map every table, figure, and metric in published documentation back to exact code lines and CSV rows.
- **Deliverable:** `results/audit/RESULT_TRACEABILITY.csv`

---

## 4. Verification Matrix & Quality Standards

| Audit Standard | Acceptance Criteria | Verification Method |
| :--- | :--- | :--- |
| **Row Count Conservation** | Train (32,908) + Purged (108) + Test (8,261) == 41,277 | Programmatic assertion in `tests/test_audit_verification.py` |
| **Zero Target Leakage** | $\max(t_{\text{train}}) < \min(t_{\text{test}}) - 2.0\text{ hr}$ for all ponds | Mathematical interval overlap check |
| **Label Accuracy** | Recalculated labels match existing labels 100% (0 errors out of 41,277) | Element-wise boolean array equality |
| **Active Model PR-AUC** | $0.7574 \pm 0.0001$ on holdout test set | Joblib model evaluation on `X_test[CONFIG_C_FEATURES]` |
| **Active Model Recall** | $0.7975 \pm 0.0001$ (752 TP out of 943 positive events) | Confusion matrix calculation |
| **Deterministic Seed** | All pipelines fix `random_state=42` | Code inspection & deterministic execution |
| **Documentation Honesty** | No claims of "novel architecture", "ESP32 certified", or "biological proof" | Textual scan and remediation across repo |

---

## 5. Execution Timeline & Sign-off

- **Phase 1: Verification & Data Extraction:** Execute reproduction scripts, compute exact confusion matrices, verify label recomputation.
- **Phase 2: Deliverable Compilation:** Generate all 16 audit markdown reports and CSV tables in `results/audit/`.
- **Phase 3: Automated Test Hardening:** Add `tests/test_audit_verification.py` to make audit standards permanent.
- **Phase 4: Documentation Remediation:** Update `PROJECT_DOCUMENTATION.md` and `README.md` to remove exaggerations.
- **Phase 5: Final Review & Git Check-in:** Run full test suite (`pytest`), commit audit deliverables, push to remote repository.
