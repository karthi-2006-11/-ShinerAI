# ShinerAI Research Gap Audit

**Date:** October 4, 2026  
**Auditor:** ShinerAI Research & Scientific Review Team  
**Review Perspective:** IEEE Transactions / Peer-Reviewed Applied Machine Learning & Environmental Sensing Standards  
**Frozen Core Scope:** Model `models/xgboost_config_c.joblib` (XGBoost Config C, 11 features, threshold $\tau = 0.50$, internal PR-AUC = 0.7574)

---

## Executive Context & Scope of Audit

The objective of ShinerAI is to provide an early warning of dissolved oxygen (DO) depletion in freshwater aquaculture ponds:
> *"Given current $\text{DO} \ge 3.0\text{ mg/L}$ and recent 2-hour DO history, predict whether DO will fall below $3.0\text{ mg/L}$ within the next 2 hours ($0 = \text{SAFE}, 1 = \text{AT\_RISK}$)."*

This audit benchmarks the current repository state against the 7 core requirements outlined by the mentor review from an IEEE research perspective. It identifies verified empirical strengths, documents unresolved scientific gaps, and defines prioritized, evidence-backed improvements.

---

## Mentor Requirement 1 — Novelty

- **Status:** PARTIAL / EMPIRICAL-ONLY CONTRIBUTION
- **Evidence in Repository:**
  - `results/reports/MODEL_COMPARISON.csv` & `MODEL_COMPARISON.md`
  - `results/reports/FEATURE_ABLATION_COMPARISON.csv` & `FEATURE_ABLATION_COMPARISON.md`
  - `results/audit/DOMAIN_CONTRIBUTION_AUDIT.md`
  - `models/xgboost_config_c.joblib` (standard XGBoost architecture)
- **What is already satisfied:**
  1. Systematic feature ablation confirming that recent 2-hour DO history adds predictive lift across all evaluated model families: Logistic Regression ($+0.0531$ PR-AUC), Random Forest ($+0.0364$ PR-AUC), and XGBoost ($+0.0257$ PR-AUC) over current-only features.
  2. Demonstration of sensor parsimony in tree ensembles: isolating DO history (Config C, 11 features) achieves superior PR-AUC ($0.7574$) compared to full-sensor history including pH and temperature (Config B, 29 features, $0.7353$ PR-AUC), preventing split fragmentation.
  3. Methodological integrity guardrails: repository explicitly disclaims novel model architectures, claims of engineering physical derivatives, and causal inferences from SHAP attributions.
- **What is missing:**
  1. The core model is off-the-shelf XGBoost. An IEEE reviewer will note that demonstrating lag features improve predictive performance over static snapshots is standard practice rather than a distinctive methodological novelty.
  2. No uncertainty-aware prediction (no prediction intervals or confidence bounds).
  3. No event-level early warning formulation (predictions remain row-level 15-minute point estimates).
  4. No adaptive thresholding mechanism reflecting diurnal or seasonal dynamics.
- **Recommended action:**
  Investigate the three candidate contributions in `NOVELTY_DECISION.md`:
  - Candidate A: Event-level early warning formulation and evaluation.
  - Candidate B: Adaptive alert thresholding.
  - Candidate C: Uncertainty-aware early warning.
  Select at most ONE contribution that is mathematically testable, supported by existing data, free of future leakage, and explainable to an IEEE reviewer without inventing novel architectural claims.

---

## Mentor Requirement 2 — External Validation

- **Status:** PARTIAL / VALIDATED SPECIFICITY ONLY
- **Evidence in Repository:**
  - `data/external/oman_tilapia/` (Nile Tilapia dataset, North Al Sharqiyah, Oman; *Sensors* 2026, 26(13), 4242)
  - `data/external/oman_tilapia_15min_eval.csv`
  - `results/external_validation/EXTERNAL_DATASET_AUDIT.md`
  - `results/external_validation/EXTERNAL_VALIDATION_REPORT.md`
  - `results/external_validation/INTERNAL_VS_EXTERNAL_COMPARISON.csv`
- **What is already satisfied:**
  1. 18-point scientific audit of the public Oman tilapia dataset.
  2. Leakage-free 15-minute right-closed resampling protocol ($(T-15\text{m}, T]$).
  3. Evaluation of frozen model `models/xgboost_config_c.joblib` without retraining, parameter adaptation, or threshold tuning.
  4. Demonstration of high operational specificity: zero false alarms ($\text{TN} = 742, \text{FP} = 0$, Specificity $= 100.0\%$, Accuracy $= 100.0\%$) and conservative probability output (mean risk $= 8.93\%$).
  5. Transparent reporting that positive-class metrics (Recall, Precision, F1, PR-AUC, ROC-AUC) are mathematically undefined due to zero true positive events in the aerated tank.
  6. 10 documented scientific limitations regarding geographic, facility scale, and species shift.
- **What is missing:**
  1. **Discrepancy Investigation:** In exploratory testing, 5 false alarms were reported ($\text{TN} = 737, \text{FP} = 5$), whereas formal validation reported zero false alarms ($\text{TN} = 742, \text{FP} = 0$). Repository evidence shows that this discrepancy was caused by 2 raw analog hardware disconnect zero-readings ($0.0\text{ mg/L}$) on April 9, 2026. In the exploratory run, these zeros were averaged directly into 15-minute lags without QC filtering, causing sudden drop artifacts that triggered 5 false alarms. When standard QC filtering was applied (`df.loc[df['do_mgL'] <= 0] = np.nan`), matching Phase 2 QC rules, the dropouts were properly handled, yielding $\text{FP} = 0$. This technical reconciliation must be formalized and archived.
  2. **External Sensitivity Evidence:** Because the Oman tank was continuously aerated, DO stayed between $6.08\text{ and }12.25\text{ mg/L}$. External sensitivity / recall has not yet been tested on an independent dataset containing real nocturnal hypoxia crashes.
  3. **Second Dataset Investigation:** The Andhra Pradesh aquaculture dataset ("Aquaculture Water Quality", 2026 study) has not been audited or evaluated.
- **Recommended action:**
  1. Formalize the Oman discrepancy reconciliation in `EXTERNAL_VALIDATION_REPORT.md`.
  2. Audit the Andhra Pradesh aquaculture dataset against strict quality criteria (sampling cadence, timestamp regularity, real hypoxia episodes, task compatibility).
  3. If suitable, evaluate the frozen model on it; if unsuitable, document reasons objectively without forcing flawed data into the pipeline.

---

## Mentor Requirement 3 — Operational Evaluation

- **Status:** MISSING ON TEST SET PREDICTIONS
- **Evidence in Repository:**
  - `results/reports/early_warning_event_analysis.csv` (contains ground-truth row-level lead time in Phase 2 data, but not model prediction operational metrics)
  - Standard classification metrics in `results/reports/MASTER_MODEL_EVALUATION.csv`
- **What is already satisfied:**
  1. Standard classification metrics on temporal holdout: Precision ($51.86\%$), Recall ($79.75\%$), Specificity ($90.46\%$), Accuracy ($89.24\%$).
  2. Row-level lead time analysis on historical transitions.
- **What is missing:**
  1. **Warning Lead Time Distribution:** Quantifying how many minutes in advance of the actual $3.0\text{ mg/L}$ crossing the active model generated its first alert.
  2. **Missed Hypoxia Events:** Identifying the specific hypoxia events that failed to receive any advance warning.
  3. **False Alarm Rate per Pond per Day:** Translating raw false positives ($698$ rows across 17 test ponds) into an operational metric (alerts/pond/day) reflecting real farm labor burden.
  4. **Alert Stability Analysis:** Measuring prediction oscillation (rapid toggling between SAFE and AT_RISK within consecutive 15-minute intervals) and testing hysteresis/stability rules without altering the core model.
- **Recommended action:**
  Develop `src/operational_evaluation.py` to evaluate warning lead time, missed events, false alarm frequency per pond-day, and alert chattering on the 8,261 test set observations. Produce `OPERATIONAL_EVALUATION.md`.

---

## Mentor Requirement 4 — Probability Calibration

- **Status:** MISSING
- **Evidence in Repository:**
  - `models/model_metadata.json` stores decision threshold $\tau = 0.50$.
  - Uncalibrated `predict_proba()[:, 1]` outputs used in REST API and metrics.
- **What is already satisfied:**
  1. Raw posterior probabilities are generated by XGBoost and logged.
  2. Binary classification evaluated across all thresholds via PR-AUC and ROC-AUC.
- **What is missing:**
  1. **Reliability Diagram & Calibration Curve:** No visual or tabular analysis showing whether a predicted probability of $70\%$ corresponds to a true $70\%$ empirical frequency of hypoxia.
  2. **Quantitative Calibration Metrics:** Brier score and Expected Calibration Error (ECE) are not reported.
  3. **Post-Processing Calibration:** Platt scaling (logistic calibration) and isotonic regression have not been investigated.
  4. **Cost-Sensitive Threshold Analysis:** The operational trade-off between False Negatives (fish suffocation / economic loss) and False Positives (unnecessary aerator energy consumption) has not been evaluated to justify decision thresholds across different farm cost regimes.
- **Recommended action:**
  Implement `src/calibration.py` to compute Brier score, generate reliability diagrams, evaluate Platt/isotonic calibration on training folds, and conduct cost-sensitive threshold analysis. Produce `CALIBRATION_ANALYSIS.md`.

---

## Mentor Requirement 5 — Stronger Baselines

- **Status:** PARTIAL (Naive baselines present; Domain baselines missing)
- **Evidence in Repository:**
  - `results/reports/MODEL_COMPARISON.csv` contains:
    - Majority Baseline (PR-AUC: $0.1142$, F1: $0.0000$)
    - Current-DO Threshold Baseline (`current_do <= 4.2 mg/L`, PR-AUC: $0.6149$, F1: $0.4516$)
- **What is already satisfied:**
  1. Naive majority baseline establishing the prevalence floor ($0.1142$).
  2. Current-DO threshold baseline establishing the value of instantaneous sensor readings ($0.6149$).
- **What is missing:**
  1. **Persistence Baseline:** A standardized time-series baseline. Under the ShinerAI operational formulation ($\text{DO}_T \ge 3.0\text{ mg/L}$), strict persistence would forecast that current DO remains above $3.0\text{ mg/L}$ indefinitely (always predicting SAFE, identical to majority class). An operationally meaningful formulation must be defined, tested, and documented.
  2. **Simple Trend-Based Forecasting Baseline:** Linear extrapolation based on recent DO trajectory (e.g., slope over the past 30–120 minutes) projected 2 hours forward. If the projected value is $< 3.0\text{ mg/L}$, predict AT_RISK. This is an essential domain heuristic that benchmarks whether machine learning outperforms basic linear calculus.
- **Recommended action:**
  Implement both Persistence and Linear Trend baselines in `src/baselines.py`. Evaluate them across all 9 standardized metrics on the identical 8,261 test samples and update `MODEL_COMPARISON.md`.

---

## Mentor Requirement 6 — Domain Validation

- **Status:** PARTIAL (Disclaimers in place; Formal physiological grounding missing)
- **Evidence in Repository:**
  - `QC_CLEANING_POLICY.md` line 134 states: *"3.0 mg/L Dissolved Oxygen is adopted as a provisional project standard... Species-specific physiological justifications must be validated before formal academic publication."*
  - `PROJECT_DOCUMENTATION.md` states that $3.0\text{ mg/L}$ is an engineering threshold, not a universal biological constant.
- **What is already satisfied:**
  1. Clear scientific language guardrails avoiding overclaiming universal biological validity.
  2. Identification of target species in internal dataset (Golden Shiner, *Notemigonus crysoleucas*, Lonoke County, AR) and external dataset (Nile Tilapia, *Oreochromis niloticus*, Oman).
- **What is missing:**
  1. Comprehensive literature review citing aquaculture authorities (e.g., Boyd 1998, 2014; Stone et al., SRAC Publication 120 on Baitfish Aquaculture; FAO water quality standards) validating why commercial managers use $3.0\text{–}4.0\text{ mg/L}$ as the operational trigger for emergency aeration.
  2. Distinction between **acute lethal hypoxia** ($< 1.0\text{–}1.5\text{ mg/L}$) and **chronic sub-lethal stress / operational early warning boundary** ($3.0\text{ mg/L}$), explaining why early warning must precede acute mortality by several hours.
- **Recommended action:**
  Compile species-specific physiological and management literature for Golden Shiner and warmwater pond aquaculture. Update `PROJECT_DOCUMENTATION.md` and documentation files with formal citations and operational rationale.

---

## Mentor Requirement 7 — Independent Reproduction

- **Status:** PARTIAL / REPRODUCIBILITY AUDITED, FORMAL END-TO-END RERUN REPORT NEEDED
- **Evidence in Repository:**
  - `results/audit/ACTIVE_MODEL_REPRODUCTION.md`
  - `results/audit/MASTER_METRIC_RECALCULATION.csv`
  - `results/audit/METRIC_RECONCILIATION.csv`
  - `tests/test_audit_verification.py`
  - 97 automated tests passing in `pytest`
- **What is already satisfied:**
  1. Verified that frozen model `models/xgboost_config_c.joblib` reproduces exact internal metrics to 4 decimal places (PR-AUC $= 0.7574$, ROC-AUC $= 0.9162$, F1 $= 0.6285$, Recall $= 0.7975$, Precision $= 0.5186$, Specificity $= 0.9046$, Accuracy $= 0.8924$; $\text{TP}=752, \text{FP}=698, \text{TN}=6620, \text{FN}=191$).
  2. Unit and integration test suite passing with 100% success rate across 10 test modules.
- **What is missing:**
  1. A single consolidated script that runs the entire pipeline from raw CSVs -> cleaning -> labeling -> splitting -> training -> evaluation -> report generation, verifying that all intermediate row counts and final tables match identically.
  2. A formal `REPRODUCTION_REPORT.md` tabulating every stage, artifact hash, and discrepancy check.
- **Recommended action:**
  Create and execute `scripts/verify_full_reproduction.py` and document results in `REPRODUCTION_REPORT.md`.

---

## Overall Research Readiness

### Summary of Repository State:
- **Completed Items:**
  - Frozen core task formulation ($\text{DO}_T \ge 3.0\text{ mg/L} \to \text{DO}_{T+2\text{h}} < 3.0\text{ mg/L}$)
  - Leak-free temporal splitting with mandatory 2-hour purge gap
  - Feature configurations (A, B, C) and ablation table
  - Master model evaluation across 11 configurations
  - SHAP TreeExplainer global and local attributions
  - Initial external validation on Oman Tilapia dataset (specificity confirmed)
  - Automated test suite (97 tests passing)
- **Partial Items:**
  - Research novelty (empirical only; lacks event-level or operational formulation)
  - Baseline comparisons (lacks persistence and trend baselines)
  - External validation (specificity verified, sensitivity unverified due to zero positive events in Oman data; Oman 5 FP vs 0 FP discrepancy requires formal documentation)
  - Domain validation (disclaimers present, formal physiological literature uncited)
  - Reproduction (reproduction verified via tests; standalone end-to-end report needed)
- **Missing Items:**
  - Operational usefulness evaluation (lead time distribution, missed events, alerts/pond/day, alert stability)
  - Event-level evaluation (episode grouping, event detection rate)
  - Probability calibration (Brier score, reliability curves, Platt/isotonic scaling)
  - Cost-sensitive decision threshold optimization
  - Audit of second external dataset (Andhra Pradesh aquaculture dataset)
  - Formal novelty investigation and decision document (`NOVELTY_DECISION.md`)

---

## Prioritized Implementation Roadmap

1. **Stage 1: Stronger Baselines & Operational Evaluation**
   - Implement Persistence Baseline and Simple Trend Forecasting Baseline (`src/baselines.py`).
   - Implement Operational Evaluation module (`src/operational_evaluation.py`) computing warning lead time, missed events, false alarms per pond-day, and alert stability.
   - Implement Event-Level grouping and metrics.
   - Create `OPERATIONAL_EVALUATION.md` and update `MODEL_COMPARISON.md`.

2. **Stage 2: Probability Calibration & Cost-Sensitive Thresholds**
   - Implement Brier score, reliability diagrams, and calibration analysis (`src/calibration.py`).
   - Conduct validation-based cost-sensitive threshold analysis.
   - Create `CALIBRATION_ANALYSIS.md`.

3. **Stage 3: External Validation Audit & Second Dataset Exploration**
   - Document the Oman discrepancy reconciliation formally.
   - Audit the Andhra Pradesh aquaculture dataset for format, timestamps, sampling cadence, and hypoxia occurrences.
   - If viable, evaluate frozen model; if non-viable, document reasons rigorously.

4. **Stage 4: Novelty Investigation & Decision**
   - Formulate `NOVELTY_DECISION.md` evaluating Event-Level Early Warning vs. Adaptive Thresholds vs. Uncertainty Quantification.
   - Select and implement the single defensible contribution supported by empirical data.

5. **Stage 5: Domain Validation & Independent Reproduction**
   - Document species-specific physiological literature and aquaculture guidelines for $3.0\text{ mg/L}$.
   - Execute full-pipeline reproduction verification and author `REPRODUCTION_REPORT.md`.

6. **Stage 6: Master Notebook, Documentation & Test Suite Consolidation**
   - Update `notebooks/ShinerAI_Complete_ML_Pipeline.ipynb`.
   - Update `PROJECT_DOCUMENTATION.md` and `README.md`.
   - Expand `pytest` test suite to cover all new modules.
   - Author final `RESEARCH_READINESS_REPORT.md`.
