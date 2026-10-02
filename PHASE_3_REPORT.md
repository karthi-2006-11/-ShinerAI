# Phase 3 Report: Machine Learning Training & Evaluation
**Project:** ShinerAI  
**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Phase:** Phase 3 — Machine Learning Training + Evaluation  
**Status:** In Progress (Calibrating Baseline Reporting)  
**Generated Date:** October 2026  

---

## 1. Executive Summary & Objective

Phase 3 established the classical machine learning benchmarking and evaluation framework for **ShinerAI**. The core scientific objective was to evaluate whether classical algorithms can provide a reliable 2-hour early warning before dissolved oxygen (DO) breaches the critical 3.0 mg/L threshold in commercial aquaculture ponds.

**Target Scope Definition:**  
The model predicts **impending low-dissolved-oxygen events** ($\text{DO} < 3.0\text{ mg/L}$ within the next 2 hours given current $\text{DO} \ge 3.0\text{ mg/L}$).  
*The project predicts water oxygen depletion dynamics; it does not directly predict fish disease or fish mortality.*

Using the leak-free, 41,277-example supervised learning dataset produced in Phase 2 ([`data/processed/ml_ready_dataset.csv`](file:///d:/FISH/data/processed/ml_ready_dataset.csv)), Phase 3 evaluated three feature configurations across multiple candidate models using strictly leak-free validation regimes:
1. **Primary Evaluation (Temporal Holdout):** The earlier 80% of time for each pond was used for training (32,908 examples), followed by a mandatory **2-hour temporal purge gap** (108 examples), and the latest 20% of time reserved as an unseen temporal test set (8,261 examples).
2. **Secondary Evaluation (Held-Out Pond Generalization):** A 5-Fold GroupKFold cross-validation across the 17 monitored ponds ensured models were evaluated on ponds completely excluded from the training partition.

### Key Findings Summary:
- **Primary Metric (PR-AUC):** **XGBoost Config C (DO History Only)** achieved the highest PR-AUC (**0.7574**), followed closely by **Random Forest Config C** (**0.7471**), **Random Forest Config B** (**0.7420**), and **XGBoost Config B** (**0.7353**).
- **Default Operating Threshold ($p = 0.50$):** **Random Forest Config C** delivered the strongest precision (**58.59%**), specificity (**93.11%**), and F1-score (**0.6602**), resulting in fewer false alarms at the default threshold (504 false positives), which could reduce unnecessary interventions in a deployment where alerts trigger aeration.
- **High Sensitivity Alternative:** **Logistic Regression Config B** achieved the highest recall (**89.93%**, catching 848 of 943 low-DO events), but produced 2,158 false alarms (precision 28.21%).
- **Value of Historical Telemetry:** A consistent improvement was observed across the three tested model families when historical telemetry was included beyond instantaneous measurements.
- **Config C Parsimony:** Within this dataset and tested feature configuration, recent DO history carried the strongest predictive signal. For both Random Forest and XGBoost, the DO-history-only configuration achieved PR-AUC at least as high as the full-history configuration.
- **Held-Out Pond Performance:** The models provided evidence of generalization to held-out ponds within this dataset, retaining predictive stability in 5-Fold GroupKFold cross-validation (Random Forest mean PR-AUC: **0.7086 ± 0.0456**; XGBoost mean PR-AUC: **0.7183 ± 0.0467**).

---

## 2. Input Dataset Verification

Before any model training, automated pre-flight integrity checks verified the ML-ready dataset:
- Source file: [`data/processed/ml_ready_dataset.csv`](file:///d:/FISH/data/processed/ml_ready_dataset.csv)
- Total observations: Exactly **41,277 rows**
- Total columns: Exactly **37 columns**
- Target values: Strictly binary $\{0, 1\}$ (36,101 SAFE, 5,176 AT_RISK)
- Missing values: Exactly **0 NaN / null values**
- Baseline condition: All rows satisfy $\text{current\_do} \ge 3.0\text{ mg/L}$ (minimum observed: 3.00 mg/L)
- Pond coverage: All **17 commercial ponds** represented with both classes present

---

## 3. Final Feature Configurations

To eliminate multicollinearity and redundant inputs, the duplicate current readings (`do_t`, `ph_t`, `temp_t`) were omitted in favor of explicit `current_*` columns. Non-predictor identifiers (`pond_id`, `prediction_timestamp`, `target`, `target_name`, `data_quality_status`) and future target columns were strictly quarantined from model training.

Three distinct feature configurations were evaluated:

| Configuration | Feature Count | Features Included | Scientific Purpose |
|---|---|---|---|
| **Config A (Current Only)** | 5 | `current_do`, `current_ph`, `current_temperature`, `hour_of_day`, `minute_of_day` | Benchmark: How well can we predict future risk using only the present moment and diurnal time? |
| **Config B (Current + Full History)** | 29 | Config A + 8 DO lags ($t-15\text{m} \dots t-120\text{m}$) + 8 pH lags + 8 Temp lags | Full sensor suite: Evaluates multi-parameter trends and interactions. |
| **Config C (DO History Only)** | 11 | `current_do`, `hour_of_day`, `minute_of_day` + 8 DO lags ($t-15\text{m} \dots t-120\text{m}$) | Parsimony test: How much predictive power stems from recent DO trajectory alone? |

*(Note: Features represent historical discrete lag values at 15-minute intervals; no explicit mathematical derivative or slope feature was engineered).*

---

## 4. Models Trained, Baseline Provenance, & Imbalance Handling

### Candidate Algorithms & Baselines:
1. **Majority-Class Baseline:** Naively predicts the majority class (`SAFE = 0`) with the empirical train prior ($P(\text{AT\_RISK}) = 0.1282$).
2. **Current-DO Baselines:** Separated into two distinct evaluations to ensure methodological clarity:
   - **Current-DO Threshold Baseline (DO <= 4.2 mg/L):** A direct Boolean decision rule. The threshold was **derived strictly using training data only (Option B)** via a grid search across $th \in [3.0, 6.0]$ in increments of $0.1\text{ mg/L}$ on `train_df` ($N = 32,908$) that maximized training F1-score ($th = 4.20\text{ mg/L}$, train F1 = 0.5940; zero test data used). Evaluated directly on the holdout test set as a binary classifier, it yields:
     $$\text{TP} = 667, \quad \text{FP} = 522, \quad \text{TN} = 6796, \quad \text{FN} = 276$$
     $$\text{Recall} = 70.73\%, \quad \text{Precision} = 56.10\%, \quad \text{F1} = 0.6257, \quad \text{Specificity} = 92.87\%$$
     *(PR-AUC and ROC-AUC are N/A for this discrete Boolean rule as it produces no continuous ranking).*
   - **Current-DO Ranking Baseline (1-D Logistic):** A continuous 1-D logistic regression fitted on `train_df[['current_do']]` with balanced weighting to evaluate the rank-ordering capability of instantaneous DO across all possible thresholds. Across the entire threshold spectrum, this continuous model yields:
     $$\text{PR-AUC} = 0.6149, \quad \text{ROC-AUC} = 0.9024$$
     At its default balanced probability threshold ($p = 0.50$, corresponding to alerting whenever $\text{current\_do} \le 5.93\text{ mg/L}$ due to balanced reweighting), it detects 845 events ($\text{Recall} = 89.61\%$) with 1,954 false alarms ($\text{Precision} = 30.19\%, \text{Specificity} = 73.30\%, \text{F1} = 0.4516$).
3. **Logistic Regression:** Linear classification baseline with a `StandardScaler` pipeline and `class_weight='balanced'`.
4. **Random Forest:** Ensemble of 100 decision trees (`max_depth=12`, `class_weight='balanced'`, `random_state=42`).
5. **XGBoost:** Gradient-boosted decision tree ensemble (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=6.80`, `random_state=42`).

### Class Imbalance Strategy:
The dataset exhibits an 87.46% SAFE vs. 12.54% AT_RISK distribution (6.97 : 1 class ratio).
- **No synthetic oversampling (SMOTE) was used**, avoiding artificial interpolation across physical sensor signals.
- Inbalance was addressed strictly via algorithmic sample weighting derived solely from the training partition:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{train, safe}}}{N_{\text{train, at\_risk}}} = \frac{28,689}{4,219} \approx 6.80$$
- This penalizes missed low-DO events ~6.8x more heavily than false alarms, aligning model optimization with operational aquaculture risk.

---

## 5. Temporal Train/Test Split & 2-Hour Purge Rule

To prevent severe time-series leakage, splitting was performed chronologically within each pond:
1. For each pond, observations were ordered chronologically by `prediction_timestamp`.
2. The 80th percentile timestamp ($t_{\text{cutoff}}$) separated historical training from future testing.
3. **The 2-Hour Purge Rule:** Any training observation within $[t_{\text{cutoff}} - 2\text{ hours}, t_{\text{cutoff}})$ was purged. Because the target label inspects the subsequent 2 hours $(T, T + 2\text{h}]$, an example recorded at $t_{\text{cutoff}} - 15\text{m}$ would otherwise inspect future telemetry belonging to the test period.
4. Exactly **108 boundary observations** across the 17 ponds were purged, guaranteeing that the maximum future timestamp in the training labels strictly precedes the earliest test observation.

### Split Size Summary:
- **Training Set:** 32,908 observations (28,689 SAFE, 4,219 AT_RISK — 12.82% positive)
- **Purged Boundary:** 108 observations (94 SAFE, 14 AT_RISK)
- **Temporal Test Set:** 8,261 observations (7,318 SAFE, 943 AT_RISK — 11.42% positive)
- **Strict Verification:** $32,908 + 108 + 8,261 = 41,277$ (0 unexplained rows)

---

## 6. Comprehensive Model Comparison (Temporal Holdout)

Benchmark results evaluated on the 8,261 unseen temporal test examples ([`results/reports/model_comparison.csv`](file:///d:/FISH/results/reports/model_comparison.csv) and [`results/reports/final_model_selection.csv`](file:///d:/FISH/results/reports/final_model_selection.csv)):

| Model | Feature Set | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|---|
| **Majority Baseline** | None | 0.1142 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8858 |
| **Current-DO Threshold Baseline ($\le 4.2$)** | `current_do` only (Boolean rule) | N/A | N/A | 0.6257 | 0.7073 | 0.5610 | 0.9287 | 0.9034 |
| **Current-DO Ranking Baseline (1-D Logistic)** | `current_do` only (continuous ranking) | 0.6149 | 0.9024 | 0.4516 | 0.8961 | 0.3019 | 0.7330 | 0.7516 |
| **Logistic Regression** | Config A (Current Only) | 0.6019 | 0.8994 | 0.4274 | 0.9003 | 0.2802 | 0.7020 | 0.7246 |
| **Random Forest** | Config A (Current Only) | 0.7107 | 0.9116 | 0.6174 | 0.7709 | 0.5149 | 0.9064 | 0.8909 |
| **XGBoost** | Config A (Current Only) | 0.7317 | 0.9150 | 0.5817 | 0.8102 | 0.4537 | 0.8743 | 0.8670 |
| **Logistic Regression** | Config B (Current + History) | 0.6763 | 0.9078 | 0.4295 | 0.8993 | 0.2821 | 0.7051 | 0.7273 |
| **Random Forest** | Config B (Current + History) | **0.7420** | **0.9169** | **0.6233** | **0.7614** | **0.5276** | **0.9121** | **0.8949** |
| **XGBoost** | Config B (Current + History) | **0.7353** | **0.9171** | **0.5922** | **0.7932** | **0.4725** | **0.8859** | **0.8753** |
| **Logistic Regression** | Config C (DO History Only) | 0.6550 | 0.9093 | 0.4549 | 0.8940 | 0.3051 | 0.7376 | 0.7555 |
| **Random Forest** | Config C (DO History Only) | **0.7471** | **0.9144** | **0.6602** | **0.7561** | **0.5859** | **0.9311** | **0.9111** |
| **XGBoost** | Config C (DO History Only) | **0.7574** | **0.9162** | **0.6285** | **0.7975** | **0.5186** | **0.9046** | **0.8924** |

### Confusion Matrix Breakdown (Temporal Test Set, 8,261 Observations, 943 Low-DO Events):
- **Current-DO Threshold Baseline (DO <= 4.2 mg/L):**
  - Impending low-DO events detected (TP): **667** (70.73% recall)
  - Missed low-DO events (FN): **276**
  - False alarms (FP): **522** (92.87% specificity)
  - Correctly safe (TN): **6,796**
- **Current-DO Ranking Baseline (1-D Logistic at $p=0.50$):**
  - Impending low-DO events detected (TP): **845** (89.61% recall)
  - Missed low-DO events (FN): **98**
  - False alarms (FP): **1,954** (73.30% specificity)
  - Correctly safe (TN): **5,364**
- **Random Forest (Config B):**
  - Impending low-DO events detected (TP): **718** (76.14% recall)
  - Missed low-DO events (FN): **225**
  - False alarms (FP): **643** (91.21% specificity)
  - Correctly safe (TN): **6,675**
- **Random Forest (Config C):**
  - Impending low-DO events detected (TP): **713** (75.61% recall)
  - Missed low-DO events (FN): **230**
  - False alarms (FP): **504** (93.11% specificity)
  - Correctly safe (TN): **6,814**
- **XGBoost (Config B):**
  - Impending low-DO events detected (TP): **748** (79.32% recall)
  - Missed low-DO events (FN): **195**
  - False alarms (FP): **835** (88.59% specificity)
  - Correctly safe (TN): **6,483**
- **XGBoost (Config C):**
  - Impending low-DO events detected (TP): **752** (79.75% recall)
  - Missed low-DO events (FN): **191**
  - False alarms (FP): **698** (90.46% specificity)
  - Correctly safe (TN): **6,620**
- **Logistic Regression (Config B):**
  - Impending low-DO events detected (TP): **848** (89.93% recall)
  - Missed low-DO events (FN): **95**
  - False alarms (FP): **2,158** (70.51% specificity)
  - Correctly safe (TN): **5,160**

---

## 7. Secondary Evaluation: Generalization Across Held-Out Ponds

To evaluate whether models retain predictive performance when transferred to ponds excluded from training, a 5-Fold GroupKFold cross-validation was conducted across the 17 ponds ([`results/reports/group_kfold_performance.csv`](file:///d:/FISH/results/reports/group_kfold_performance.csv)):

| Model (Config B) | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 | Mean Recall | Mean Precision |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.5950 | ± 0.1175 | 0.8941 ± 0.0304 | 0.4589 ± 0.0720 | **0.8662 ± 0.0677** | 0.3167 ± 0.0718 |
| **Random Forest** | **0.7086** | **± 0.0456** | **0.9069 ± 0.0313** | **0.6326 ± 0.0426** | 0.7076 ± 0.0983 | **0.5867 ± 0.0832** |
| **XGBoost** | **0.7183** | **± 0.0467** | **0.9041 ± 0.0285** | **0.6028 ± 0.0533** | **0.7379 ± 0.0983** | 0.5262 ± 0.1019 |

### Observations on Held-Out Ponds:
- Both Random Forest and XGBoost provided evidence of generalization to held-out ponds within this dataset, retaining average PR-AUCs exceeding **0.70** with low standard deviations (< 0.047).
- The models retained predictive performance when evaluated on ponds excluded from training, confirming that their learned decision rules transfer across different physical ponds in the monitoring campaign.

---

## 8. Neutral Model Selection & Operational Trade-offs

A comprehensive model selection table was produced to evaluate trade-offs objectively ([`results/reports/final_model_selection.csv`](file:///d:/FISH/results/reports/final_model_selection.csv)):

| Model | Feature Config | PR-AUC | ROC-AUC | Precision | Recall | F1 | Specificity | GroupKFold PR-AUC | Operational Role / Trade-off |
|---|---|---|---|---|---|---|---|---|---|
| **XGBoost** | Config C | **0.7574** | 0.9162 | 0.5186 | **0.7975** | 0.6285 | 0.9046 | N/A | **Highest PR-AUC overall.** Strong balance of sensitivity (79.8% recall) and ranking quality across thresholds. |
| **Random Forest** | Config C | **0.7471** | 0.9144 | **0.5859** | 0.7561 | **0.6602** | **0.9311** | N/A | **Highest F1 and Specificity.** Produces fewer false alarms (504 FPs) at the default threshold, which could reduce unnecessary interventions where alerts trigger aeration. |
| **Random Forest** | Config B | **0.7420** | **0.9169** | 0.5276 | 0.7614 | 0.6233 | 0.9121 | 0.7086 ± 0.0456 | **Balanced multi-sensor profile.** Stable held-out pond generalization across 17 ponds. |
| **XGBoost** | Config B | **0.7353** | **0.9171** | 0.4725 | **0.7932** | 0.5922 | 0.8859 | **0.7183 ± 0.0467** | **High sensitivity with full sensors.** Highest GroupKFold cross-validation PR-AUC. |
| **Logistic Regression** | Config B | 0.6763 | 0.9078 | 0.2821 | **0.8993** | 0.4295 | 0.7051 | 0.5950 ± 0.1175 | **High-Recall alternative.** Catches 89.9% of low-DO events, but generates 2,158 false alarms. |
| **Current-DO Threshold Baseline ($\le 4.2$)** | `current_do` only (Boolean rule) | N/A | N/A | 0.5610 | 0.7073 | 0.6257 | 0.9287 | N/A | **Training-derived Boolean heuristic.** Fixed threshold classification; catches 70.7% of events with 522 false alarms. |
| **Current-DO Ranking Baseline (1-D Logistic)** | `current_do` only (continuous ranking) | 0.6149 | 0.9024 | 0.3019 | 0.8961 | 0.4516 | 0.7330 | N/A | **Continuous 1-D ranking baseline.** Evaluates discriminative capacity of instantaneous DO alone across all thresholds. |
| **Majority Baseline** | None | 0.1142 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A | **Naive benchmark.** Demonstrates why high accuracy (88.58%) is misleading. |

### Operational Recommendation Framing:
No single model is universally "best" for all scenarios; selection depends directly on the farm's operational objective:
1. **If the priority is overall threshold-independent discriminative power (PR-AUC):** **XGBoost Config C** is the strongest performer (PR-AUC = 0.7574).
2. **If the priority is minimizing false alarms at the default threshold:** **Random Forest Config C** achieves the highest specificity (93.11%), highest precision (58.59%), and highest F1 (0.6602), producing fewer false alarms (504 FPs) at the default threshold, which could reduce unnecessary interventions in a deployment where alerts trigger aeration.
3. **If the priority is maximum low-DO detection (risk-averse operation):** **Logistic Regression Config B** offers the highest raw recall (89.93%), or **XGBoost Config C** offers a high-recall tree alternative (79.75%) with far fewer false alarms (698 vs. 2,158).

---

## 9. Main Scientific Observations

1. **Does History Provide Meaningful Information Beyond Current DO?**  
   **A consistent improvement was observed across the three tested model families.**
   - In Logistic Regression, PR-AUC improved from 0.6019 (Config A) to 0.6763 (Config B).
   - In Random Forest, PR-AUC improved from 0.7107 (Config A) to 0.7420 (Config B) and 0.7471 (Config C).
   - In XGBoost, PR-AUC improved from 0.7317 (Config A) to 0.7353 (Config B) and 0.7574 (Config C).
   - The recent temporal trajectory of DO provides additional predictive information beyond the current measurement. *(Note: Features consist of raw 15-minute historical lag observations; no explicit rate-of-change or derivative feature was engineered).*

2. **The Signal in Config C (DO History Only):**  
   For both Random Forest and XGBoost, the DO-history-only configuration achieved PR-AUC at least as high as the full-history configuration in the current experiment. Within this dataset and tested feature configuration, recent DO history carried the strongest predictive signal. This does not indicate that pH and temperature are universally uninformative in aquaculture, but rather that in this 9-week winter monitoring period, recent oxygen trajectory provided the primary predictive variance.

3. **Why Accuracy is Deceptive in Early-Warning Systems:**  
   The Majority Baseline achieved 88.58% accuracy while failing to detect a single low-DO event (0.0% Recall). Evaluating early-warning systems by accuracy creates a dangerous illusion of safety. PR-AUC, Recall, and Specificity are the vital operational metrics.

---

## 10. Limitations & Scope Clarifications

1. **Target Boundary:** The model predicts impending low-dissolved-oxygen events ($\text{DO} < 3.0\text{ mg/L}$ in 2 hours); **it does not directly predict fish mortality or disease**.
2. **Provisional Biological Threshold:** 3.0 mg/L remains an operational biological threshold; tolerance varies across fish species and growth stages.
3. **Seasonal Window:** Data spans 9 winter weeks in Andhra Pradesh, India; seasonal generalizability to monsoon or summer conditions remains unverified.
4. **Baseline Hyperparameters:** Models were evaluated using sensible baseline hyperparameters without exhaustive search.
5. **Research Prototype:** ShinerAI Phase 3 models are research artifacts, not yet connected to live farm control systems.

---

## 11. Visualizations Reference

All generated figures are saved in [`results/figures/`](file:///d:/FISH/results/figures/):
1. **Confusion Matrices:**
   - Logistic Regression: [`results/figures/confusion_logistic_regression.png`](file:///d:/FISH/results/figures/confusion_logistic_regression.png)
   - Random Forest: [`results/figures/confusion_random_forest.png`](file:///d:/FISH/results/figures/confusion_random_forest.png)
   - XGBoost: [`results/figures/confusion_xgboost.png`](file:///d:/FISH/results/figures/confusion_xgboost.png)
2. **Curve Comparisons:**
   - ROC Curve Comparison: [`results/figures/roc_curve_comparison.png`](file:///d:/FISH/results/figures/roc_curve_comparison.png)
   - Precision-Recall Curve Comparison: [`results/figures/pr_curve_comparison.png`](file:///d:/FISH/results/figures/pr_curve_comparison.png)
3. **ML Pipeline Architecture:**
   - Machine Learning Workflow: [`results/figures/ml_pipeline_diagram.png`](file:///d:/FISH/results/figures/ml_pipeline_diagram.png)

---

## 12. Reproducibility Instructions

All results can be reproduced by executing:
```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Train all baseline and ML models, save joblib artifacts & comparison reports
python src/train.py

# 3. Generate per-pond breakdown, GroupKFold cross-validation, and all figures
python src/evaluate.py

# 4. Run automated test suite (32 unit tests)
pytest -v
```

---

## 13. Strict Phase Boundary Confirmation

In accordance with strict project boundaries:
- No web application, Flask/FastAPI backend, or REST API has been built.
- No frontend dashboard or mobile UI has been created.
- No IoT device or microcontroller hardware has been interfaced.
- No fish disease, pathogen, or mortality prediction has been attempted.
- No deep learning (LSTM/GRU) or SHAP interpretability packages have been added.
- **Phase 4 has NOT been started.**
