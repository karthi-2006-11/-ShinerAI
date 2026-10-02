# Phase 3 Report: Machine Learning Training & Evaluation
**Project:** ShinerAI  
**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Phase:** Phase 3 — Machine Learning Training + Evaluation  
**Status:** Completed  
**Generated Date:** October 2026  

---

## 1. Executive Summary & Objective

Phase 3 established the first machine learning models for ShinerAI, evaluating whether classical algorithms can provide a reliable 2-hour early warning before dissolved oxygen breaches the critical 3.0 mg/L threshold in commercial aquaculture ponds.

Using the leak-free, 41,277-example supervised learning dataset produced in Phase 2 ([`data/processed/ml_ready_dataset.csv`](file:///d:/FISH/data/processed/ml_ready_dataset.csv)), Phase 3 tested three feature configurations across multiple candidate models using strictly leak-free validation regimes:
1. **Primary Evaluation (Temporal Holdout):** The earlier 80% of time for each pond was used for training (32,908 examples), followed by a mandatory **2-hour temporal purge gap** (108 examples), and the latest 20% of time reserved as an unseen temporal test set (8,261 examples).
2. **Secondary Evaluation (Unseen-Pond Generalization):** A 5-Fold GroupKFold cross-validation across the 17 monitored ponds ensured models were tested exclusively on ponds unseen during training.

### Key Finding:
**Temporal history substantially improves predictive accuracy over current-only conditions.**
- Random Forest achieved a primary **PR-AUC of 0.7420 to 0.7471**, an **F1-score of 0.6233 to 0.6602**, and a **Recall of 76.1% to 75.6%** with high Specificity (91.2% to 93.1%).
- XGBoost achieved a primary **PR-AUC of 0.7353 to 0.7574**, an **F1-score of 0.5922 to 0.6285**, and a **Recall of 79.3% to 79.8%**.
- In comparison, the **Majority Baseline** scored **PR-AUC 0.1142** (Recall 0.0%), and the **Current-DO Baseline** scored **PR-AUC 0.6149** (Recall 89.6%, but Precision only 30.2%, generating 1,954 false alarms).
- On unseen ponds (GroupKFold), Random Forest and XGBoost maintained strong generalizability with mean PR-AUC of **0.7086 ± 0.0456** and **0.7183 ± 0.0467** respectively.

---

## 2. Input Dataset Verification

Before any model training, automated pre-flight integrity checks verified:
- Source file: [`data/processed/ml_ready_dataset.csv`](file:///d:/FISH/data/processed/ml_ready_dataset.csv)
- Total observations: Exactly **41,277 rows**
- Total columns: Exactly **37 columns**
- Target values: Strictly binary $\{0, 1\}$ (36,101 SAFE, 5,176 AT_RISK)
- Missing values: Exactly **0 NaN / null values**
- Baseline condition: All rows satisfy $\text{current\_do} \ge 3.0\text{ mg/L}$ (minimum observed: 3.00 mg/L)
- Pond coverage: All **17 commercial ponds** represented with both classes present

---

## 3. Final Feature Configurations

To eliminate multicollinearity and redundant inputs, the duplicate current readings (`do_t`, `ph_t`, `temp_t`) were omitted in favor of explicit `current_*` columns. Non-predictor identifiers (`pond_id`, `prediction_timestamp`, `target`, `target_name`, `data_quality_status`) were quarantined from model training.

Three distinct feature configurations were evaluated:

| Configuration | Feature Count | Features Included | Scientific Purpose |
|---|---|---|---|
| **Config A (Current Only)** | 5 | `current_do`, `current_ph`, `current_temperature`, `hour_of_day`, `minute_of_day` | Benchmark: How well can we predict future risk using only the present moment? |
| **Config B (Current + Full History)** | 29 | Config A + 8 DO lags ($t-15\text{m} \dots t-120\text{m}$) + 8 pH lags + 8 Temp lags | Full input set: Evaluates the combined dynamics of multi-parameter trends. |
| **Config C (DO History Only)** | 11 | `current_do`, `hour_of_day`, `minute_of_day` + 8 DO lags ($t-15\text{m} \dots t-120\text{m}$) | Parsimony test: How much predictive power stems from DO dynamics alone? |

---

## 4. Models Trained & Imbalance Handling

### Candidate Algorithms:
1. **Majority-Class Baseline:** Naively predicts the majority class (`SAFE = 0`) with the empirical train prior ($P(\text{AT\_RISK}) = 0.1282$).
2. **Current-DO Baseline:** A physical rule-of-thumb threshold baseline. On training data, $\text{DO} \le 4.2\text{ mg/L}$ maximizes F1. A 1D Logistic Regression on `current_do` generates continuous risk probabilities.
3. **Logistic Regression:** Linear classification baseline with a `StandardScaler` pipeline and `class_weight='balanced'`.
4. **Random Forest:** Ensemble of 100 decision trees (`max_depth=12`, `class_weight='balanced'`, `random_state=42`).
5. **XGBoost:** Gradient-boosted decision tree ensemble (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=6.80`, `random_state=42`).

### Class Imbalance Strategy:
The dataset has an 87.46% SAFE vs. 12.54% AT_RISK distribution (6.97 : 1 class ratio).
- **No synthetic oversampling (SMOTE) was used**, avoiding artificial interpolation across sensor signals.
- Inbalance was addressed strictly via algorithmic sample weighting derived solely from the training partition:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{train, safe}}}{N_{\text{train, at\_risk}}} = \frac{28,689}{4,219} \approx 6.80$$
- This penalizes missed AT_RISK events ~6.8x more heavily than false alarms, aligning model behavior with real-world aquaculture risk.

---

## 5. Temporal Train/Test Split & 2-Hour Purge Rule

To prevent severe time-series leakage, splitting was performed strictly by timestamp within each pond:
1. For each pond, observations were ordered chronologically by `prediction_timestamp`.
2. The 80th percentile timestamp ($t_{\text{cutoff}}$) separated historical training from future testing.
3. **The 2-Hour Purge Rule:** Any training observation within $[t_{\text{cutoff}} - 2\text{ hours}, t_{\text{cutoff}})$ was purged. Because the target label inspects the subsequent 2 hours $(T, T + 2\text{h}]$, an example recorded at $t_{\text{cutoff}} - 15\text{m}$ would otherwise inspect future telemetry belonging to the test period.
4. Exactly **108 boundary observations** across the 17 ponds were purged, guaranteeing that the maximum future timestamp in the training labels strictly precedes the earliest test observation.

### Split Size Summary:
- **Training Set:** 32,908 observations (28,689 SAFE, 4,219 AT_RISK — 12.82% positive)
- **Purged Boundary:** 108 observations
- **Temporal Test Set:** 8,261 observations (7,318 SAFE, 943 AT_RISK — 11.42% positive)
- **Strict Verification:** $32,908 + 108 + 8,261 = 41,277$ (0 unexplained rows)

---

## 6. Comprehensive Model Comparison (Temporal Holdout)

The complete benchmark results on the 8,261 unseen temporal test examples ([`results/reports/model_comparison.csv`](file:///d:/FISH/results/reports/model_comparison.csv)):

| Model | Feature Set | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|---|
| **Majority Baseline** | None | 0.1142 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8858 |
| **Current-DO Baseline ($\le 4.2$)** | `current_do` only | 0.6149 | 0.9024 | 0.4516 | 0.8961 | 0.3019 | 0.7330 | 0.7516 |
| **Logistic Regression** | Config A (Current Only) | 0.6019 | 0.8994 | 0.4274 | 0.9003 | 0.2802 | 0.7020 | 0.7246 |
| **Random Forest** | Config A (Current Only) | 0.7107 | 0.9116 | 0.6174 | 0.7709 | 0.5149 | 0.9064 | 0.8909 |
| **XGBoost** | Config A (Current Only) | 0.7317 | 0.9150 | 0.5817 | 0.8102 | 0.4537 | 0.8743 | 0.8670 |
| **Logistic Regression** | Config B (Current + History) | 0.6763 | 0.9078 | 0.4295 | 0.8993 | 0.2821 | 0.7051 | 0.7273 |
| **Random Forest** | Config B (Current + History) | **0.7420** | **0.9169** | **0.6233** | **0.7614** | **0.5276** | **0.9121** | **0.8949** |
| **XGBoost** | Config B (Current + History) | **0.7353** | **0.9171** | **0.5922** | **0.7932** | **0.4725** | **0.8859** | **0.8753** |
| **Logistic Regression** | Config C (DO History Only) | 0.6550 | 0.9093 | 0.4549 | 0.8940 | 0.3051 | 0.7376 | 0.7555 |
| **Random Forest** | Config C (DO History Only) | **0.7471** | **0.9144** | **0.6602** | **0.7561** | **0.5859** | **0.9311** | **0.9111** |
| **XGBoost** | Config C (DO History Only) | **0.7574** | **0.9162** | **0.6285** | **0.7975** | **0.5186** | **0.9046** | **0.8924** |

### Confusion Matrix Breakdown (Temporal Test Set, 8,261 Observations):
- **Random Forest (Config B):**
  - True Positives (Crises Detected): **718** (76.1% recall)
  - False Negatives (Missed Crises): **225**
  - False Positives (False Alarms): **643** (91.2% specificity)
  - True Negatives (Safe Operations): **6,675**
- **XGBoost (Config B):**
  - True Positives (Crises Detected): **748** (79.3% recall)
  - False Negatives (Missed Crises): **195**
  - False Positives (False Alarms): **835** (88.6% specificity)
  - True Negatives (Safe Operations): **6,483**
- **Logistic Regression (Config B):**
  - True Positives: **848** (89.9% recall)
  - False Positives: **2,158** (Excessive false alarm rate; 70.5% specificity)

---

## 7. Secondary Evaluation: Unseen-Pond Generalization

To test whether models learn generalized physical dynamics rather than memorizing individual pond idiosyncrasies, a 5-Fold GroupKFold cross-validation was conducted across the 17 ponds ([`results/reports/group_kfold_performance.csv`](file:///d:/FISH/results/reports/group_kfold_performance.csv)):

| Model | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 | Mean Recall | Mean Precision |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.5950 | ± 0.1175 | 0.8941 ± 0.0304 | 0.4589 ± 0.0720 | 0.8662 ± 0.0677 | 0.3167 ± 0.0718 |
| **Random Forest** | **0.7086** | **± 0.0456** | **0.9069 ± 0.0313** | **0.6326 ± 0.0426** | **0.7076 ± 0.0983** | **0.5867 ± 0.0832** |
| **XGBoost** | **0.7183** | **± 0.0467** | **0.9041 ± 0.0285** | **0.6028 ± 0.0533** | **0.7379 ± 0.0983** | **0.5262 ± 0.1019** |

### Insights on Unseen Ponds:
- Both Random Forest and XGBoost demonstrated stable cross-pond generalization, achieving average PR-AUCs exceeding **0.70** with a low standard deviation (< 0.047).
- In Fold 5 (testing on ponds `ara2_0677080b`, `ara2_29660d32`, `ara2_4f79fc8d`, `ara2_d52ddb31`), XGBoost achieved a PR-AUC of **0.7925**, indicating strong transferability across earthen ponds.

---

## 8. Visualizations Reference

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

## 9. Main Scientific Observations

1. **Does History Provide Meaningful Information Beyond Current DO?**
   **Yes, unequivocally.**
   - In Logistic Regression, PR-AUC improved from 0.6019 (Config A) to 0.6763 (Config B).
   - In Random Forest, PR-AUC improved from 0.7107 (Config A) to 0.7420 (Config B) and 0.7471 (Config C).
   - In XGBoost, PR-AUC reached 0.7574 on Config C.
   - The rate of change in dissolved oxygen over the past 2 hours allows tree models to distinguish a stable pond at 3.5 mg/L from a rapidly deteriorating pond at 3.5 mg/L.
2. **Why Accuracy is Deceptive:**
   The Majority Baseline achieved 88.58% accuracy while failing to detect a single crisis. Evaluating early-warning systems by accuracy would result in deploying completely non-functional models.
3. **Precision vs. Recall Operational Trade-off:**
   - Logistic Regression maximizes Recall (~90%) but produces over 2,100 false alarms (Precision 28%).
   - Random Forest strikes the most balanced operational profile: 76.1% Recall, 52.8% Precision, and 91.2% Specificity, with the lowest total false alarms (643).
   - XGBoost offers slightly higher Recall (79.3%) with modest Precision (47.3%).

---

## 10. Limitations

1. **Provisional Threshold:** 3.0 mg/L remains a provisional operational threshold. Biological hypoxia thresholds vary by fish species and life stage.
2. **Seasonal Window:** Data spans 9 winter weeks in southern India; performance during high-temperature summer or monsoon conditions remains unverified.
3. **Default Hyperparameters:** Models were evaluated using sensible baseline hyperparameters without exhaustive search.
4. **Research Prototype:** ShinerAI Phase 3 models are research prototypes, not yet deployed in automated field control loops.

---

## 11. Recommended Candidate Model for Phase 4

**Primary Recommendation: Random Forest (Config B / Config C)**
- **Empirical Evidence:**
  - Highest F1-score (**0.6233 on Config B, 0.6602 on Config C**).
  - High PR-AUC (**0.7420 on Config B, 0.7471 on Config C**).
  - Highest Specificity (**91.2% on Config B, 93.1% on Config C**), reducing false alarms and unnecessary farmer electricity overhead.
  - Robust cross-pond generalization (0.7086 ± 0.0456).
  - Fast, low-latency CPU inference ideal for edge deployment on low-cost microcontrollers or lightweight Raspberry Pi farm hubs.
- **Secondary Candidate:** **XGBoost (Config B / Config C)** offers an attractive alternative if a farm prioritizes higher Recall (79.3%–79.8%) over false alarm minimization.

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

## 13. What Has NOT Been Done Yet

In accordance with strict project boundaries:
- No web application, Flask/FastAPI backend, or REST API has been built.
- No frontend dashboard or mobile UI has been created.
- No IoT device or microcontroller hardware has been interfaced.
- No fish disease, pathogen, or mortality prediction has been attempted.
- No deep learning (LSTM/GRU) or SHAP interpretability packages have been added.
- **Phase 4 has NOT started.**

---

## 14. Phase 4 Starting Point

When Phase 4 is approved, the project will transition to **System Integration & Inference Service**:
1. Package the serialized candidate model (`models/random_forest.joblib` or `models/xgboost.joblib`) into a lightweight, modular inference engine.
2. Build an API endpoint (e.g. FastAPI / Flask) accepting real-time pond telemetry streams and returning risk probabilities and alert classifications.
3. Implement operational threshold tuning allowing farmers to toggle between "High Sensitivity" (high recall) and "Energy Saver" (high precision) modes.
