# ShinerAI: Feature Leakage & Column Exclusion Audit

**Audit Date:** October 3, 2026  
**Auditor:** Scientific Audit Agent  
**Dataset:** `data/processed/ml_ready_dataset.csv` ($N = 41,277$, Columns = 37)  

---

## 1. Executive Summary

A critical failure mode in machine learning for real-time sensor streams is feature leakage. Leakage occurs when:
1. Future sensor values ($t + \Delta t$) are accidentally included in feature matrices.
2. Information from test observations contaminates feature engineering or scaling.
3. Target labels or identifiers are exposed during model training.

This audit provides a comprehensive accounting of every single column in `data/processed/ml_ready_dataset.csv`, demonstrating that **all 37 columns are strictly accounted for, no future features exist, and feature transformers are leak-free**.

---

## 2. Feature Configuration Definitions

### Configuration A: Current Only (Snapshot Model) — 5 Features
Designed to test whether instantaneous sensor telemetry and time-of-day alone can predict imminent hypoxia:
- `current_do`: Dissolved oxygen at prediction time $t$ ($\ge 3.0\text{ mg/L}$).
- `current_ph`: Water pH at prediction time $t$.
- `current_temperature`: Water temperature (°C) at prediction time $t$.
- `hour_of_day`: Integer hour ($0–23$).
- `minute_of_day`: Minute of the day ($0–1439$).

### Configuration B: Current + Full History (Multi-Sensor Dynamic) — 29 Features
Comprehensive historical feature set evaluating 2 hours of past dynamics across all three sensors:
- 5 Snapshot features from Config A.
- 8 Dissolved Oxygen lags: `do_t_minus_15`, `do_t_minus_30`, ..., `do_t_minus_120`.
- 8 pH lags: `ph_t_minus_15`, `ph_t_minus_30`, ..., `ph_t_minus_120`.
- 8 Temperature lags: `temp_t_minus_15`, `temp_t_minus_30`, ..., `temp_t_minus_120`.

### Configuration C: DO History Only (Parsimonious Dynamic Model) — 11 Features
Selected production model configuration focusing exclusively on dissolved oxygen dynamics and time:
- `current_do`
- `hour_of_day`, `minute_of_day`
- 8 Dissolved Oxygen lags: `do_t_minus_15`, `do_t_minus_30`, ..., `do_t_minus_120`.

---

## 3. Comprehensive Column Inventory & Audit (37 Columns)

| Column Name | Data Type | Null Count | Audit Classification |
| :--- | :---: | :---: | :--- |
| `pond_id` | str | 0 | EXCLUDED: Non-Predictor Metadata / Target |
| `prediction_timestamp` | str | 0 | EXCLUDED: Non-Predictor Metadata / Target |
| `hour_of_day` | int64 | 0 | Predictor (Config A, Config B, Config C) |
| `minute_of_day` | int64 | 0 | Predictor (Config A, Config B, Config C) |
| `current_do` | float64 | 0 | Predictor (Config A, Config B, Config C) |
| `current_ph` | float64 | 0 | Predictor (Config A, Config B) |
| `current_temperature` | float64 | 0 | Predictor (Config A, Config B) |
| `do_t` | float64 | 0 | EXCLUDED: Redundant Raw Duplicate |
| `do_t_minus_15` | float64 | 0 | Predictor (Config B, Config C) |
| `do_t_minus_30` | float64 | 0 | Predictor (Config B, Config C) |
| `do_t_minus_45` | float64 | 0 | Predictor (Config B, Config C) |
| `do_t_minus_60` | float64 | 0 | Predictor (Config B, Config C) |
| `do_t_minus_75` | float64 | 0 | Predictor (Config B, Config C) |
| `do_t_minus_90` | float64 | 0 | Predictor (Config B, Config C) |
| `do_t_minus_105` | float64 | 0 | Predictor (Config B, Config C) |
| `do_t_minus_120` | float64 | 0 | Predictor (Config B, Config C) |
| `ph_t` | float64 | 0 | EXCLUDED: Redundant Raw Duplicate |
| `ph_t_minus_15` | float64 | 0 | Predictor (Config B) |
| `ph_t_minus_30` | float64 | 0 | Predictor (Config B) |
| `ph_t_minus_45` | float64 | 0 | Predictor (Config B) |
| `ph_t_minus_60` | float64 | 0 | Predictor (Config B) |
| `ph_t_minus_75` | float64 | 0 | Predictor (Config B) |
| `ph_t_minus_90` | float64 | 0 | Predictor (Config B) |
| `ph_t_minus_105` | float64 | 0 | Predictor (Config B) |
| `ph_t_minus_120` | float64 | 0 | Predictor (Config B) |
| `temp_t` | float64 | 0 | EXCLUDED: Redundant Raw Duplicate |
| `temp_t_minus_15` | float64 | 0 | Predictor (Config B) |
| `temp_t_minus_30` | float64 | 0 | Predictor (Config B) |
| `temp_t_minus_45` | float64 | 0 | Predictor (Config B) |
| `temp_t_minus_60` | float64 | 0 | Predictor (Config B) |
| `temp_t_minus_75` | float64 | 0 | Predictor (Config B) |
| `temp_t_minus_90` | float64 | 0 | Predictor (Config B) |
| `temp_t_minus_105` | float64 | 0 | Predictor (Config B) |
| `temp_t_minus_120` | float64 | 0 | Predictor (Config B) |
| `target` | int64 | 0 | EXCLUDED: Non-Predictor Metadata / Target |
| `target_name` | str | 0 | EXCLUDED: Non-Predictor Metadata / Target |
| `data_quality_status` | str | 0 | EXCLUDED: Non-Predictor Metadata / Target |

---

## 4. Audit of Excluded Columns

Eight columns present in the processed dataset are **strictly excluded from all model training and inference pipelines**:

1. **Redundant Duplicate Columns (3):**
   - `do_t`, `ph_t`, `temp_t`: Raw unnormalized telemetry at time $t$. These are identical to `current_do`, `current_ph`, and `current_temperature`. They are excluded to prevent multicollinearity and duplicate column warnings.

2. **Metadata Columns (2):**
   - `pond_id`: Categorical pond identifier. Excluded to force models to learn universal physical dynamics rather than memorizing individual pond biases.
   - `prediction_timestamp`: Datetime string of observation. Used strictly for temporal ordering and purge calculations; never passed as a numerical predictor.

3. **Ground-Truth Target Columns (2):**
   - `target`: Binary ground truth ($0 = \text{SAFE}, 1 = \text{AT_RISK}$). Excluded from feature matrices $X$ and used solely as target vector $y$.
   - `target_name`: Human-readable label string (`'SAFE'`, `'AT_RISK'`). Excluded.

4. **Quality Assurance Column (1):**
   - `data_quality_status`: Processing flag (`'VALID'`). Excluded.

---

## 5. Preprocessing & Scaler Leakage Verification

In `src/model_utils.py`, `src/train.py`, and `notebooks/ShinerAI_Complete_ML_Pipeline.ipynb`:
- For linear models (Logistic Regression), `StandardScaler` is wrapped inside an explicit `sklearn.pipeline.Pipeline`:
  ```python
  Pipeline([
      ("scaler", StandardScaler()),
      ("clf", LogisticRegression(...))
  ])
  ```
- The scaler fits strictly on the training partition (`X_train`) during `fit()`, and computes mean $\mu_{\text{train}}$ and variance $\sigma_{\text{train}}^2$.
- Test features (`X_test`) are scaled strictly via `transform()` using training parameters.
- For tree-based models (Random Forest, XGBoost), raw feature values are used directly without global scaling, eliminating scaler contamination.

---

## 6. Audit Conclusion

The feature engineering and data ingestion pipelines are **100% leak-free**. No future sensor values exist in the feature set, excluded columns are strictly isolated, and transformers are encapsulated inside scikit-learn pipelines.
