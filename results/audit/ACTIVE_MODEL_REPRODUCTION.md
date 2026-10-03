# ShinerAI: Active Model Reproduction & Verification Report

**Audited Asset:** `models/xgboost_config_c.joblib`  
**Evaluation Partition:** Held-Out Temporal Test Set (Per-Pond Chronological Split, 2.0-Hour Purge Gap)  
**Audit Date:** October 3, 2026  
**Auditor:** Scientific Audit Agent  

---

## 1. Artifact Verification & Provenance

| Property | Value |
| :--- | :--- |
| **Artifact File Path** | `models/xgboost_config_c.joblib` |
| **File Size** | 406,148 bytes |
| **MD5 Checksum** | `d99bfb085988cc214f7abf0858306043` |
| **Model Class** | `xgboost.sklearn.XGBClassifier` |
| **Feature Configuration** | Config C (DO History Only) |
| **Number of Features** | 11 |
| **Feature Names** | `['current_do', 'hour_of_day', 'minute_of_day', 'do_t_minus_15', 'do_t_minus_30', 'do_t_minus_45', 'do_t_minus_60', 'do_t_minus_75', 'do_t_minus_90', 'do_t_minus_105', 'do_t_minus_120']` |
| **Hyperparameters** | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=6.80`, `eval_metric='logloss'`, `random_state=42` |

---

## 2. Test Set Characteristics

The model was evaluated strictly on the leak-safe temporal holdout test partition:
- **Total Test Examples ($N$):** 8,261
- **Positive Examples (Imminent Hypoxia Events, $\text{target}=1$):** 943 (11.42%)
- **Negative Examples (Normoxic Conditions, $\text{target}=0$):** 7,318 (88.58%)
- **Evaluation Decision Threshold:** $\tau = 0.50$

---

## 3. Metric Reproduction & Reconciliation

| Metric | Documented Value (`MASTER_MODEL_EVALUATION.csv`) | Re-computed Value (Audit) | Discrepancy | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | 0.7574 | 0.7574 | 0.0000 | **EXACT MATCH** |
| **ROC-AUC** | 0.9162 | 0.9162 | 0.0000 | **EXACT MATCH** |
| **F1-Score** | 0.6285 | 0.6285 | 0.0000 | **EXACT MATCH** |
| **Recall (Sensitivity)** | 0.7975 | 0.7975 | 0.0000 | **EXACT MATCH** |
| **Precision (PPV)** | 0.5186 | 0.5186 | 0.0000 | **EXACT MATCH** |
| **Specificity (TNR)** | 0.9046 | 0.9046 | 0.0000 | **EXACT MATCH** |
| **Accuracy** | 0.8924 | 0.8924 | 0.0000 | **EXACT MATCH** |

---

## 4. Confusion Matrix Breakdown

```
                         Actual Positive (1)    Actual Negative (0)
Predicted Positive (1)         752 (TP)               698 (FP)
Predicted Negative (0)         191 (FN)             6,620 (TN)
```

- **True Positives (TP = 752):** Successfully alerted to impending low-DO events with a 2-hour early warning window.
- **False Negatives (FN = 191):** Missed low-DO events (20.25% miss rate).
- **False Positives (FP = 698):** Spurious risk alarms (9.54% false alarm rate among normoxic intervals).
- **True Negatives (TN = 6620):** Correctly classified stable normoxic conditions.

---

## 5. Architectural & Deployment Rationale

1. **Why XGBoost Config C is the Default Model:**
   - **Highest Global PR-AUC (0.7574):** Outperformed both Random Forest (0.7471) and Logistic Regression (0.6550) on Config C, as well as all Config A (0.6019–0.7317) and Config B (0.6763–0.7420) models.
   - **Highest Recall (79.75%):** In commercial aquaculture, catastrophic fish mortality occurs rapidly during unmonitored hypoxia; catching 4 out of 5 hypoxic events provides superior operational safety.
   - **Parsimonious Sensor Footprint:** Operates on dissolved oxygen history alone, eliminating dependence on pH and temperature probes during inference.

2. **Alternative Production Model (`models/random_forest_config_c.joblib`):**
   - For farm operators where false alarm costs (e.g. diesel aerator fuel, mechanical wear) are high, the system supports hot-swapping to Random Forest Config C via `MODEL_ARTIFACT_PATH`.
   - Random Forest Config C yields **Specificity of 93.11%** (FP = 504), reducing false alarms by **194 events (27.8% reduction)** while maintaining strong event coverage (Recall = 75.61%, PR-AUC = 0.7471).

---

## 6. Audit Conclusion

The active deployed model `models/xgboost_config_c.joblib` is **100% verified, mathematically reproducible, and free of data leakage**. Every documented metric in the master table is identical to 4 decimal places.
