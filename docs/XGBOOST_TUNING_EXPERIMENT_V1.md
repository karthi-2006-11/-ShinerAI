# ShinerAI: Experimental XGBoost Tuning and Feature Engineering (V1)
**Research Branch**: `experiment/xgb-tuning-v1`  
**Date**: October 2026  
**Status**: COMPLETED — EXPERIMENTAL RESEARCH ARCHIVE (NEGATIVE RESULT / BASELINE RETAINED)

---

## 1. Executive Summary & Objective

The objective of this experimental campaign was to determine whether the frozen production baseline model (**XGBoost Config C**, 11 features, decision threshold $\tau = 0.50$) could be genuinely improved in **PR-AUC**, **F1**, and **Recall** through systematic hyperparameter optimization and leakage-free temporal feature engineering, without compromising temporal validity, leakage isolation, or real-world early warning utility.

### Authoritative Scientific Conclusion
> **DECISION: KEEP BASELINE**  
> While hyperparameter tuning combined with temporal acceleration features achieved a slight apparent gain on 5-fold training cross-validation ($\text{PR-AUC}_{\text{val}} = 0.7443$ vs. $0.7352$), it **failed to generalize on the locked temporal test set** ($\text{PR-AUC}_{\text{test}} = 0.7544$ vs. $0.7574$, a decrease of $-0.0030$). Furthermore, maintaining the operational recall constraint ($\ge 78\%$) required lowering the decision threshold to $0.34$, which generated **+91 additional false positives** ($789$ vs. $698$) and **missed 4 additional real-world physical hypoxia episodes** ($120/136$ vs. $124/136$ detected).  
> The existing frozen production model (`models/xgboost_config_c.joblib`) demonstrates superior generalization, higher test precision, lower operational alarm fatigue, and superior event detection. **The production baseline is retained unchanged.**

---

## 2. Frozen Production Baseline Reference

The production model remained strictly locked and was never modified or overwritten:
- **Model Artifact**: `models/xgboost_config_c.joblib`
- **Artifact SHA256**: `57eb3cf72259d149028129d24920b97a7f593ed31f9708d2a1d436351f157475` (Verified unchanged)
- **Features (11)**:
  `current_do`, `hour_of_day`, `minute_of_day`, `do_t_minus_15`, `do_t_minus_30`, `do_t_minus_45`, `do_t_minus_60`, `do_t_minus_75`, `do_t_minus_90`, `do_t_minus_105`, `do_t_minus_120`
- **Operational DO Boundary**: $3.0\text{ mg/L}$ (hypoxia early warning boundary)
- **Forecast Horizon**: $2\text{ hours}$
- **Operational Decision Threshold**: $\tau = 0.50$

### Exact Baseline Hyperparameters
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': 6.799952595401754,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
    'min_child_weight': 1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'gamma': 0.0,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0
}
```

### Authoritative Baseline Held-Out Test Metrics ($N = 8,261$, 2h Purge Gap)
- **PR-AUC**: `0.7574`
- **ROC-AUC**: `0.9162`
- **F1**: `0.6285`
- **Recall**: `0.7975` ($752 / 943$)
- **Precision**: `0.5186` ($752 / 1,450$)
- **Specificity**: `0.9046` ($6,620 / 7,318$)
- **Accuracy**: `0.8924` ($7,372 / 8,261$)
- **Confusion Matrix**: $\text{TN}=6620, \text{FP}=698, \text{FN}=191, \text{TP}=752$
- **Event-Level Detection**: $124 / 136$ physical episodes ($91.18\%$), Mean Physical Lead Time = $93.1\text{ min}$

---

## 3. Strict Scientific Validation Protocol

To guarantee total scientific honesty and prevent data leakage:
1. **Absolute Test-Set Lock**: The temporal test set ($N = 8,261$, $943$ AT_RISK, $7,318$ SAFE) was isolated prior to all tuning. Zero test observations entered any hyperparameter search, feature selection, threshold tuning, or early stopping.
2. **Training-Only 5-Fold GroupKFold**: Cross-validation was executed strictly across the $32,908$ training observations grouped by `pond_id` ($17$ unique ponds). For every fold, validation ponds were strictly disjoint from training ponds ($\text{train\_ponds} \cap \text{val\_ponds} = \emptyset$).
3. **Deterministic Seeds**: Random seeds fixed at $42$. All evaluations are $100\%$ reproducible.

---

## 4. Hyperparameter Search Space & Experimental Design

A structured set of **64 experimental trials** was evaluated across 5 folds ($320$ total model fits), spanning:
- **Tree Depth (`max_depth`)**: $3, 4, 5, 6, 7$
- **Learning Rate (`learning_rate`)**: $0.03, 0.05, 0.08, 0.10$
- **Number of Estimators (`n_estimators`)**: $100, 150, 200, 250$
- **Minimum Child Weight (`min_child_weight`)**: $1, 3, 5, 7$
- **Subsampling Ratio (`subsample`)**: $0.7, 0.8, 1.0$
- **Feature Subsampling (`colsample_bytree`)**: $0.7, 0.8, 1.0$
- **Split Regularization (`gamma`)**: $0.0, 0.1, 0.3, 0.5$
- **L1 Regularization (`reg_alpha`)**: $0.0, 0.05, 0.1, 0.5$
- **L2 Regularization (`reg_lambda`)**: $0.5, 1.0, 2.0$
- **Class Balancing (`scale_pos_weight`)**: $5.44, 6.80, 8.16$ ($0.8\times, 1.0\times, 1.2\times$ empirical imbalance)

All 64 trials were recorded in [`results/reports/xgb_tuning_v1_results.csv`](file:///d:/Shiner_AI/results/reports/xgb_tuning_v1_results.csv).

---

## 5. Feature Engineering & Ablation Analysis

We evaluated candidate row-wise temporal features derived strictly from current and past observations at prediction time $T$:
1. **Baseline Config C (11 features)**: Current DO, time of day, and 8 historical lags ($t-15\text{m}$ to $t-120\text{m}$).
2. **Config C + Trends (15 features)**: Added rate-of-change deltas `do_change_15`, `do_change_30`, `do_change_60`, `do_change_120`.
3. **Config C + Trends + Acceleration (16 features)**: Added second-order derivative `do_accel_15 = (do_t - do_{t-15}) - (do_{t-15} - do_{t-30})`.
4. **Config C + Trends + Acceleration + Window Stats (21 features)**: Added rolling 2-hour minimum, maximum, mean, standard deviation, and range.

### Feature Ablation Comparison (5-Fold CV on Training Set)
| Configuration | Num Features | Mean Val PR-AUC | Std Val PR-AUC | Mean Val F1 | Mean Val Recall | Mean Val Precision |
|---|---:|---:|---:|---:|---:|---:|
| **Config C Baseline** | 11 | 0.7352 | $\pm$ 0.0474 | 0.6168 | 0.7788 | 0.5180 |
| **Config C + Trends** | 15 | 0.7417 | $\pm$ 0.0470 | 0.6227 | 0.7793 | 0.5261 |
| **Config C + Accel** | 16 | **0.7443** | $\pm$ 0.0465 | **0.6573** | 0.7268 | **0.6052** |
| **Config C + All Stats**| 21 | 0.7387 | $\pm$ 0.0487 | 0.6161 | 0.7784 | 0.5168 |

*Observation*: Adding rate of change and acceleration slightly improved cross-validation PR-AUC and precision, whereas adding redundant window statistics (21 features) degraded PR-AUC due to colinearity with raw lag inputs.

---

## 6. Winning Validation Candidate Selection

Based strictly on validation PR-AUC, the top candidate across all 64 trials was:
- **Trial ID**: `T_accel_grid_059`
- **Feature Set**: `config_c_accel` (16 features)
- **Hyperparameters**:
  ```python
  {
      'n_estimators': 250,
      'max_depth': 7,
      'learning_rate': 0.05,
      'min_child_weight': 1,
      'subsample': 0.8,
      'colsample_bytree': 1.0,
      'gamma': 0.1,
      'reg_alpha': 0.05,
      'reg_lambda': 0.5,
      'scale_pos_weight': 5.439962076321404,
      'eval_metric': 'logloss',
      'random_state': 42,
      'n_jobs': -1
  }
  ```
- **Validation Metrics**:
  - Mean Val PR-AUC = `0.7443` ($+0.0091$ over baseline)
  - Mean Val F1 = `0.6573` ($+0.0405$ over baseline)
  - Mean Val Recall = `0.7268` ($-0.0520$ vs. baseline $0.7788$)

---

## 7. Threshold Optimization on Validation OOF Probabilities

Evaluating decision thresholds $\tau \in [0.10, 0.90]$ across out-of-fold validation probabilities:
- **F1-Max Threshold**: $\tau = 0.68$ ($\text{F1} = 0.6832$), but Recall drops to $0.6449$ (unacceptable for farm safety).
- **Default Operational Threshold**: $\tau = 0.50$ ($\text{F1} = 0.6622$, $\text{Recall} = 0.7198$).
- **Recall-Constrained Threshold ($\text{Recall} \ge 78\%$)**: $\tau = 0.34$
  - Out-of-fold Validation Recall = $0.7836$ ($\ge 78\%$)
  - Out-of-fold Validation F1 = $0.6124$
  - Out-of-fold Validation Precision = $0.5027$

*Selected Operational Decision Threshold*: $\tau = 0.34$

---

## 8. Calibration Analysis

- **Uncalibrated Validation Brier Score**: `0.0725`
- As PR-AUC is threshold- and rank-invariant under monotonic calibration, Platt scaling and isotonic regression do not alter the fundamental ranking or test-set PR-AUC. The uncalibrated score of $0.0725$ indicates well-bounded probability estimations.

---

## 9. Untouched Test Set Evaluation (Evaluated Strictly ONCE)

The winning candidate was trained on the full $32,908$ training set and evaluated on the untouched $8,261$ test set:

### Master Comparison Table: Frozen Baseline vs. Experimental Model

| Metric | Frozen Baseline ($\tau = 0.50$) | Experimental ($\tau = 0.34$, recall-matched) | Experimental ($\tau = 0.50$, default) | Scientific Change (Recall-Matched) |
|---|---:|---:|---:|---:|
| **PR-AUC** | **0.7574** | 0.7544 | 0.7544 | $-0.0030$ (Regression) |
| **ROC-AUC** | **0.9162** | 0.9136 | 0.9136 | $-0.0026$ (Regression) |
| **F1** | **0.6285** | 0.6032 | **0.6667** | $-0.0253$ (Trade-off) |
| **Recall** | **0.7975** | 0.7932 | 0.7402 | $-0.0043$ ($-0.0573$ at $\tau=0.50$) |
| **Precision** | **0.5186** | 0.4867 | **0.6064** | $-0.0319$ ($+0.0878$ at $\tau=0.50$) |
| **Specificity**| **0.9046** | 0.8922 | **0.9381** | $-0.0124$ ($+0.0335$ at $\tau=0.50$) |
| **Accuracy** | **0.8924** | 0.8809 | **0.9155** | $-0.0115$ ($+0.0231$ at $\tau=0.50$) |

### Test Confusion Matrix Breakdown
- **Frozen Baseline ($\tau = 0.50$)**:
  - $\text{TP} = 752$
  - $\text{FP} = 698$
  - $\text{TN} = 6,620$
  - $\text{FN} = 191$
- **Experimental Model ($\tau = 0.34$, Recall-Matched)**:
  - $\text{TP} = 748$ ($-4$ detected)
  - $\text{FP} = 789$ (**$+91$ additional false alarms!**)
  - $\text{TN} = 6,529$
  - $\text{FN} = 195$ ($+4$ missed points)
- **Experimental Model ($\tau = 0.50$)**:
  - $\text{TP} = 698$ (**$-54$ fewer detected points!**)
  - $\text{FP} = 453$
  - $\text{TN} = 6,865$
  - $\text{FN} = 245$ (**$+54$ missed points!**)

---

## 10. Operational Event-Level Evaluation (136 Physical Episodes)

Evaluating warnings across the 136 impending hypoxia episodes in the test set:
- **Frozen Baseline Detection**: **$124 / 136$ episodes ($91.18\%$)** | Mean Lead Time = **$93.1\text{ min}$**
- **Experimental Model Detection**: **$120 / 136$ episodes ($88.24\%$)** | Mean Lead Time = **$94.3\text{ min}$** | Median = $120.0\text{ min}$

*Finding*: The experimental model **missed 4 additional real-world hypoxia episodes** ($16$ missed vs. $12$ missed in baseline).

---

## 11. Error & Diagnostic Analysis

1. **Why did validation show improvement while test degraded?**  
   Tree depth $7$ with $250$ estimators slightly overfitted the specific pond temporal dynamics within the training set. When deployed to unseen future test timepoints, the deeper trees were less resilient than the baseline shallower trees (depth 6, 100 trees).
2. **Operational Trade-off Dilemma**:
   - At $\tau = 0.50$, the experimental model achieves higher precision ($60.64\%$ vs. $51.86\%$) and higher F1 ($0.6667$ vs. $0.6285$), but **recall drops by $5.73\%$ percentage points** ($74.02\%$ vs. $79.75\%$), causing $54$ additional low-DO intervals and $4$ entire fish-kill episodes to be missed. In aquaculture, missing an anoxic collapse is catastrophic.
   - At $\tau = 0.34$ (calibrated to preserve safety recall), false alarms increase by $+13.0\%$ ($789$ vs. $698$), degrading user trust without providing any PR-AUC gain.

---

## 12. SHAP Feature Attribution Analysis

SHAP `TreeExplainer` attribution ranking on the experimental model:
1. `current_do`: Mean $|\text{SHAP}| = 1.3918$ (dominates risk assessment)
2. `minute_of_day`: Mean $|\text{SHAP}| = 0.5433$ (captures diurnal photosynthetic cycle)
3. `do_t_minus_15`: Mean $|\text{SHAP}| = 0.2702$ (short-term lag)
4. `do_t_minus_120`: Mean $|\text{SHAP}| = 0.2369$ (2-hour lag reference)
5. `do_change_120`: Mean $|\text{SHAP}| = 0.2185$ (overall 2-hour trajectory slope)
6. `do_accel_15`: Mean $|\text{SHAP}| = 0.1642$ (rate of drop acceleration)

*Verification*: The engineered features acted as legitimate supplementary signals without pathology, but did not overcome the structural predictability limit of 2-hour ahead hypoxia.

---

## 13. Artifact & Safety Verification

- **Production Model Artifact**: `models/xgboost_config_c.joblib` — **STRICTLY UNCHANGED** (SHA256: `57eb3cf72259d149028129d24920b97a7f593ed31f9708d2a1d436351f157475`)
- **Experimental Model Artifact**: `models/experimental_xgboost_v1.joblib` (Saved separately as experimental research artifact)
- **Dataset**: `data/processed/ml_ready_dataset.csv` — **STRICTLY UNCHANGED** (SHA256: `887a6382992daa68a32f9182d111ded2c0120543af80c8700d24e1bd6556ef2f`)
- **Test Integrity**: Locked until final single evaluation; no leakage detected.
- **Git Branch**: `experiment/xgb-tuning-v1` (Isolated from `main`).
