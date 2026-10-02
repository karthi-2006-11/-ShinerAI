# ShinerAI: Model Explainability Report (Phase 4)
**Project:** ShinerAI  
**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Scope:** Phase 4 — Explainability Analysis for Config C Tree-Based Models  
**Evaluated Models:** XGBoost Config C and Random Forest Config C  

---

## 1. Executive Summary & Methodology

This analysis interprets the decision behavior of the two leading tree-based models on **Config C (DO History Only, 11 features)** using **SHAP (SHapley Additive exPlanations) TreeExplainer**:
- **Global Feature Importance:** Measures the mean absolute contribution of each feature across 1,000 held-out test set observations.
- **Local Explanations:** Deconstructs specific individual predictions into positive contributions (pushing risk toward `AT_RISK`) and negative contributions (pushing risk toward `SAFE`).
- **Representative Case Studies:** Examines one daytime rising DO event (SAFE) and one nighttime declining DO event (AT_RISK).

> [!NOTE]
> **Model Artifact Provenance:**  
> Config C model artifacts were reproduced using the frozen Phase 3 training procedure solely to create dedicated explainability/API artifacts; no model architecture, dataset, split, feature set, or training procedure was changed.

> [!IMPORTANT]
> **Scientific & Causal Guardrail:**  
> Feature importance and SHAP values quantify **statistical association within this model and dataset**; they do **not prove biological causality**. Historical inputs are discrete lag observations ($t-15\text{m} \dots t-120\text{m}$); no explicit rate-of-change or derivative feature was engineered. ShinerAI forecasts water oxygen depletion events ($	ext{DO} < 3.0\text{ mg/L}$ within 2 hours); it does not directly predict fish mortality or disease.

---

## 2. Global Feature Importance Ranking

### XGBoost Config C (11 Features, Evaluated on Held-Out Test Set):
| Rank | Feature | Mean |SHAP Value| Role in Model Decisions |
|:---:|:---|:---:|:---|
| 1 | `current_do` | 1.3896 | Primary baseline level at prediction time T |
| 2 | `minute_of_day` | 0.5388 | Encodes time-of-day patterns observed in the dataset |
| 3 | `hour_of_day` | 0.2549 | Historical trajectory predictor |
| 4 | `do_t_minus_30` | 0.2388 | Historical trajectory predictor |
| 5 | `do_t_minus_15` | 0.2385 | Represents DO 15 min prior; contributes to trajectory |
| 6 | `do_t_minus_120` | 0.1562 | Historical trajectory predictor |
| 7 | `do_t_minus_45` | 0.1555 | Historical trajectory predictor |
| 8 | `do_t_minus_90` | 0.1237 | Historical trajectory predictor |
| 9 | `do_t_minus_105` | 0.1117 | Historical trajectory predictor |
| 10 | `do_t_minus_60` | 0.1058 | Historical trajectory predictor |
| 11 | `do_t_minus_75` | 0.0986 | Historical trajectory predictor |

### Random Forest Config C (11 Features, Evaluated on Held-Out Test Set):
| Rank | Feature | Mean |SHAP Value| Role in Model Decisions |
|:---:|:---|:---:|:---|
| 1 | `current_do` | 0.1049 | Primary baseline level at prediction time T |
| 2 | `do_t_minus_15` | 0.0758 | Represents DO 15 min prior; contributes to trajectory |
| 3 | `minute_of_day` | 0.0485 | Encodes time-of-day patterns observed in the dataset |
| 4 | `do_t_minus_30` | 0.0434 | Historical trajectory predictor |
| 5 | `hour_of_day` | 0.0340 | Historical trajectory predictor |
| 6 | `do_t_minus_45` | 0.0317 | Historical trajectory predictor |
| 7 | `do_t_minus_60` | 0.0257 | Historical trajectory predictor |
| 8 | `do_t_minus_75` | 0.0212 | Historical trajectory predictor |
| 9 | `do_t_minus_105` | 0.0158 | Historical trajectory predictor |
| 10 | `do_t_minus_120` | 0.0155 | Historical trajectory predictor |
| 11 | `do_t_minus_90` | 0.0144 | Historical trajectory predictor |

---

## 3. Case Studies: Local Prediction Deconstructions

### SAFE Case Study — Rising DO During Daytime (Ground Truth: SAFE)
- **Pond:** `ara2_0677080b` | **Timestamp:** `2026-01-26 10:15:00`
- **Current DO:** `5.40 mg/L`
- **Model Output:** 
  - XGBoost: Risk Probability = `6.67%` (`SAFE`)
  - Random Forest: Risk Probability = `5.17%` (`SAFE`)
- **Interpretation:**  
  Although DO had been at 2.92 mg/L 2 hours prior, the subsequent lags increased monotonically (3.23 -> 3.72 -> 4.21 -> 5.40 mg/L) alongside daytime hours (`hour_of_day = 10`). The model recognized this ascending trajectory and correctly drove risk contributions strongly negative (toward `SAFE`).

### AT_RISK Case Study — Declining DO During Nighttime (Ground Truth: AT_RISK)
- **Pond:** `ara2_0677080b` | **Timestamp:** `2026-01-26 03:30:00`
- **Current DO:** `3.84 mg/L`
- **Model Output:** 
  - XGBoost: Risk Probability = `93.41%` (`AT_RISK`)
  - Random Forest: Risk Probability = `91.67%` (`AT_RISK`)
- **Interpretation:**  
  Although current DO remained above the 3.0 threshold at 3.84 mg/L, the recent trajectory showed a steady drop from 5.02 mg/L over 2 hours during nighttime hours (`hour_of_day = 3`). The model recognized the steep decline and drove positive SHAP contributions strongly toward `AT_RISK`, providing the critical 2-hour early warning.

---

## 4. Summary of Saved Visualizations

1. [`results/figures/explainability/global_feature_importance_xgb_config_c.png`](file:///d:/FISH/results/figures/explainability/global_feature_importance_xgb_config_c.png)
2. [`results/figures/explainability/global_feature_importance_rf_config_c.png`](file:///d:/FISH/results/figures/explainability/global_feature_importance_rf_config_c.png)
3. [`results/figures/explainability/local_explanation_safe_xgb_config_c.png`](file:///d:/FISH/results/figures/explainability/local_explanation_safe_xgb_config_c.png)
4. [`results/figures/explainability/local_explanation_at_risk_xgb_config_c.png`](file:///d:/FISH/results/figures/explainability/local_explanation_at_risk_xgb_config_c.png)
5. [`results/figures/explainability/local_explanation_safe_rf_config_c.png`](file:///d:/FISH/results/figures/explainability/local_explanation_safe_rf_config_c.png)
6. [`results/figures/explainability/local_explanation_at_risk_rf_config_c.png`](file:///d:/FISH/results/figures/explainability/local_explanation_at_risk_rf_config_c.png)
