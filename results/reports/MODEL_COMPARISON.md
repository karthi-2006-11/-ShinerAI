# ShinerAI: Consolidated Model Comparison (Internal Temporal Holdout)

**Dataset:** Commercial Golden Shiner Aquaculture Dataset (17 Earthen Ponds, Lonoke County, AR, USA)  
**Evaluation Scheme:** Chronological 80/20 Holdout with a Strict 2.0-Hour Purge Gap ($N = 8,261$ test observations)  
**Positive Prevalence:** 11.42% (943 low-DO events $< 3.0$ mg/L)  
**Decision Threshold:** $\tau = 0.50$ (applied across all classifiers)  

| Model | Feature Configuration | PR-AUC | ROC-AUC | F1 | Recall | Precision | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Majority Baseline | None (Class Distribution Only) | **0.1142** | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8858 |
| Current-DO Baseline (DO <= 4.2) | Current DO Only (Static Threshold <= 4.2 mg/L) | **0.6149** | 0.9024 | 0.4516 | 0.8961 | 0.3019 | 0.7330 | 0.7516 |
| Logistic Regression | Config A — Current DO + Time (5 Predictors) | **0.6019** | 0.8994 | 0.4274 | 0.9003 | 0.2802 | 0.7020 | 0.7246 |
| Logistic Regression | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.6763** | 0.9078 | 0.4295 | 0.8993 | 0.2821 | 0.7051 | 0.7273 |
| Logistic Regression | Config C — Current + Recent DO History (11 Predictors) | **0.6550** | 0.9093 | 0.4549 | 0.8940 | 0.3051 | 0.7376 | 0.7555 |
| Random Forest | Config A — Current DO + Time (5 Predictors) | **0.7107** | 0.9116 | 0.6174 | 0.7709 | 0.5149 | 0.9064 | 0.8909 |
| Random Forest | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.7420** | 0.9169 | 0.6233 | 0.7614 | 0.5276 | 0.9121 | 0.8949 |
| Random Forest | Config C — Current + Recent DO History (11 Predictors) | **0.7471** | 0.9144 | 0.6602 | 0.7561 | 0.5859 | 0.9311 | 0.9111 |
| XGBoost | Config A — Current DO + Time (5 Predictors) | **0.7317** | 0.9150 | 0.5817 | 0.8102 | 0.4537 | 0.8743 | 0.8670 |
| XGBoost | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.7353** | 0.9171 | 0.5922 | 0.7932 | 0.4725 | 0.8859 | 0.8753 |
| XGBoost | Config C — Current + Recent DO History (11 Predictors) | **0.7574** | 0.9162 | 0.6285 | 0.7975 | 0.5186 | 0.9046 | 0.8924 |

> **Note:** All values reflect exact verified metrics from `results/reports/MASTER_MODEL_EVALUATION.csv`. XGBoost Config C serves as the production frozen early warning model.