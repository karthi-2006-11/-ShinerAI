"""
Updates README.md and PROJECT_DOCUMENTATION.md with:
1. Model Comparison & Feature Ablation Analysis (with table and chart)
2. Independent External Validation (Oman Nile Tilapia Dataset)
3. Updated project file structure and test count (97 tests)
"""

from pathlib import Path
import re

# ==============================================================================
# 1. Update README.md
# ==============================================================================
readme_path = Path("README.md")
readme_text = readme_path.read_text(encoding="utf-8")

# Fix old file:///d:/FISH/ links to relative links
readme_text = re.sub(r'file:///d:/FISH/', '', readme_text)

model_comparison_and_external_val_section = """
### Authoritative Active Model Benchmark (Production Standard)

The production model [`models/xgboost_config_c.joblib`](models/xgboost_config_c.joblib) is evaluated on the held-out temporal partition ($N = 8,261$, $943$ positive events) with a mandatory 2-hour purge gap:

| Metric | Verified Value | Benchmark Detail |
| :--- | :---: | :--- |
| **PR-AUC (Primary)** | **0.7574** | **+0.1425 lift** over Current-DO baseline (0.6149) |
| **ROC-AUC** | **0.9162** | High discriminatory capability across classification thresholds |
| **F1-Score** | **0.6285** | Balanced performance at default operational decision threshold $\\tau = 0.50$ |
| **Recall (Sensitivity)** | **0.7975** | Catches 79.75% of impending hypoxic events (752 of 943) |
| **Precision (PPV)** | **0.5186** | 752 true positives out of 1,450 total alerts |
| **Specificity (TNR)** | **0.9046** | Correctly rejects 90.46% of normoxic intervals (6,620 of 7,318) |
| **Accuracy** | **0.8924** | Overall classification accuracy on temporal holdout (7,372 of 8,261) |
| **Confusion Matrix** | **TP: 752, FP: 698, TN: 6,620, FN: 191** | Evaluated on 8,261 holdout observations with 2-hour purge gap |

---

## Model Comparison & Feature Ablation

To address mentor review feedback (*"Make a comparison so that it will be easy for me"*), ShinerAI provides a consolidated benchmark table evaluating all 11 model configurations and 2 baselines on the chronological temporal holdout ($N = 8,261$, 11.42% positive prevalence):

### Consolidated Model Comparison Table

| Model | Feature Configuration | PR-AUC | ROC-AUC | F1 | Recall | Precision | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|---|
| **Majority Baseline** | None (Class Distribution Only) | **0.1142** | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8858 |
| **Current-DO Baseline** | Current DO Only (Static Threshold $\\le 4.2$ mg/L) | **0.6149** | 0.9024 | 0.4516 | 0.8961 | 0.3019 | 0.7330 | 0.7516 |
| **Logistic Regression** | Config A — Current DO + Time (5 Predictors) | **0.6019** | 0.8994 | 0.4274 | 0.9003 | 0.2802 | 0.7020 | 0.7246 |
| **Logistic Regression** | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.6763** | 0.9078 | 0.4295 | 0.8993 | 0.2821 | 0.7051 | 0.7273 |
| **Logistic Regression** | Config C — Current + Recent DO History (11 Predictors) | **0.6550** | 0.9093 | 0.4549 | 0.8940 | 0.3051 | 0.7376 | 0.7555 |
| **Random Forest** | Config A — Current DO + Time (5 Predictors) | **0.7107** | 0.9116 | 0.6174 | 0.7709 | 0.5149 | 0.9064 | 0.8909 |
| **Random Forest** | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.7420** | 0.9169 | 0.6233 | 0.7614 | 0.5276 | 0.9121 | 0.8949 |
| **Random Forest** | Config C — Current + Recent DO History (11 Predictors) | **0.7471** | 0.9144 | 0.6602 | 0.7561 | 0.5859 | 0.9311 | 0.9111 |
| **XGBoost** | Config A — Current DO + Time (5 Predictors) | **0.7317** | 0.9150 | 0.5817 | 0.8102 | 0.4537 | 0.8743 | 0.8670 |
| **XGBoost** | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.7353** | 0.9171 | 0.5922 | 0.7932 | 0.4725 | 0.8859 | 0.8753 |
| **XGBoost (Champion)** | **Config C — Current + Recent DO History (11 Predictors)** | **0.7574** | **0.9162** | **0.6285** | **0.7975** | **0.5186** | **0.9046** | **0.8924** |

> *Authoritative source artifact:* [`results/reports/MODEL_COMPARISON.csv`](results/reports/MODEL_COMPARISON.csv) | [`results/reports/MODEL_COMPARISON.md`](results/reports/MODEL_COMPARISON.md)

![Model Comparison PR-AUC](results/figures/model_comparison_prauc.png)

### Feature Ablation: Does Recent DO History Add Predictive Value?

To isolate the predictive contribution of temporal context versus static telemetry, we evaluate three sensor configurations:
- **Config A (5 features):** Current values (`current_do`, `current_ph`, `current_temperature`, `hour_of_day`, `minute_of_day`).
- **Config B (29 features):** Current values + 8 lags each of DO, pH, and Temperature ($t-15\\text{m}$ to $t-120\\text{m}$) + diurnal time.
- **Config C (11 features):** Current values + 8 lags of DO only ($t-15\\text{m}$ to $t-120\\text{m}$) + diurnal time.

| Model Architecture | Config A PR-AUC | Config B PR-AUC | Config C PR-AUC | Config B - Config A | Config C - Config A Lift | Config C - Config B |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.6019 | 0.6763 | 0.6550 | +0.0744 | **+0.0531** | -0.0213 |
| **Random Forest** | 0.7107 | 0.7420 | 0.7471 | +0.0313 | **+0.0364** | +0.0051 |
| **XGBoost** | 0.7317 | 0.7353 | 0.7574 | +0.0036 | **+0.0257** | +0.0221 |

**Key Ablation Insights:**
1. **Recent DO History Adds Defensible Predictive Lift:** For every model family, adding 2 hours of DO history yields substantial PR-AUC improvements over current-only features alone (+0.0531 for LR, +0.0364 for RF, +0.0257 for XGBoost).
2. **Model-Specific Sensor Trade-offs:** Config C does *not* strictly dominate Config B across all models. For linear Logistic Regression, auxiliary water quality sensors (pH, temperature) provide additional linear separability (+0.0744 vs +0.0531). For non-linear tree ensembles (RF and XGBoost), isolating historical DO avoids feature dilution and tree split fragmentation, achieving the highest overall PR-AUC (0.7471 and 0.7574).
3. **No Explicit Derivative Engineered:** All temporal inputs are discrete 15-minute sensor observations; no mathematical derivative or explicit velocity feature was computed.

> *Artifacts:* [`results/reports/FEATURE_ABLATION_COMPARISON.csv`](results/reports/FEATURE_ABLATION_COMPARISON.csv) | [`results/reports/FEATURE_ABLATION_COMPARISON.md`](results/reports/FEATURE_ABLATION_COMPARISON.md)

---

## Independent External Validation (Oman Nile Tilapia Dataset)

To directly answer mentor feedback (*"Make a Independent external validation for this work pa"*), the frozen production model (`models/xgboost_config_c.joblib`) was evaluated on an independent external aquaculture dataset published in 2026:

- **Dataset:** *Dissolved Oxygen Forecasting Dataset for Nile Tilapia Aquaculture in Oman*
- **Authors:** Ahmed M. Al-Khaldi, Ramadoss Dhandapani, Mohammed A. Al-Badri
- **Citation:** *Sensors* 2026, 26(13), 4242; DOI: [10.3390/s26134242](https://doi.org/10.3390/s26134242) (CC BY 4.0)
- **Repository:** [https://github.com/AhmedTheNetCoder/DO-Forecasting-Tilapia-Dataset](https://github.com/AhmedTheNetCoder/DO-Forecasting-Tilapia-Dataset)

### Strict Scientific Safeguards (Zero Retraining / Zero Adaptation):
1. **Model Strictly Frozen:** The model artifact `models/xgboost_config_c.joblib` was loaded directly. **Zero retraining, zero fitting, and zero fine-tuning** were performed.
2. **Fixed Decision Threshold:** The classification threshold was locked at $\\tau = 0.50$ (zero threshold search on external data).
3. **Leakage-Free 15-Minute Resampling:** Raw ~7-second IoT readings were resampled into non-overlapping 15-minute right-closed windows $(T-15\\text{m}, T]$.
4. **Identical Task:** Given current $\\text{DO} \\ge 3.0\\text{ mg/L}$ and 2 hours of DO history, forecast whether DO will fall below $3.0\\text{ mg/L}$ within the subsequent 2 hours.

### Internal vs. External Validation Benchmark Comparison

| Metric / Dimension | Internal FWI Temporal Holdout | External Oman Tilapia Validation |
|---|---|---|
| **Target Organism** | Golden Shiner (*Notemigonus crysoleucas*) | Nile Tilapia (*Oreochromis niloticus*) |
| **Geographic Region** | Lonoke County, Arkansas, USA (Humid Subtropical) | North Al Sharqiyah, Oman (Arid Desert) |
| **Facility Context** | Commercial production earthen ponds (17 ponds) | Controlled 180L recirculating tank with live tilapia |
| **Sensor Platform** | Continuous optical/photometer multiparameter sonde | Low-cost Gravity analog DO probe + ESP32 IoT |
| **Raw Sampling Rate** | 15-minute nominal intervals | ~5–7 second high-frequency readings |
| **Eligible Test Samples ($N$)** | **8,261** 15-minute intervals | **742** 15-minute intervals (8.74 continuous days) |
| **Positive Events (AT_RISK)** | **943** events (11.42% prevalence) | **0** events (0.00% prevalence; well-aerated tank) |
| **Negative Samples (SAFE)** | 7,318 samples | 742 samples |
| **Decision Threshold ($\\tau$)** | 0.50 (frozen) | 0.50 (frozen, zero adaptation) |
| **True Negatives (TN)** | 6,620 | **742** |
| **False Positives (FP)** | 698 | **0** |
| **True Positives (TP)** | 752 | **0** |
| **False Negatives (FN)** | 191 | **0** |
| **Specificity (TNR)** | **0.9046 (90.46%)** | **1.0000 (100.0%)** |
| **Accuracy** | **0.8924 (89.24%)** | **1.0000 (100.0%)** |
| **Precision (PPV)** | 0.5186 (51.86%) | 0.0000 (0 TP / 0 predicted risk) |
| **Recall (Sensitivity)** | **0.7975 (79.75%)** | *Undefined* (0 positive ground-truth events) |
| **F1 Score** | 0.6285 | *Undefined* (no positive ground-truth events) |
| **PR-AUC (Primary)** | **0.7574** | *Undefined* (single-class ground truth) |
| **ROC-AUC** | **0.9162** | *Undefined* (single-class ground truth) |

> *Source artifact:* [`results/external_validation/INTERNAL_VS_EXTERNAL_COMPARISON.csv`](results/external_validation/INTERNAL_VS_EXTERNAL_COMPARISON.csv)

### Key Validation Findings & Honest Disclosure:
1. **Zero False Alarm Rate on Clean Telemetry (100.0% Specificity):** The frozen model achieved **100.0% Specificity** across all 742 clean evaluation intervals ($\\text{TN} = 742, \\text{FP} = 0$), generating zero false alarms during continuous live monitoring.
2. **Conservative Risk Calibration:** The mean predicted risk probability was **8.93%** (median 7.11%, max 44.06%), safely below the 50% action threshold.
3. **Absence of External Low-DO Ground Truth:** Because the Oman experimental tank maintained active mechanical aeration, DO remained between **6.08 and 12.25 mg/L** (mean 7.78 mg/L). Not a single true hypoxic event ($< 3.0\\text{ mg/L}$) occurred in the ground truth.
4. **Transparent Incomplete Metric Disclosure:** Because the external positive class is completely absent, metrics requiring positive instances (Recall, F1, PR-AUC, ROC-AUC) are mathematically undefined. In accordance with strict scientific integrity standards, ShinerAI **reports these metrics as undefined** rather than fabricating synthetic scores.

### The 10 Documented Scientific Limitations:
1. **Zero External Positive Events:** Continuous mechanical aeration prevented any dissolved oxygen crash ($< 3.0\\text{ mg/L}$), precluding empirical evaluation of external Recall/Sensitivity.
2. **Geographic & Climate Shift:** Arid desert environment in Oman with extreme diurnal ambient swings vs. humid subtropical Arkansas.
3. **Species Biology Shift:** Nile tilapia (*O. niloticus*, higher hypoxia tolerance) vs. golden shiner (*N. crysoleucas*).
4. **Scale & Facility Divergence:** 180-liter closed indoor/covered tank vs. multi-acre open commercial earthen ponds with massive thermal inertia and sediment oxygen demand.
5. **Sensor Hardware Divergence:** Low-cost analog galvanic probe + ESP32 vs. commercial optical multi-parameter sonde.
6. **Short Monitoring Horizon:** 8.74 continuous days in Oman vs. 63 days of seasonal monitoring across 17 Arkansas ponds.
7. **Continuous Aeration Regime:** Masks natural nocturnal respiration drops driven by phytoplankton blooms.
8. **Analog Sensor Dropouts:** Hardware disconnects producing instantaneous 0.0 mg/L readings required explicit quality filters to prevent spurious feature shifts.
9. **Unused Co-variates:** External telemetry recorded temperature and pH, but Config C intentionally omits them for sensor parsimony.
10. **Unproven External Sensitivity Generalization:** Model specificity is empirically verified, but out-of-distribution sensitivity during real oxygen crashes remains to be validated when external hypoxic telemetry becomes publicly available.

> *Full Reports:* [`results/external_validation/EXTERNAL_VALIDATION_REPORT.md`](results/external_validation/EXTERNAL_VALIDATION_REPORT.md) | [`results/external_validation/EXTERNAL_DATASET_AUDIT.md`](results/external_validation/EXTERNAL_DATASET_AUDIT.md) | [`results/external_validation/EXTERNAL_VALIDATION_PROTOCOL.md`](results/external_validation/EXTERNAL_VALIDATION_PROTOCOL.md)
"""

# Replace the Authoritative Active Model Benchmark section with the expanded benchmark + comparison + validation
target_section = """### Authoritative Active Model Benchmark (Production Standard)

The production model [`models/xgboost_config_c.joblib`](models/xgboost_config_c.joblib) is evaluated on the held-out temporal partition ($N = 8,261$, $943$ positive events) with a mandatory 2-hour purge gap:

| Metric | Verified Value | Benchmark Detail |
| :--- | :---: | :--- |
| **PR-AUC (Primary)** | **0.7574** | **+0.1425 lift** over Current-DO baseline (0.6149) |
| **ROC-AUC** | **0.9162** | High discriminatory capability across classification thresholds |
| **F1-Score** | **0.6285** | Balanced performance at default operational decision threshold $\\tau = 0.50$ |
| **Recall (Sensitivity)** | **0.7975** | Catches 79.75% of impending hypoxic events (752 of 943) |
| **Precision (PPV)** | **0.5186** | 752 true positives out of 1,450 total alerts |
| **Specificity (TNR)** | **0.9046** | Correctly rejects 90.46% of normoxic intervals (6,620 of 7,318) |
| **Accuracy** | **0.8924** | Overall classification accuracy on temporal holdout (7,372 of 8,261) |
| **Confusion Matrix** | **TP: 752, FP: 698, TN: 6,620, FN: 191** | Evaluated on 8,261 holdout observations with 2-hour purge gap |"""

if target_section in readme_text:
    readme_text = readme_text.replace(target_section, model_comparison_and_external_val_section)
    print("Replaced benchmark section in README.md")
else:
    # Try regex replacement if exact string didn't match
    pattern = r"### Authoritative Active Model Benchmark \(Production Standard\).*?TP: 752, FP: 698, TN: 6,620, FN: 191.*?\n"
    readme_text = re.sub(pattern, model_comparison_and_external_val_section + "\n", readme_text, flags=re.DOTALL)
    print("Replaced benchmark section in README.md via regex")

# Update test suite mentions in README
readme_text = readme_text.replace("85 Tests", "97 Tests")
readme_text = readme_text.replace("85 automated tests", "97 automated tests")
readme_text = readme_text.replace("78 tests", "97 tests")

readme_path.write_text(readme_text, encoding="utf-8")
print("Updated README.md successfully!")

# ==============================================================================
# 2. Update PROJECT_DOCUMENTATION.md
# ==============================================================================
doc_path = Path("PROJECT_DOCUMENTATION.md")
doc_text = doc_path.read_text(encoding="utf-8")
doc_text = re.sub(r'file:///d:/FISH/', '', doc_text)

doc_addition = """
---

## 6. Consolidated Model Comparison & Feature Ablation Analysis

### 6.1 Benchmark Comparison across 11 Model Configurations
To ensure rigorous evaluation and transparent presentation, all baseline heuristics and candidate models were tested on the exact same leak-free chronological holdout ($N = 8,261$, 943 positive events, 2-hour purge gap):

| Model | Feature Configuration | PR-AUC | ROC-AUC | F1 | Recall | Precision | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|---|
| **Majority Baseline** | None (Class Distribution Only) | **0.1142** | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8858 |
| **Current-DO Baseline** | Current DO Only (Static Threshold $\\le 4.2$ mg/L) | **0.6149** | 0.9024 | 0.4516 | 0.8961 | 0.3019 | 0.7330 | 0.7516 |
| **Logistic Regression** | Config A — Current DO + Time (5 Predictors) | **0.6019** | 0.8994 | 0.4274 | 0.9003 | 0.2802 | 0.7020 | 0.7246 |
| **Logistic Regression** | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.6763** | 0.9078 | 0.4295 | 0.8993 | 0.2821 | 0.7051 | 0.7273 |
| **Logistic Regression** | Config C — Current + Recent DO History (11 Predictors) | **0.6550** | 0.9093 | 0.4549 | 0.8940 | 0.3051 | 0.7376 | 0.7555 |
| **Random Forest** | Config A — Current DO + Time (5 Predictors) | **0.7107** | 0.9116 | 0.6174 | 0.7709 | 0.5149 | 0.9064 | 0.8909 |
| **Random Forest** | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.7420** | 0.9169 | 0.6233 | 0.7614 | 0.5276 | 0.9121 | 0.8949 |
| **Random Forest** | Config C — Current + Recent DO History (11 Predictors) | **0.7471** | 0.9144 | 0.6602 | 0.7561 | 0.5859 | 0.9311 | 0.9111 |
| **XGBoost** | Config A — Current DO + Time (5 Predictors) | **0.7317** | 0.9150 | 0.5817 | 0.8102 | 0.4537 | 0.8743 | 0.8670 |
| **XGBoost** | Config B — Current + DO/pH/Temp History (29 Predictors) | **0.7353** | 0.9171 | 0.5922 | 0.7932 | 0.4725 | 0.8859 | 0.8753 |
| **XGBoost (Champion)** | **Config C — Current + Recent DO History (11 Predictors)** | **0.7574** | **0.9162** | **0.6285** | **0.7975** | **0.5186** | **0.9046** | **0.8924** |

> *Authoritative source artifact:* [`results/reports/MODEL_COMPARISON.csv`](results/reports/MODEL_COMPARISON.csv) | Visualization: [`results/figures/model_comparison_prauc.png`](results/figures/model_comparison_prauc.png)

### 6.2 Feature Ablation: Quantifying Temporal Context
| Model Architecture | Config A PR-AUC | Config B PR-AUC | Config C PR-AUC | Config B - Config A | Config C - Config A Lift | Config C - Config B |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.6019 | 0.6763 | 0.6550 | +0.0744 | **+0.0531** | -0.0213 |
| **Random Forest** | 0.7107 | 0.7420 | 0.7471 | +0.0313 | **+0.0364** | +0.0051 |
| **XGBoost** | 0.7317 | 0.7353 | 0.7574 | +0.0036 | **+0.0257** | +0.0221 |

---

## 7. Independent External Validation (Oman Nile Tilapia Telemetry)

### 7.1 Dataset & Protocol Specification
To assess real-world out-of-distribution transferability, the frozen ShinerAI production model (`models/xgboost_config_c.joblib`) was evaluated on an external dataset from Nile tilapia (*Oreochromis niloticus*) aquaculture in North Al Sharqiyah, Oman (*Sensors* 2026, 26(13), 4242; DOI: 10.3390/s26134242; CC BY 4.0; GitHub: `https://github.com/AhmedTheNetCoder/DO-Forecasting-Tilapia-Dataset`).

**Protocol Guardrails:**
- **Zero Retraining:** The active model was strictly locked and evaluated without weight or threshold adaptation.
- **Decision Threshold:** Fixed at $\\tau = 0.50$.
- **15-Minute Resampling:** Raw readings aggregated into right-closed $(T-15\\text{m}, T]$ intervals.
- **Eligible Samples:** 742 15-minute intervals spanning 8.74 continuous days of live tilapia monitoring.

### 7.2 Internal vs. External Validation Benchmark
| Metric | Internal FWI Holdout | External Oman Tilapia Validation |
|---|---|---|
| **Sample Count ($N$)** | 8,261 intervals | 742 intervals |
| **Positive Events (AT_RISK)** | 943 (11.42%) | 0 (0.00%; continuous aeration) |
| **Negative Samples (SAFE)** | 7,318 | 742 |
| **True Negatives (TN)** | 6,620 | 742 |
| **False Positives (FP)** | 698 | 0 |
| **True Positives (TP)** | 752 | 0 |
| **False Negatives (FN)** | 191 | 0 |
| **Specificity (TNR)** | **0.9046 (90.46%)** | **1.0000 (100.0%)** |
| **Accuracy** | **0.8924 (89.24%)** | **1.0000 (100.0%)** |
| **Precision** | 0.5186 (51.86%) | 0.0000 (0 TP / 0 alerts) |
| **Recall / Sensitivity** | 0.7975 (79.75%) | *Undefined* (0 ground-truth events) |
| **F1 Score** | 0.6285 | *Undefined* (no ground-truth events) |
| **PR-AUC** | 0.7574 | *Undefined* (single-class ground truth) |
| **ROC-AUC** | 0.9162 | *Undefined* (single-class ground truth) |

### 7.3 Documented Scientific Limitations
1. **Absence of Ground-Truth Positives:** Continuous mechanical aeration prevented DO from dropping below 3.0 mg/L (min 6.08 mg/L), precluding empirical evaluation of Recall/Sensitivity.
2. **Climate & Geographic Shift:** Arid desert Oman vs. humid subtropical Arkansas.
3. **Species Biology Shift:** Nile tilapia (*O. niloticus*) vs. golden shiner (*N. crysoleucas*).
4. **Facility Scale Shift:** 180-liter closed indoor/covered experimental tank vs. multi-acre open earthen ponds.
5. **Sensor Platform Shift:** Low-cost analog probe + ESP32 vs. optical multi-parameter sonde.
6. **Short Duration:** 8.74 continuous days vs. 63 days of seasonal monitoring.
7. **Continuous Aeration Regime:** Suppressed natural nocturnal phytoplankton respiration crashes.
8. **Analog Sensor Dropouts:** Hardware dropouts to 0.0 mg/L required preprocessing filters.
9. **Unused Co-variates:** External telemetry recorded temperature and pH, but Config C uses DO history only.
10. **Unproven External Sensitivity Generalization:** Model specificity is verified (100%), but early warning capability in hypoxic external settings requires future open-source hypoxic datasets.
"""

# Append before the Appendix
if "## Appendix: Complete Figures & Visualizations Reference" in doc_text:
    doc_text = doc_text.replace("## Appendix: Complete Figures & Visualizations Reference", doc_addition + "\n## Appendix: Complete Figures & Visualizations Reference")
    print("Added sections to PROJECT_DOCUMENTATION.md")
else:
    doc_text += doc_addition
    print("Appended sections to PROJECT_DOCUMENTATION.md")

doc_path.write_text(doc_text, encoding="utf-8")
print("Updated PROJECT_DOCUMENTATION.md successfully!")
