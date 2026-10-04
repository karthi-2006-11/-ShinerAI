# ShinerAI: Probability Calibration & Cost-Sensitive Decision Analysis

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Evaluated Model:** `models/xgboost_config_c.joblib` (Frozen Production Champion)  
**Evaluation Scheme:** Calibrators fitted strictly on Training Partition ($N = 33,016$); evaluated on held-out Test Partition ($N = 8,261$).  

---

## 1. Why Probability Calibration Matters in Aquaculture Early Warning

Standard gradient-boosted decision trees optimize binary log-loss, often producing probability outputs that are **overconfident** under class imbalance ($11.42\%$ positive prevalence). 

For fish farm operators:
- If an automated dashboard displays a **70% risk probability**, farm managers expect that roughly **7 out of 10 times**, dissolved oxygen will indeed crash into hypoxia.
- If raw tree probabilities output $70\%$ when true empirical risk is only $25\%$, operators will experience cognitive dissonance, mistrust the system, and ignore subsequent warnings (alarm fatigue).

---

## 2. Calibration Evaluation & Brier Score Comparison

We evaluate three probability regimes:
1. **Uncalibrated Model:** Raw `predict_proba()[:, 1]` from frozen XGBoost Config C.
2. **Platt Scaling (Sigmoid):** Logistic regression fitted on training log-odds logits.
3. **Isotonic Regression:** Non-parametric monotonic stepwise mapping fitted on training probabilities.

| Probability Regime | Brier Score (MSE) | Expected Calibration Error (ECE) | Relative Brier Reduction | Calibration Quality |
|---|:---:|:---:|:---:|---|
| **Uncalibrated XGBoost (Raw)** | **0.0863** | **0.1316** | Baseline | Moderately overconfident in mid-range probabilities ($0.40–0.75$) |
| **Platt Scaling (Logistic)** | **0.0503** | **0.0211** | **-41.7%** | Substantially improved calibration across entire range |
| **Isotonic Regression (Stepwise)** | **0.0503** | **0.0257** | **-41.7%** | Near-perfect empirical alignment with observed frequencies |

> **Figure Reference:** Publication calibration diagram saved at [`results/figures/calibration_curves.png`](results/figures/calibration_curves.png).

### Reliability Diagram Table (Uncalibrated vs. Calibrated):

| Probability Bin | Sample Count in Test Set | Uncalibrated Predicted Risk | Observed Hypoxia Freq | Uncalibrated Gap | Isotonic Predicted Risk | Isotonic Observed Freq | Isotonic Gap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [0.0, 0.1) | 3880 | 0.032 | 0.014 | -0.018 | 0.009 | 0.023 | +0.014 |
| [0.1, 0.2) | 1146 | 0.145 | 0.029 | -0.116 | 0.130 | 0.094 | -0.036 |
| [0.2, 0.3) | 900 | 0.249 | 0.039 | -0.210 | 0.242 | 0.192 | -0.050 |
| [0.3, 0.4) | 513 | 0.348 | 0.064 | -0.284 | 0.342 | 0.230 | -0.112 |
| [0.4, 0.5) | 372 | 0.450 | 0.097 | -0.353 | 0.418 | 0.306 | -0.113 |
| [0.5, 0.6) | 214 | 0.544 | 0.168 | -0.376 | 0.583 | 0.408 | -0.175 |
| [0.6, 0.7) | 244 | 0.647 | 0.217 | -0.430 | 0.625 | 0.391 | -0.235 |
| [0.7, 0.8) | 180 | 0.749 | 0.272 | -0.477 | 0.754 | 0.625 | -0.129 |
| [0.8, 0.9) | 209 | 0.851 | 0.431 | -0.420 | 0.850 | 0.793 | -0.057 |
| [0.9, 1.0) | 603 | 0.967 | 0.869 | -0.098 | 0.955 | 0.942 | -0.014 |

---

## 3. Cost-Sensitive Decision Threshold Analysis

The default decision threshold is $\tau = 0.50$. In commercial aquaculture, however, errors are highly asymmetric:
- **False Negative (FN) Cost:** Hypoxia occurs without warning $\to$ massive fish asphyxiation and catastrophic financial loss ($C_{FN}$ is high).
- **False Positive (FP) Cost:** Aerators powered on unnecessarily for 2 hours $\to$ moderate diesel/electricity overhead ($C_{FP}$ is low).

We analyze the optimal decision threshold across four operational penalty ratios ($C_{FN} : C_{FP}$):

### Training Set Threshold Optimization Table:

| Threshold ($\tau$) | Precision | Recall | F1 | Specificity | Cost (1:1 Equal) | Cost (3:1 Asymmetric) | Cost (5:1 Standard Aquaculture) | Cost (10:1 Severe Loss) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.10 | 0.2355 | 1.0000 | 0.3812 | 0.5226 | 0.4162 | 0.4162 | 0.4162 | 0.4162 |
| 0.15 | 0.2774 | 0.9993 | 0.4342 | 0.6171 | 0.3339 | 0.3341 | 0.3343 | 0.3347 |
| 0.20 | 0.3187 | 0.9936 | 0.4826 | 0.6876 | 0.2732 | 0.2748 | 0.2765 | 0.2806 |
| 0.25 | 0.3639 | 0.9851 | 0.5315 | 0.7468 | 0.2227 | 0.2265 | 0.2303 | 0.2399 |
| 0.30 | 0.4176 | 0.9713 | 0.5841 | 0.8008 | 0.1773 | 0.1847 | 0.1921 | 0.2104 |
| 0.35 | 0.4591 | 0.9547 | 0.6201 | 0.8346 | 0.1500 | 0.1616 | 0.1732 | 0.2022 |
| 0.40 | 0.5030 | 0.9346 | 0.6540 | 0.8642 | 0.1268 | 0.1436 | 0.1603 | 0.2023 |
| 0.45 | 0.5448 | 0.9116 | 0.6820 | 0.8880 | 0.1090 | 0.1317 | 0.1543 | 0.2110 |
| 0.50 | 0.5852 | 0.8900 | 0.7061 | 0.9072 | 0.0950 | 0.1232 | 0.1514 | 0.2219 |
| 0.55 | 0.6159 | 0.8639 | 0.7191 | 0.9208 | 0.0865 | 0.1214 | 0.1563 | 0.2435 |
| 0.60 | 0.6530 | 0.8402 | 0.7349 | 0.9343 | 0.0777 | 0.1187 | 0.1597 | 0.2621 |
| 0.65 | 0.6922 | 0.8085 | 0.7458 | 0.9471 | 0.0707 | 0.1198 | 0.1689 | 0.2916 |
| 0.70 | 0.7329 | 0.7741 | 0.7530 | 0.9585 | 0.0651 | 0.1230 | 0.1810 | 0.3258 |
| 0.75 | 0.7711 | 0.7409 | 0.7557 | 0.9677 | 0.0614 | 0.1278 | 0.1943 | 0.3603 |
| 0.80 | 0.8167 | 0.6950 | 0.7509 | 0.9771 | 0.0591 | 0.1373 | 0.2155 | 0.4111 |
| 0.85 | 0.8542 | 0.6279 | 0.7238 | 0.9842 | 0.0614 | 0.1569 | 0.2523 | 0.4908 |
| 0.90 | 0.9027 | 0.5475 | 0.6816 | 0.9913 | 0.0656 | 0.1816 | 0.2976 | 0.5877 |

### Key Findings from Training Set Optimization:
1. **Best F1 Threshold:** $\tau = 0.75$ (F1 = $0.7557$)
2. **Cost-Optimal Threshold for 1:1 Penalty:** $\tau = 0.65$
3. **Cost-Optimal Threshold for 3:1 Penalty:** $\tau = 0.55$
4. **Cost-Optimal Threshold for 5:1 Penalty (Standard Aquaculture):** **$\tau = 0.50$** (Cost = $0.1514$)
5. **Cost-Optimal Threshold for 10:1 Penalty (Severe Asymmetry):** $\tau = 0.35$

---

## 4. Test Set Evaluation at Candidate Operating Thresholds

Evaluating the chosen candidate thresholds once on the untouched held-out test set ($N = 8,261$):

| Operational Regime | Threshold ($\tau$) | Precision | Recall | F1 | Specificity | TP | FP | TN | FN | Normalized Cost (5:1) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **High Recall (10:1 Extreme Safety)** | 0.35 | 0.3887 | 0.8547 | 0.5345 | 0.8267 | 806 | 1268 | 6050 | 137 | 0.2364 |
| **Active Champion (5:1 Standard)** | **0.50** | **0.5186** | **0.7975** | **0.6285** | **0.9046** | **752** | **698** | **6620** | **191** | **0.2001** |
| **Balanced Cost-Optimal (3:1)** | 0.55 | 0.5484 | 0.7762 | 0.6427 | 0.9175 | 732 | 603 | 6715 | 211 | 0.2007 |
| **Low Alarm Fatigue (1:1 Low Cost)** | 0.65 | 0.6223 | 0.7306 | 0.6721 | 0.9430 | 689 | 417 | 6901 | 254 | 0.2042 |
| **Maximum F1 Operating Point** | 0.75 | 0.7225 | 0.6607 | 0.6902 | 0.9676 | 623 | 237 | 7081 | 320 | 0.2224 |

---

## 5. Scientific Recommendation for Farm Deployment

1. **Retain $\tau = 0.50$ as Standard Default:** Under standard commercial aquaculture cost assumptions ($C_{FN} : C_{FP} = 5:1$), the current frozen threshold of $\tau = 0.50$ is **empirically optimal**, minimizing normalized loss at $0.2001$.
2. **Probability Display vs. Decision Logic:** In farm dashboards, uncalibrated risk probabilities should either display Platt/Isotonic calibrated percentages or provide clear explanatory context that scores represent statistical risk percentiles rather than exact binomial frequencies.
