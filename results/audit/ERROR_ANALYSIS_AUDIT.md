# ShinerAI: Empirical Error Analysis Audit

**Audit Date:** October 3, 2026  
**Auditor:** Scientific Audit Agent  
**Active Production Model:** `models/xgboost_config_c.joblib`  
**Test Set:** Held-out Temporal Partition ($N = 8,261$, Positives = 943, Negatives = 7,318)  

---

## 1. Executive Summary

In mission-critical environmental machine learning, overall metrics (such as PR-AUC or accuracy) can conceal distinct failure modes. A responsible engineering evaluation requires examining both categories of predictive error:
- **False Positives ($FP = 698$):** The model raises an alarm, but dissolved oxygen does not drop below $3.0\text{ mg/L}$ within 2 hours.
- **False Negatives ($FN = 191$):** Dissolved oxygen drops below $3.0\text{ mg/L}$ within 2 hours, but the model fails to predict the event.

This audit dissects the physical telemetry characteristics of these errors, explains why they occur, and details operational mitigation strategies.

---

## 2. Quantitative Error Breakdown

| Cohort | Count | Mean Current DO (mg/L) | Median Current DO (mg/L) | Mean 1-Hour DO Velocity (mg/L/hr) | Operational Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **True Positive (TP)** | 752 | 3.64 | 3.50 | -0.48 | Timely early warning triggered (2h lead time). |
| **False Positive (FP)** | 698 | 4.51 | 4.37 | -0.48 | Spurious aerator spin-up; minor operational cost. |
| **False Negative (FN)** | 191 | 6.93 | 6.10 | +0.20 | Unanticipated hypoxia event; fish stress risk. |
| **True Negative (TN)** | 6,620 | 9.58 | 8.96 | +0.17 | Correct quiet operation; energy conserved. |

---

## 3. Dissection of False Positives ($FP = 698$)

### Physical Characteristics:
1. **Near-Hypoxic Pre-Conditions:** Over **31.4% (219 of 698)** of False Positives occur when current DO is already depressed below $4.0\text{ mg/L}$ (mean $\text{DO} = 4.51\text{ mg/L}$).
2. **Steep Preceding Depletion:** The average 1-hour trajectory was declining rapidly ($\Delta \text{DO}_{1\text{h}} = -0.48\text{ mg/L/hr}$).
3. **Stabilization Above Threshold:** In these cases, the oxygen curve flattened out just above the threshold (e.g. stabilizing at $3.15–3.30\text{ mg/L}$) rather than crossing below $3.00\text{ mg/L}$. This often occurs due to wind-induced surface re-aeration or human intervention (farmers turning on paddlewheels).

### Operational Assessment:
From a biological risk standpoint, False Positives in ShinerAI are **"near-miss warnings"** rather than wild hallucinations. Aerating a pond whose DO is declining towards $3.2\text{ mg/L}$ is standard protective management in commercial farming.

---

## 4. Dissection of False Negatives ($FN = 191$)

### Physical Characteristics:
1. **High Initial DO:** Over **81.2% (155 of 191)** of False Negatives had current DO above $4.50\text{ mg/L}$ (mean $\text{DO} = 6.93\text{ mg/L}$).
2. **Delayed Hypoxia Crashes:** In these instances, DO remained flat or gently declined during the initial 60 minutes, then experienced an abrupt, non-linear plunge during minutes 90–120.
3. **Flash Inversions / Sudden Weather Events:** Sudden thunderstorm downpours or heavy overcast during afternoon hours can cause sudden thermal destratification and acute algal collapse, dropping DO faster than normal nocturnal respiration curves.

### Operational Assessment:
The model captures **79.75% of all events** ($752$ out of $943$). The missed $191$ events represent edge cases where the decline trajectory had not yet manifested in the 2-hour observation window.

---

## 5. Mitigation Strategies for Production Deployment

1. **Threshold Tuning Based on Cost Asymmetry:**
   - Where the cost of lost fish biomass ($C_{FN}$) far exceeds energy costs ($C_{FP}$), lowering the decision threshold to $\tau = 0.40$ increases Recall to **$84.9\%$** while maintaining acceptable specificity.
   - For energy-constrained farms, the model can be configured to $\tau = 0.55$, raising Precision to $54.6\%$.

2. **Hot-Swapping to Random Forest Config C:**
   - In facilities with strict noise limits or expensive diesel fuel, switching to `models/random_forest_config_c.joblib` reduces false positives from $698$ to **$504$ (a $27.8\%$ reduction)** with a Specificity of **$93.11\%$**.

---

## 6. Audit Conclusion

The error profile of ShinerAI reflects physically coherent failure modes: False Positives are predominantly near-miss downward trajectories, and False Negatives are high-DO abrupt drop events. The operational risk profile is controllable via threshold calibration.
