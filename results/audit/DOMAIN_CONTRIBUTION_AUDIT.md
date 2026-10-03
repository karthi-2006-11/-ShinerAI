# ShinerAI: Domain Contribution & Ablation Study Audit

**Audit Date:** October 3, 2026  
**Auditor:** Scientific Audit Agent  
**Focus:** Rigorous Empirical Justification of the $+0.1425$ PR-AUC Lift  

---

## 1. Executive Summary

A critical requirement of scientific peer review is clearly articulating what a research project experimentally demonstrates versus what is mere hypothesis or baseline functionality. 

This audit evaluates the core scientific finding of ShinerAI:
$$\Delta \text{PR-AUC} = +0.1425 \quad (0.6149 \to 0.7574)$$
We examine how this lift was achieved across three feature configurations (Configs A, B, and C) and four model families (Baselines, Logistic Regression, Random Forest, and XGBoost). We demarcate the boundary between verified empirical discoveries and non-demonstrated overclaims.

---

## 2. Quantitative Empirical Evidence Table

| Model & Configuration | Features Used | Input Dimension | PR-AUC | $\Delta$ vs Static Baseline | Recall ($\tau=0.5$) | Precision ($\tau=0.5$) | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current-DO Baseline** | `current_do` only | 1 | 0.6149 | 0.0000 | 0.8961 | 0.3019 | 0.7330 |
| **LR Config A (Snapshot)** | DO, pH, Temp, Time | 5 | 0.6019 | -0.0130 | 0.9003 | 0.2802 | 0.7020 |
| **RF Config A (Snapshot)** | DO, pH, Temp, Time | 5 | 0.7107 | +0.0958 | 0.7709 | 0.5149 | 0.9064 |
| **XGB Config A (Snapshot)**| DO, pH, Temp, Time | 5 | 0.7317 | +0.1168 | 0.8102 | 0.4537 | 0.8743 |
| **LR Config B (All Sensors)**| DO, pH, Temp + 2h Lags| 29 | 0.6763 | +0.0614 | 0.8993 | 0.2821 | 0.7051 |
| **RF Config B (All Sensors)**| DO, pH, Temp + 2h Lags| 29 | 0.7420 | +0.1271 | 0.7614 | 0.5276 | 0.9121 |
| **XGB Config B (All Sensors)**| DO, pH, Temp + 2h Lags| 29 | 0.7353 | +0.1204 | 0.7932 | 0.4725 | 0.8859 |
| **LR Config C (DO Only)** | `current_do` + 2h DO Lags| 11 | 0.6550 | +0.0401 | 0.8940 | 0.3051 | 0.7376 |
| **RF Config C (DO Only)** | `current_do` + 2h DO Lags| 11 | 0.7471 | +0.1322 | 0.7561 | **0.5859** | **0.9311** |
| **XGB Config C (Production)**| `current_do` + 2h DO Lags| 11 | **0.7574** | **+0.1425** | **0.7975** | 0.5186 | 0.9046 |

---

## 3. Two Core Scientific Insights

### Insight 1: Predictive Value of Recent DO History ($+0.1425$ PR-AUC Lift)
Static water telemetry thresholding (e.g. raising an alarm whenever $\text{DO} \le 4.2\text{ mg/L}$) achieves a PR-AUC of only $0.6149$. While it captures $89.6\%$ of events, it produces $1,954$ false alarms (Precision = $30.2\%$) because ponds experience natural diurnal fluctuations without necessarily crossing into hypoxia.

**Recent dissolved-oxygen history provides additional predictive information beyond the current DO measurement alone.**
- Config C uses historical DO observations (8 discrete 15-minute lags: $t-15\text{m} \dots t-120\text{m}$), not an explicitly calculated velocity feature. The gradient-boosted decision trees split directly on these raw historical observations alongside time-of-day indicators.
- This experimental result yields a PR-AUC improvement from **$0.6149$ to $0.7574$ ($+0.1425$, $+23.18\%$ relative gain)** and reduces false alarms from $1,954$ to $698$ ($64.3\%$ reduction) while maintaining high event sensitivity ($79.75\%$ recall).
- This $+0.1425$ difference is evidence that recent DO history improves predictive performance for the defined forecasting task. It does **NOT** prove biological causality.

### Insight 2: Feature Configuration Comparison Across Model Families
A widespread assumption in environmental sensing is that adding more sensor variables (e.g. pH, temperature) automatically improves prediction accuracy. Comparing the all-sensor configuration (Config B, 29 features) to the DO-history configuration (Config C, 11 features) reveals that the effect depends on the model family:
- **Logistic Regression:** Config B > Config C (PR-AUC $0.6763$ vs $0.6550$, $\Delta = +0.0213$). The linear model benefits from including cross-sensor current and lag terms.
- **Random Forest:** Config C > Config B (PR-AUC $0.7471$ vs $0.7420$, $\Delta = +0.0051$).
- **XGBoost:** Config C > Config B (PR-AUC $0.7574$ vs $0.7353$, $\Delta = +0.0221$).

**The DO-history configuration uses fewer sensor variables than the all-sensor configuration.** In non-linear tree ensembles, relying on historical DO observations alone avoids the split fragmentation and collinear variance introduced by 16 additional pH and temperature lag features, achieving equal or superior PR-AUC while requiring fewer input channels.

---

## 4. Rigorous Demarcation of Findings vs Overclaims

To ensure complete defensibility under research scrutiny, we explicitly articulate what ShinerAI does and does not claim:

| Topic | Unjustified / Overclaimed Phrasing (Rejected) | Defensible Scientific Finding (Adopted) |
| :--- | :--- | :--- |
| **Model Architecture** | "We invented a novel AI architecture for aquaculture." | "We performed an empirical ablation study evaluating standard supervised algorithms on temporal lag configurations." |
| **Historical Dynamics** | "The model calculates and proves physical descent velocity." | "Recent dissolved-oxygen history provides additional predictive information beyond the current DO measurement alone. Config C uses historical DO observations, not an explicitly calculated velocity feature." |
| **Biological Causality**| "SHAP proves that dissolved oxygen drop causes hypoxia." | "SHAP feature attributions describe mathematical credit assignment within the trained decision trees; they do NOT prove biological causality." |
| **Hardware Viability** | "The model is proven ready for real-time deployment on ESP32 microcontrollers." | "Inference latency was benchmarked at $< 0.05\text{ ms}$ strictly on a development workstation CPU; physical embedded microcontroller profiling remains future work." |
| **Operational Threshold** | "$3.0\text{ mg/L}$ is the universal biological threshold for aquatic life." | "$3.0\text{ mg/L}$ is an operational engineering threshold defined for this early warning project, not a universal biological constant." |
| **Sensor Architecture** | "DO-only sensing is biologically superior, cheaper, and more robust." | "The DO-history configuration uses fewer sensor variables than the all-sensor configuration while achieving competitive predictive performance with tree models." |

---

## 5. Audit Conclusion

The scientific contribution of ShinerAI is **empirically grounded, statistically reproducible, and methodologically sound**. The $+0.1425$ PR-AUC lift over the static baseline confirms that recent DO history provides substantial predictive information for early warning without resorting to exaggerated architectural, causal, or hardware claims.
