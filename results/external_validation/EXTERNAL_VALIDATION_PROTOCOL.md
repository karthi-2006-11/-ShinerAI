# Independent External Validation Protocol

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Document:** External Validation Experimental Protocol  
**Version:** 1.0 (Frozen Protocol)  
**Target Model:** `models/xgboost_config_c.joblib` (Frozen Phase 3 Champion)  

---

## 1. Scientific Objectives & Guiding Principles

This protocol governs the independent external validation of ShinerAI against an external, publicly available aquaculture dataset. The experiment strictly adheres to clinical and regulatory principles of **locked algorithmic validation**:

1. **Zero Retraining:** The model weights, tree structures, and hyperparameter configurations are completely frozen.
2. **Zero Parameter Adaptation:** No fine-tuning, domain adaptation, transfer learning, or calibration on the external data.
3. **Zero Threshold Optimization:** The operational decision threshold is permanently fixed at **0.50**, identical to the internal temporal benchmark.
4. **Task Preservation:** The definition of an impending low-DO event ($< 3.0\text{ mg/L}$ within the next 2 hours given current DO $\ge 3.0\text{ mg/L}$) remains uncompromised.
5. **No Synthetic Manipulation:** Data are evaluated as observed; no synthetic hypoxic points are inserted to force metric computation.

---

## 2. External Dataset Specification

- **Source Repository:** [AhmedTheNetCoder/DO-Forecasting-Tilapia-Dataset](https://github.com/AhmedTheNetCoder/DO-Forecasting-Tilapia-Dataset)
- **Primary Data Stream:** `data/raw/live/raw_readings_new.csv` (102,670 raw high-frequency observations)
- **Secondary Data Stream:** `data/raw/offline/raw_readings.csv` (41,663 raw observations) and `data/processed/aggregated_data.csv` (897 5-minute pre-aggregated observations)
- **Location:** North Al Sharqiyah, Oman
- **Target Organism:** Nile Tilapia (*Oreochromis niloticus*)
- **Sensor Platform:** Low-cost Gravity analog optical DO sensor connected to ESP32 microcontroller
- **Sampling Frequency:** ~5–7 second raw telemetry

---

## 3. Preprocessing & Leakage-Free 15-Minute Grid Resampling

The internal ShinerAI pipeline operates on discrete **15-minute telemetry intervals**. Because the external dataset provides raw sensor readings at ~5–7 second intervals (and 5-minute aggregations), a mathematically rigorous resampling method is required:

### Resampling Rules:
1. **Right-Closed, Right-Labeled Bins:**
   $$\text{Bin } T = (T - 15\text{ min}, T]$$
   An interval labeled $T = \text{10:15:00}$ aggregates all sensor observations occurring strictly in the half-open window $(10:00:00, 10:15:00]$.
2. **Zero Future Information:**
   No reading timestamped $> T$ can influence the measurement for time $T$.
3. **Sensor Dropout Filtering:**
   Exact zero readings ($\text{DO} \le 0.0\text{ mg/L}$) occurring due to electrical disconnects or sensor warmups are treated as invalid (`NaN`) and excluded from window means.
4. **Missing Window Accounting:**
   Any 15-minute window with zero valid sensor readings is marked as `NaN` (unobserved). Forward-filling across long network outages is prohibited.

---

## 4. Feature Extraction (Configuration C)

At each evaluation point $T$, the 11 Config C predictors are constructed:
1. `current_do`: Window mean DO at time $T$.
2. `hour_of_day`: Integer hour of $T$ ($0 \le \text{hour} \le 23$).
3. `minute_of_day`: Minute of day of $T$ ($\text{hour} \times 60 + \text{minute}$).
4. `do_t_minus_15`: Window mean DO at $T - 15\text{ min}$.
5. `do_t_minus_30`: Window mean DO at $T - 30\text{ min}$.
6. `do_t_minus_45`: Window mean DO at $T - 45\text{ min}$.
7. `do_t_minus_60`: Window mean DO at $T - 60\text{ min}$.
8. `do_t_minus_75`: Window mean DO at $T - 75\text{ min}$.
9. `do_t_minus_90`: Window mean DO at $T - 90\text{ min}$.
10. `do_t_minus_105`: Window mean DO at $T - 105\text{ min}$.
11. `do_t_minus_120`: Window mean DO at $T - 120\text{ min}$.

---

## 5. Ground-Truth Target Generation

For every observation at time $T$:
1. **Precondition:** Current DO must satisfy the operational boundary condition:
   $$\text{current\_do} \ge 3.0\text{ mg/L}$$
2. **Future Forecast Window:** The next 2-hour trajectory is defined by the subsequent 8 consecutive 15-minute intervals:
   $$\mathcal{F}(T) = \{T + 15\text{m}, T + 30\text{m}, T + 45\text{m}, T + 60\text{m}, T + 75\text{m}, T + 90\text{m}, T + 105\text{m}, T + 120\text{m}\}$$
3. **Label Assignment:**
   $$y_T = \begin{cases} 
   1 \text{ (AT\_RISK)} & \text{if } \min_{t \in \mathcal{F}(T)} \text{DO}_t < 3.0\text{ mg/L} \\
   0 \text{ (SAFE)} & \text{if } \min_{t \in \mathcal{F}(T)} \text{DO}_t \ge 3.0\text{ mg/L}
   \end{cases}$$
4. **Exclusion Criteria:**
   - If any past lag in $\{T - 15\text{m}, \dots, T - 120\text{m}\}$ is missing, the sample is excluded.
   - If any future window in $\mathcal{F}(T)$ is missing (e.g., at the end of recording), the sample is excluded because the ground truth is unverifiable.

---

## 6. Execution & Metric Reporting Rules

- Compute model predicted risk probabilities: $P(y = 1 \mid X)$.
- Apply the locked decision threshold $\tau = 0.50$ to generate binary predictions $\hat{y} = \mathbb{I}(P \ge 0.50)$.
- Populate the full confusion matrix:
  - $\text{TP} = \sum \mathbb{I}(y = 1 \land \hat{y} = 1)$
  - $\text{FP} = \sum \mathbb{I}(y = 0 \land \hat{y} = 1)$
  - $\text{TN} = \sum \mathbb{I}(y = 0 \land \hat{y} = 0)$
  - $\text{FN} = \sum \mathbb{I}(y = 1 \land \hat{y} = 0)$
- Compute Specificity ($\text{TN} / (\text{TN} + \text{FP})$) and Accuracy ($(\text{TP} + \text{TN}) / N$).
- **Handling Zero-Positive Distributions:** If the external dataset contains zero true positive events ($\sum y = 0$), report Recall, F1, PR-AUC, and ROC-AUC as mathematically undefined rather than reporting synthetic or misleading scores.
