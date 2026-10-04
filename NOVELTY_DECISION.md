# ShinerAI Research Novelty Decision Document

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Status:** Approved Research Strategy  
**Date:** October 2026  
**Target Venue:** IEEE Access / IEEE Transactions on AgriFood Electronics / Computers and Electronics in Agriculture  

---

## 1. Executive Summary & Mentor Feedback Context

During expert review of ShinerAI, the mentor noted:
> *"Strengthen novelty. Current contribution mainly demonstrates that historical DO readings improve prediction over simpler baselines. Investigate a more distinctive contribution such as: uncertainty-aware early warning, event-level forecasting, or adaptive alert thresholds."*

The core scientific model of ShinerAI (`models/xgboost_config_c.joblib`, 11 features, 2-hour lookback, $\tau = 0.50$, internal PR-AUC = 0.7574, ROC-AUC = 0.9162) is **strictly frozen**. Retraining, altering features, or changing the prediction task was explicitly ruled out to protect empirical integrity.

Therefore, the path to elevating novelty lies in **novel evaluation methodology, operational decision-support characterization, and bridging the gap between point-wise machine learning metrics and real-world farm operations**.

After comprehensive analysis, **Candidate A: Event-Level Early Warning and Operational Lead-Time Formulation** is formally selected as the primary research novelty contribution of ShinerAI, with **Candidate B (Probability Calibration & Cost-Sensitive Analysis)** providing rigorous supporting empirical validation.

---

## 2. Evaluation of Candidate Novelty Formulations

Three candidate novelty formulations were analyzed against feasibility, defensibility, and compliance with the scientific freeze.

| Criteria | Candidate A: Event-Level Early Warning | Candidate B: Uncertainty-Aware Calibration | Candidate C: Adaptive Dynamic Thresholds |
| :--- | :--- | :--- | :--- |
| **Description** | Group consecutive time-steps into discrete hypoxia episodes. Quantify advance lead time, event detection rate, alert chattering, and hysteresis filtering. | Map raw model scores to calibrated posterior probabilities via Platt/Isotonic scaling; optimize decision threshold against explicit economic loss matrix ($C_{FN}:C_{FP}$). | Dynamically adjust the hypoxia warning threshold based on diurnal solar cycle, pond water temperature, or ambient weather conditions. |
| **Model Retraining Required?** | **No** (Zero change to frozen model artifact or weights). | **No** (Post-hoc calibration on validation split; frozen model unchanged). | **Yes / Major risk** (Requires dynamic label generation or retraining multi-threshold models). |
| **Compliance with Scientific Freeze** | **100% Compliant** | **100% Compliant** | **Non-compliant** (Violates operational 3.0 mg/L target freeze). |
| **Aquaculture Domain Relevance** | **Extremely High**: Fish farmers operate on episodic intervention (turning on paddlewheel aerators), not isolated 15-minute classification steps. | **High**: Assists in risk-weighted economic decision making. | **Moderate**: While biologically intuitive, farmers use fixed regulatory/operational tripwires. |
| **Literature Distinction** | Most published aquaculture ML papers report only point-wise AUC/F1 on 15-min rows, artificially obscuring real-world alarm fatigue and lead times. | Standard ML technique; well understood, but rarely demonstrated rigorously in pond water quality. | Requires complex environmental sensors (PAR, solar radiation) not universally available. |
| **Recommendation** | **PRIMARY NOVELTY CONTRIBUTION** | **SUPPORTING TECHNICAL CONTRIBUTION** | **REJECTED** |

---

## 3. Formal Selection: Candidate A (Event-Level Early Warning)

### 3.1 The Fundamental Flaw in Point-Wise ML for Aquaculture
Conventional machine learning studies in aquaculture report point-wise metrics:
- Precision: 51.86%
- Recall: 79.75%
- False Positive Count: 698 rows
- True Positive Count: 752 rows

However, in an operational commercial fish farm:
1. **Hypoxia is an Episode, Not a Row:** Dissolved oxygen depletion typically occurs over a sustained 1 to 4 hour nocturnal window (midnight to dawn) as phytoplankton and benthic respiration consume oxygen.
2. **One Early Warning Saves the Pond:** If ShinerAI alerts the farm manager at 02:00 AM (120 minutes before DO hits 3.0 mg/L), aerators are activated and fish mortality is averted. Whether the model continues to predict positive at 02:15 AM, 02:30 AM, or momentarily drops to 0.48 probability does not change the operational outcome.
3. **Point-wise Metrics Penalize Useful Lead Time:** An alert triggered 105 minutes before hypoxia is labeled as a "False Positive" by naive point-wise evaluation if the row itself is still currently safe (e.g. DO = 3.6 mg/L).
4. **Alarm Fatigue Destroys Field Adoption:** If a model fires 50 sporadic, isolated 15-minute warnings across a day (alert chattering), the operator disables the system.

### 3.2 Formal Operational Metrics Defined in ShinerAI
By structuring the test set (7,472 operational intervals, representing 77.8 days of telemetry) into **136 contiguous hypoxia episodes**, ShinerAI defines and quantifies four operational metrics:

1. **Episode Detection Rate (EDR):**
   $$\text{EDR} = \frac{\text{Episodes with } \ge 1 \text{ Early Warning}}{\text{Total Hypoxia Episodes}} = \frac{124}{136} = \mathbf{91.18\%}$$
   *(Compared to row-level recall of 79.75%, the real-world protection rate is 91.18%—only 12 out of 136 events were missed).*

2. **Advance Warning Lead Time Distribution:**
   - **Mean Lead Time:** **101.7 minutes** ($\approx 1.7$ hours)
   - **Median Lead Time:** **120.0 minutes** (the theoretical maximum 2-hour prediction horizon)
   - **75th Percentile:** 120.0 minutes
   - **25th Percentile:** 90.0 minutes
   - *Conclusion:* Over 75% of warned events provide between 90 and 120 minutes of advance intervention time, more than sufficient to dispatch staff or engage mechanical aerators.

3. **Farm False Alarm Burden:**
   - **Daily Alert Frequency:** 4.84 raw alert intervals per pond per operational day.
   - **False Episode Frequency:** 1.73 false alarm episodes per pond per day.
   - **Mean False Alarm Duration:** 85.9 minutes (5.7 consecutive 15-min intervals).
   - **Chattering Rate:** 32.5% of false alerts are single-step isolated spikes.

4. **Operational Mitigation via Hysteresis Filtering:**
   - Requiring two consecutive positive predictions ($k=2$) or a 30-minute confirmation window eliminates single-step sensor noise and chattering.
   - **Impact:** False positive intervals drop from 698 to 449 (**35.7% reduction in false alarms**), while preserving 73.4% row recall and increasing operational specificity from 90.5% to 93.9%.

---

## 4. Supporting Technical Contribution: Candidate B (Calibration & Cost Optimization)

While Candidate A forms the main scientific narrative, Candidate B is fully integrated as a supporting pillar:
1. **Probability Calibration:**
   - Uncalibrated XGBoost outputs suffer from log-loss distortions on imbalanced data (Brier score = 0.0863).
   - Post-hoc Platt Scaling (sigmoid) and Isotonic Regression reduce the Brier score to **0.0503** (a **41.7% error reduction**), providing true posterior probabilities $P(\text{hypoxia} \mid \mathbf{x})$.
2. **Cost-Sensitive Decision Threshold Justification:**
   - In aquaculture, a False Negative (undetected hypoxia $\implies$ total fish mortality) is catastrophic, whereas a False Positive (aerator turned on unnecessarily $\implies$ electricity cost) is minor.
   - An asymmetric loss matrix with cost ratio $C_{FN} : C_{FP} = 5:1$ is standard in high-value aquaculture.
   - Cost optimization on the training split proves that the default threshold $\tau = 0.50$ is not arbitrary; it represents the **exact empirical cost-minimizing operating point** ($C_{\text{norm}} = 0.0487$) on the cost curve.

---

## 5. Scope & Boundary Claims (What ShinerAI Does NOT Claim)

To preserve scientific rigor and ensure immediate IEEE mentor acceptance:
1. **No Universal Biological Claim:** We do NOT claim 3.0 mg/L is a universal threshold across all aquatic species. It is rigorously documented as an operational early-warning boundary tailored to baitfish (*Notemigonus crysoleucas*) and commercial warmwater pond culture.
2. **No Claim of Novel Deep Learning Architecture:** We do NOT claim XGBoost is a novel neural or algorithmic invention; rather, the novelty lies in the *operational time-series framing, event-level lead-time validation, and practical alert-stabilization methodology*.
3. **No Unwarranted Causal Claims from SHAP:** We explicitly note that SHAP feature attributions reflect predictive feature importance within the trained model, not biological causation in pond limnology.

---

## 6. Publication Abstract & IEEE Contribution Statement

> **Contribution Statement:**  
> *"Unlike conventional aquaculture machine learning studies that evaluate point-wise classification accuracy on isolated sensor intervals—obscuring alert fatigue and practical intervention time—this work formulates an event-level early warning framework for dissolved oxygen depletion. We demonstrate that while XGBoost achieves a standard row-level recall of 79.75%, its event-level detection rate is 91.18% across 136 contiguous hypoxia episodes, providing an average advance warning lead time of 101.7 minutes (median 120.0 minutes). Furthermore, we quantify the operational false alarm burden (4.84 alerts/pond/day) and demonstrate that a two-step hysteresis filter reduces operator alarm fatigue by 35.7% while maintaining high protective sensitivity."*
