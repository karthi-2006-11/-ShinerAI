# ShinerAI: Alert Stability & Hysteresis Filter Verification Report

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Status:** Certified Independent Audit  
**Date:** October 2026  
**Audited Artifacts:**
- Test Set Telemetry ($N = 8,261$ intervals across 17 commercial ponds)
- Frozen Model: `models/xgboost_config_c.joblib` ($\tau = 0.50$)

---

## 1. Executive Summary

This independent audit evaluates the operational behavior of the raw XGBoost Config C model against a **2-step confirmation hysteresis filter** designed to suppress sensor chattering and operator alarm fatigue.

### Core Audit Takeaway (Honest Trade-off Disclosure):
- **False Alarm Suppression:** The 2-step hysteresis filter successfully reduces False Positive intervals from 698 to 507 (**-27.4% reduction**) and distinct false alarm clusters from 234 to 157 (**-32.9% reduction**), lowering daily alert burden from 4.84 to 3.52 alerts per pond per day.
- **Safety Impact (Crucial Finding):** Hysteresis does **NOT** operate "without compromising safety." Requiring two consecutive positive predictions results in:
  1. **6 newly missed hypoxia episodes** (Event Detection Rate drops from **91.18% to 86.76%**; 118/136 detected).
  2. **Advance lead-time delay of ~7.2 minutes** (Mean true lead time decreases from **93.1 to 85.9 minutes**; median drops from **120.0 to 112.5 minutes**).
  3. **Row-level recall reduction** from 79.75% to 73.59% (TP drops from 752 to 694).

---

## 2. Alert Definitions

To establish unambiguous mathematical definitions:

1. **Raw Model Alert:**
   An alert is triggered at timestamp $T_i$ if the uncalibrated model probability meets or exceeds the operational decision threshold:
   $$\hat{y}_i = \mathbb{I}(P(\text{AT\_RISK} \mid \mathbf{x}_i) \ge 0.50)$$

2. **2-Step Confirmation Hysteresis Alert:**
   An alert is confirmed at timestamp $T_i$ if and only if two consecutive 15-minute readings are predicted positive within the same contiguous sequence:
   $$\hat{y}_i^{\text{hyst}} = \mathbb{I}(\hat{y}_i = 1 \land \hat{y}_{i-1} = 1)$$
   *(Note: Single isolated 15-minute spikes where $\hat{y}_{i-1} = 0, \hat{y}_i = 1, \hat{y}_{i+1} = 0$ are suppressed).*

---

## 3. Independent Quantitative Audit Comparison

Evaluating both alert regimes on the identical 8,261 held-out test intervals:

| Operational Metric | Raw Model ($\tau = 0.50$) | 2-Step Hysteresis Filter | Impact / Trade-off |
| :--- | :---: | :---: | :--- |
| **False Positive (FP) Intervals** | 698 | 507 | **-27.36% (-191 intervals)** |
| **True Positive (TP) Intervals** | 752 | 694 | -7.71% (-58 intervals) |
| **Row-Level Precision** | 51.86% | 57.79% | **+5.93% improvement** |
| **Row-Level Recall** | 79.75% | 73.59% | **-6.16% degradation** |
| **Row-Level Specificity** | 90.46% | 93.07% | **+2.61% improvement** |
| **False Alarms per Pond-Day** | 4.84 alerts/day | 3.52 alerts/day | **-1.32 alerts/pond/day** |
| **Distinct False Alarm Clusters** | 234 episodes | 157 episodes | **-32.91% (-77 clusters)** |
| **Event Detection Rate (136 Episodes)**| **91.18% (124/136)** | **86.76% (118/136)** | **-4.41% (6 newly missed episodes)** |
| **Missed Hypoxia Episodes** | 12 | 18 | +6 missed episodes |
| **Mean Physical Lead Time** | **93.1 min** | **85.9 min** | **-7.2 min delay** |
| **Median Physical Lead Time** | **120.0 min** | **112.5 min** | -7.5 min delay |
| **75th Percentile Lead Time** | 120.0 min | 120.0 min | 0 min delay |
| **25th Percentile Lead Time** | 60.0 min | 45.0 min | -15.0 min delay |

---

## 4. Analysis of the 6 Newly Missed Episodes

An audit of the 6 episodes missed under the hysteresis filter reveals why they were suppressed:
- All 6 episodes were brief, transient hypoxia dips characterized by only **a single pre-event alert reading** ($\hat{y} = 1$) before the pond crossed below 3.0 mg/L or before the sensor sequence ended.
- Because the filter strictly requires a second confirmation step ($k=2$), these isolated pre-event alerts were discarded as probable transient noise.
- **Physical Consequence:** In all 6 cases, DO did cross below 3.0 mg/L, but the hypoxia event lasted for only 15 to 45 minutes before returning to safe levels.

---

## 5. Engineering & Deployment Recommendation

The 2-step hysteresis filter presents a classic engineering trade-off:

1. **When to use Raw Mode:** In high-density, high-value nursery or fry ponds where any oxygen dip—no matter how brief—threatens mortality, operators should use **Raw Mode** to guarantee the maximum **91.18% event detection rate** and full **93.1-minute lead time**.
2. **When to use Hysteresis Mode:** In grow-out ponds where operators experience severe alert fatigue from automated SMS/siren alarms, **Hysteresis Mode** reduces false alarm episodes by **32.9%** while maintaining an **86.76% event detection rate** and **85.9 minutes of advance notice**.

> **Documentation Rule Applied:** Previous claims stating that hysteresis reduces false alarms "without compromising safety" have been corrected across all project records. The trade-off is now reported transparently.
