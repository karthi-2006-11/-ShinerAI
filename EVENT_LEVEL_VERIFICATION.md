# ShinerAI: Independent Event-Level Early-Warning Verification Report

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Status:** Certified Independent Audit  
**Date:** October 2026  
**Audited Artifacts:**
- Processed Dataset: `data/processed/ml_ready_dataset.csv`
- Continuous Telemetry: `data/processed/cleaned_pond_data.csv`
- Frozen Model: `models/xgboost_config_c.joblib`
- Split Policy: 80/20 chronological partition with 2.0-hour purge gap ($N_{\text{test}} = 8,261$)

---

## 1. Executive Summary

This independent audit rigorously evaluates the event-level early-warning claims made in the recent ShinerAI research reports. Recomputing from the underlying continuous sensor measurements and held-out test predictions reveals:

1. **Event Detection Rate Verified at 91.18% (124 / 136 episodes):**
   - The reported detection rate of **91.18% (124 detected, 12 missed out of 136 impending episodes)** is **INDEPENDENTLY VERIFIED**.
   - Crucially, all 124 detected episodes had their earliest alert triggered **strictly BEFORE the physical low-DO onset** ($\text{DO} < 3.0\text{ mg/L}$). Exactly 0 alerts occurred after event onset.
2. **Lead Time Refinement (101.7 min $\to$ 93.1 min):**
   - The previously reported mean lead time of **101.7 minutes** was calculated relative to the *end* of the impending target=1 run ($T_{\text{end}} + 15\text{m}$).
   - When strictly measured to the **first physical crossing below 3.0 mg/L**, the true advance warning lead time is:
     - **Mean Lead Time:** **93.1 minutes** ($\approx 1.55$ hours)
     - **Median Lead Time:** **120.0 minutes** (2.0 hours)
     - **Maximum Lead Time:** **120.0 minutes** (the theoretical 2-hour contract ceiling; the previous 345-minute maximum was an artifact of merging sub-events).
     - **Lead Time Distribution:** 91.1% of warned events provide $\ge 30$ min, 83.9% provide $\ge 60$ min, 70.2% provide $\ge 90$ min, and 52.4% provide full 120 min advance notice.

---

## 2. Event Definition Audit (Part 1A)

To avoid ambiguity, three distinct temporal concepts must be distinguished:

```text
       Prediction Time (T)             Impending Target=1 Block             Physical DO Crossing (<3.0)
       [T-120m ... T] (Past)       -------> (T, T+120m] (Lookahead) -------> Continuous Telemetry in Pond
```

1. **Prediction Time ($T$):** A candidate timestamp where current $\text{DO} \ge 3.0\text{ mg/L}$ and 2 hours of contiguous history exist.
2. **Impending Event Block ($T_{\text{start}}$ to $T_{\text{end}}$):** A contiguous sequence of candidate prediction rows where `target == 1` (DO will fall below 3.0 mg/L within the subsequent 2 hours).
3. **Physical Hypoxia Onset ($T_{\text{phys}}$):** The exact first observation in the pond's continuous telemetry where $\text{DO} < 3.0\text{ mg/L}$ following normoxic water quality.

### Invariant Discovery:
In `cleaned_pond_data.csv`, for **136 out of 136 impending episodes** in `test_df`, the observation immediately following $T_{\text{end}}$ has $\text{DO} < 3.0\text{ mg/L}$. Because any observation with $\text{DO} < 3.0$ is excluded from candidate prediction points, $T_{\text{end}}$ is always the final observation before the pond enters physical hypoxia.

---

## 3. Existing Methodology vs. Independent Physical Audit (Parts 1B & 1C)

### Existing Methodology:
- Grouped contiguous `target == 1` rows in `test_df` with inter-observation gap $\le 30$ minutes.
- Defined assumed crossing: $T_{\text{cross}} = T_{\text{end}} + 15\text{ minutes}$.
- Computed lead time: $\Delta t = T_{\text{cross}} - T_{\text{alert}}$.
- **Limitation:** In 20 of the 136 episodes, a pond experienced a brief nocturnal dip below 3.0 mg/L (e.g. 15–30 min), briefly rebounded to 3.1–3.4 mg/L, and then dipped again within 2 hours. Because the gap between candidate rows was $\le 30$ minutes, the existing script merged them into a single block of up to 21 rows (5+ hours), artificially extending the computed lead time up to 345 minutes.

### Independent Physical Telemetry Methodology:
- Audited the exact continuous telemetry in `data/processed/cleaned_pond_data.csv`.
- Located the exact timestamp $T_{\text{phys}}$ where $\text{DO} < 3.0\text{ mg/L}$ for the *first* time in each episode.
- Filtered model predictions strictly to those occurring:
  $$\text{prediction\_timestamp} < T_{\text{phys}} \quad \text{AND} \quad \text{prediction\_timestamp} \ge T_{\text{phys}} - 120\text{ minutes}$$
- Calculated true lead time:
  $$\text{Lead Time} = T_{\text{phys}} - T_{\text{first\_pre\_alert}}$$
- If no alert occurred before $T_{\text{phys}}$, the event is classified as **MISSED**. Alerts after $T_{\text{phys}}$ are strictly disallowed.

---

## 4. Verification Results & Comparison (Parts 1D, 1E, 1F)

### 4.1 Event Detection Rate: 91.18% Certified
Across the 136 impending hypoxia episodes in the test partition:

| Metric | Existing Report | Independent Physical Audit | Audit Status |
| :--- | :---: | :---: | :---: |
| **Total Impending Episodes** | 136 | 136 | **EXACT MATCH** |
| **Detected Episodes** | 124 | 124 | **EXACT MATCH** |
| **Missed Episodes** | 12 | 12 | **EXACT MATCH** |
| **Event Detection Rate** | **91.18%** | **91.18%** | **EXACT MATCH** |
| **Alerts Post-Onset** | 0 | 0 | **VERIFIED (0 post-onset alerts)** |

**Conclusion:** The **91.18% event detection rate is certified as completely accurate**. In all 124 detected episodes, the model triggered its alert well before physical oxygen depletion began.

### 4.2 Lead-Time Distribution Reconciliation
Comparing the lead times for the 124 detected episodes:

| Metric | Existing Report | Independent Physical Audit | Difference / Rationale |
| :--- | :---: | :---: | :--- |
| **Mean Lead Time** | 101.7 min | **93.1 min** | -8.6 min (corrected for sub-event merging) |
| **Median Lead Time** | 120.0 min | **120.0 min** | **EXACT MATCH** |
| **25th Percentile** | 60.0 min | **60.0 min** | **EXACT MATCH** |
| **75th Percentile** | 120.0 min | **120.0 min** | **EXACT MATCH** |
| **Minimum Lead Time** | 15.0 min | **15.0 min** | **EXACT MATCH** |
| **Maximum Lead Time** | 345.0 min | **120.0 min** | Capped at strict 2-hour forecasting contract |
| **$\ge$ 30 Minutes Lead Time** | 89.5% | **91.1%** | +1.6% (113 of 124 detected events) |
| **$\ge$ 60 Minutes Lead Time** | 81.5% | **83.9%** | +2.4% (104 of 124 detected events) |
| **$\ge$ 90 Minutes Lead Time** | 69.4% | **70.2%** | +0.8% (87 of 124 detected events) |
| **$\ge$ 120 Minutes Lead Time**| 55.6% | **52.4%** | -3.2% (65 of 124 detected events) |

---

## 5. Continuous Physical Telemetry Benchmark (All 201 Physical Crossings)

To go beyond the 136 candidate blocks in `test_df`, we also audited every single physical low-DO crossing in `cleaned_pond_data.csv` during the test monitoring period:
- **Total Physical Low-DO Crossings in Test Period:** 201
- **Crossings with $\ge 1$ Pre-Event Prediction Opportunities in `test_df`:** 164
- **Crossings with 0 Pre-Event Opportunities:** 37 (due to prior low DO $<3.0$, sensor reboot gaps $>20$ min, or occurring inside the 2-hour purge gap).
- **Detection Rate on Eligible Physical Crossings ($N=164$):**
  - Detected before onset: **152 / 164 (92.68%)**
  - Missed: **12 / 164 (7.32%)**
  - Mean lead time: **95.4 minutes** (median **120.0 minutes**)
- **Detection Rate on All Physical Crossings ($N=201$):**
  - Detected before onset: **152 / 201 (75.62%)**
  - Missed (including sensor offline/gap events): **49 / 201 (24.38%)**

---

## 6. Required Documentation Corrections

To ensure complete scientific defensibility for IEEE review:
1. **Report True Physical Lead Time:** Update mean lead time from 101.7 minutes to **93.1 minutes** (median 120.0 minutes).
2. **Correct Maximum Lead Time:** Correct max lead time from 345 minutes to **120 minutes**, enforcing the strict 2-hour forecasting contract ceiling.
3. **Transparent Reporting:** Present both the impending-episode detection rate (**91.18% across 136 episodes**) and the eligible physical crossing detection rate (**92.68% across 164 crossings**).
