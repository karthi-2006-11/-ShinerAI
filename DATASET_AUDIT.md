# Dataset Audit Report: FWI Continuous Water Quality Monitoring Dataset
**Project:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Phase:** Phase 1 — Environment Setup & Dataset Understanding  
**Status:** Completed  
**Audit Date:** October 2026  

---

## 1. Dataset Overview

The dataset audited in Phase 1 originates from the **Fish Welfare Initiative (FWI) Continuous Water Quality Monitoring Campaign** conducted in commercial aquaculture ponds located in Eluru, Andhra Pradesh, India. The campaign was designed to monitor water quality dynamically across multiple ponds in tandem with satellite imagery passes.

Continuous in-situ monitoring probes were deployed to record high-resolution aquatic parameters at approximately 15-minute intervals. The dataset contains 17 individual pond time-series CSV files (identified by anonymized IDs of the form `ara2_xxxxxxxx`), alongside supplementary metadata and comparison files (handheld ProDSS meter readings, key event logs, and quality control flags).

Phase 1 focuses strictly on understanding, verifying, and auditing the continuous time-series data without altering the raw files, modifying sensor values, interpolating gaps, or training predictive models.

---

## 2. Number of Pond Files

- **Total CSV Files in `data/raw/csv/`:** 22 files
- **Pond Time-Series Files Detected:** 17 files
- **Supplementary / Metadata Files Detected:** 5 files
  1. `Continuous_Monitor_vs_ProDSS_Comparison.csv`
  2. `Key_Events.csv`
  3. `ProDSS_and_Photometer.csv`
  4. `ProDSS_vs_Continuous_Monitors_Comparison.csv`
  5. `QC_Flags.csv`

All 17 pond time-series files conform to the naming pattern `ara2_*.csv` and correspond directly to the 17 continuous monitoring devices documented in `Key_Events.csv`.

---

## 3. Total Number of Observations

- **Total Raw Observations:** 72,750 rows
- **Minimum Rows per Pond:** 2,298 rows (`ara2_148f1633`)
- **Maximum Rows per Pond:** 5,592 rows (`ara2_29660d32`)
- **Median Rows per Pond:** 4,436 rows
- **Total Unique Timestamps across Ponds:** 72,492 unique timestamps
- **Duplicate Timestamps Detected:** 258 excess rows beyond first occurrences via `keep='first'` (occurring in `ara2_3ed4df8e` with 224 excess rows and `ara2_6ca69422` with 34 excess rows; representing 512 total records participating in timestamp collisions: 444 in `ara2_3ed4df8e` and 68 in `ara2_6ca69422`)

---

## 4. Date Range & Temporal Coverage

- **Earliest Timestamp:** 2025-11-28 21:30:00 IST (`ara2_29660d32`)
- **Latest Timestamp:** 2026-01-30 23:45:39 IST (`ara2_3eab831e`)
- **Overall Time Span:** 63.09 days (~9 weeks)
- **Individual Pond Durations:**
  - Ponds deployed in late November 2025 (`ara2_29660d32`, `ara2_3b2f3973`, `ara2_3eab831e`) cover 61 to 63 days.
  - Ponds deployed in mid-December 2025 cover ~48 days.
  - Later deployments or early harvests cover 25 to 27 days (`ara2_858b914c`: uninstalled Dec 24 due to harvest; `ara2_148f1633`: installed Jan 4).

---

## 5. Available Parameters

Each pond time-series CSV file provides the following parameters:

| Parameter Name in Raw CSV | Standardized Identifier | Data Type | Units | Description |
|---|---|---|---|---|
| `Date/Time (IST)` | `timestamp` / `Timestamp` | datetime64[ns] | YYYY-MM-DD HH:MM:SS | Local timestamp in Indian Standard Time (UTC+05:30) |
| `DO (mg/L)` | `do_mg_l` | float64 | mg/L (milligrams per liter) | In-situ Dissolved Oxygen concentration |
| `pH` | `ph` | float64 | pH units (0–14 scale) | Aquatic acidity/alkalinity measurement |
| `Temperature (°C)` | `temperature_c` | float64 | degrees Celsius (°C) | Water temperature |
| `QC_Flag_DateTime` | `qc_flag_datetime` | string / object | Text flags | Quality control labels relating to timestamp continuity |
| `QC_Flag_DO` | `qc_flag_do` | string / object | Text flags | Quality control labels relating to DO anomalies |
| `QC_Flag_pH` | `qc_flag_ph` | string / object | Text flags | Quality control labels relating to pH anomalies |

---

## 6. Sampling Interval Analysis

The dataset documentation specifies an intended nominal sampling interval of **15 minutes**.

- **Expected Sampling Interval:** 15 minutes (`00:15:00`)
- **Median Observed Interval:** Exactly 15 minutes (`00:15:00`) for all 17 ponds
- **Adherence Rate:** Across all 17 ponds, **97.37%** of recorded consecutive intervals fall within 14 to 16 minutes (close to 15 minutes).
- **Highest Adherence:** `ara2_3b2f3973` at 99.19%
- **Lowest Adherence:** `ara2_3ed4df8e` at 93.89% (impacted by sensor outage and duplicate timestamps)

---

## 7. Missing-Value Summary

An audit of null and NaN values across all 72,750 rows revealed:

- **Missing `DO (mg/L)` (NaN / null):** 0 (0.00%)
- **Missing `pH` (NaN / null):** 0 (0.00%)
- **Missing `Temperature (°C)` (NaN / null):** 0 (0.00%)
- **Missing `Date/Time (IST)`:** 0 (0.00%)

There are **zero missing values (NaN)** in the numeric sensor fields. Instead, probe failures, disconnects, resets, and firmware calibration events were recorded as **exact zero (0.0)** values.

---

## 8. Zero-Value Summary (Equipment Artifacts)

According to official dataset documentation, exact zero readings (`0.0`) in DO, pH, or Temperature do **not** represent true physiological or environmental measurements; they are equipment artifacts arising from startup sensor initialization, device resets, power failures, or temporary sensor removal.

- **Exact Zero DO Readings (`DO == 0.0`):** 131 observations (0.18% of all readings)
- **Exact Zero pH Readings (`pH == 0.0`):** 516 observations (0.71% of all readings)
- **Exact Zero Temperature Readings (`Temp == 0.0`):** 111 observations (0.15% of all readings)

### Distribution of Zero Artifacts:
- **January 8, 2026 Coordinated Event:** Nearly all monitoring devices recorded 1–2 zero readings on January 8–9, 2026 during a coordinated site-wide sensor calibration and firmware service.
- **`ara2_3ed4df8e` Multi-Day Outage:** Contributed 96 zero DO and 96 zero Temperature readings between January 5 and January 7 due to an extended device disconnect.
- **`ara2_ca187575` pH Sensor Fault:** Contributed 495 zero pH readings after being re-installed on January 25 following a partial fish harvest. The pH bulb failed to initialize, while the DO sensor continued operating normally.
- **`ara2_3eab831e`:** Contributed 21 zero DO readings during maintenance intervals.

> **Decision for Future Modeling:** In accordance with project instructions, exact zero values are identified, reported, and preserved in the raw data files. For the Phase 1 feasibility check, exact zeros were filtered out of valid biological oxygen calculations to prevent false-positive anomaly skew.

---

## 9. Quality Control (QC) Flag Summary

The dataset incorporates pre-annotated QC text labels matching conditional formatting from the original Excel logs. Multiple flags can co-occur in a single reading (separated by `; `).

### Dataset-Wide Frequency of Important Flags:

| QC Flag Name | Total Rows Tagged | Flag Meaning / Trigger |
|---|---|---|
| `DO_jump_>2` | 3,338 | DO changed by > 2 mg/L within ~15 minutes (sensor disturbance or rapid aeration) |
| `time_gap_>20min` | 3,116 | Interruption in continuous logging > 20 minutes (network/battery issue) |
| `DO_>10_before_noon` | 1,193 | Morning DO exceeds 10 mg/L (potential sensor noise or strong algal bloom) |
| `pH_out_of_range` | 541 | pH < 6.0 or pH > 10.0 (unusual for aquaculture ponds; 495 from `pH==0.0`) |
| `DO_<0.1` | 272 | Dissolved oxygen near zero (< 0.1 mg/L; extreme hypoxia or sensor out of water) |
| `pH_jump_>1` | 75 | Rapid pH shift > 1.0 unit within 15 minutes |

### Flag Overlap & Most Flagged Ponds:
- **Rows with Multiple Simultaneous Flags:** 353 rows (0.49%)
- **Ponds with the Most Flagged Measurements:**
  1. `ara2_ca187575`: 853 flagged rows (20.28% of pond data; heavily driven by pH bulb failure after harvest)
  2. `ara2_573a2826`: 765 flagged rows (17.08% of pond data; high frequency of abrupt DO jumps)
  3. `ara2_148f1633`: 364 flagged rows (15.84% of pond data; frequent time gaps)
  4. `ara2_3eab831e`: 813 flagged rows (14.56% of pond data; high gap count and DO jumps)
  5. `ara2_3b2f3973`: 715 flagged rows (14.50% of pond data; frequent morning DO > 10 mg/L)

---

## 10. Gap Analysis

When continuous recording is interrupted, the interval between consecutive timestamps exceeds the expected 15 minutes.

- **Total Gaps > 20 Minutes:** 1,645 instances across all 17 ponds
- **Average Gaps > 20 Min per Pond:** ~97 gaps
- **Pond with Fewest Gaps:** `ara2_858b914c` (31 gaps over 25 days)
- **Pond with Most Gaps:** `ara2_3eab831e` (209 gaps over 62 days)

### Unusually Large Gaps (> 1 Day):
Several ponds experienced substantial multi-day gaps corresponding to planned farm activities documented in `Key_Events.csv`:
1. **`ara2_3b2f3973`:** Max gap of **9 days, 17 hours, 45 minutes** (Device uninstalled on Dec 20 for fish harvest and reinstalled Dec 30).
2. **`ara2_0677080b`:** Max gap of **7 days, 15 hours, 15 minutes** (Device ceased pushing telemetry from Jan 18 to Jan 25 due to power/network failure).
3. **`ara2_6ca69422`:** Max gap of **3 days, 21 hours, 30 minutes** (Uninstalled Jan 19 for harvest; re-installed Jan 23).
4. **`ara2_ca187575`:** Max gap of **3 days, 1 hour, 15 minutes** (Uninstalled Jan 22 for harvest; re-installed Jan 25).
5. **`ara2_29660d32`:** Max gap of **2 days, 6 hours, 45 minutes** (Sensor maintenance following cleaning).

All other ponds had maximum recording gaps under 24 hours (typically 7 to 19 hours).

---

## 11. DO < 3.0 mg/L Feasibility Analysis

The future goal of this project is to build an early warning system that predicts whether dissolved oxygen will drop below **3.0 mg/L** within a **2-hour prediction horizon**. 

For Phase 1, we evaluated the empirical prevalence, pond distribution, and episode patterns of valid DO < 3.0 mg/L readings:

- **Total Valid DO Readings (< 3.0 mg/L):** **15,451 observations**
- **Percentage of Dataset Below 3.0 mg/L:** **21.24%**
- **Ponds Exhibiting DO < 3.0 mg/L:** **17 out of 17 ponds (100.0%)**
- **Range of Low-DO Prevalence per Pond:** From 3.83% (`ara2_0f143c64`) to 36.07% (`ara2_0677080b`)
- **Total Discrete Low-DO Episodes:** 1,189 distinct episodes across all ponds
- **Longest Continuous Low-DO Event:** 18.25 hours (73 consecutive readings) in `ara2_3b2f3973`
- **Typical Low-DO Diurnal Window:** Extended periods between 01:00 AM and 09:00 AM IST, reflecting natural nocturnal oxygen depletion before algal photosynthesis resumes.

### Feasibility Conclusion:
With over 15,000 positive instances distributed across every monitored pond, the 3.0 mg/L threshold represents a natural, recurring biological phenomenon with sufficient positive class balance (~21%) to train effective supervised machine learning algorithms.

---

## 12. Pond-by-Pond Summary Table

| Pond ID | Total Rows | Duration (Days) | Gaps >20m | Zero DO | Zero pH | Zero Temp | Mean DO (mg/L) | Mean pH | Mean Temp (°C) | Valid DO <3mg/L | % DO <3mg/L | Episodes <3mg/L | Max Low-DO Period |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ara2_0677080b` | 3,042 | 47.90 | 63 | 1 | 1 | 1 | 5.50 | 8.52 | 25.94 | 1,097 | 36.07% | 46 | 17.50 hrs |
| `ara2_0f143c64` | 4,470 | 48.01 | 94 | 1 | 1 | 1 | 9.13 | 8.67 | 26.07 | 171 | 3.83% | 22 | 7.75 hrs |
| `ara2_148f1633` | 2,298 | 26.65 | 98 | 1 | 1 | 1 | 7.52 | 8.43 | 26.29 | 210 | 9.14% | 35 | 7.00 hrs |
| `ara2_176528d3` | 4,149 | 47.90 | 158 | 1 | 1 | 1 | 5.75 | 8.65 | 25.66 | 1,044 | 25.17% | 91 | 16.00 hrs |
| `ara2_29660d32` | 5,592 | 63.09 | 138 | 2 | 1 | 1 | 5.36 | 8.13 | 25.91 | 1,947 | 34.83% | 82 | 15.75 hrs |
| `ara2_3b2f3973` | 4,930 | 61.78 | 40 | 4 | 5 | 3 | 8.98 | 8.27 | 26.15 | 709 | 14.39% | 40 | 18.25 hrs |
| `ara2_3eab831e` | 5,584 | 62.31 | 209 | 21 | 3 | 3 | 6.76 | 8.90 | 25.77 | 1,242 | 22.33% | 115 | 13.50 hrs |
| `ara2_3ed4df8e` | 4,665 | 48.30 | 61 | 96 | 3 | 96 | 6.82 | 8.29 | 25.39 | 1,255 | 27.47% | 68 | 14.25 hrs |
| `ara2_45e1cde5` | 4,414 | 47.93 | 134 | 1 | 1 | 1 | 7.53 | 8.39 | 25.94 | 555 | 12.58% | 65 | 14.25 hrs |
| `ara2_4f79fc8d` | 4,436 | 47.93 | 106 | 1 | 1 | 1 | 6.98 | 8.67 | 25.85 | 1,075 | 24.24% | 57 | 17.00 hrs |
| `ara2_573a2826` | 4,480 | 47.99 | 67 | 0 | 0 | 0 | 6.34 | 8.55 | 25.84 | 1,601 | 35.74% | 119 | 12.50 hrs |
| `ara2_6ca69422` | 3,776 | 44.76 | 73 | 1 | 1 | 1 | 7.66 | 8.45 | 26.21 | 706 | 18.70% | 65 | 15.50 hrs |
| `ara2_858b914c` | 2,338 | 25.75 | 31 | 0 | 0 | 0 | 7.33 | 8.18 | 26.22 | 279 | 11.93% | 47 | 11.50 hrs |
| `ara2_b3d128ac` | 4,435 | 47.94 | 118 | 0 | 1 | 0 | 5.61 | 8.75 | 25.85 | 1,354 | 30.53% | 61 | 17.50 hrs |
| `ara2_c9cdacda` | 5,477 | 58.45 | 70 | 0 | 0 | 0 | 8.33 | 8.31 | 26.12 | 378 | 6.90% | 55 | 10.00 hrs |
| `ara2_ca187575` | 4,206 | 48.00 | 92 | 0 | 495 | 0 | 7.02 | 7.53 | 26.19 | 795 | 18.90% | 64 | 12.00 hrs |
| `ara2_d52ddb31` | 4,458 | 47.94 | 93 | 1 | 1 | 1 | 6.92 | 8.46 | 26.31 | 1,033 | 23.18% | 89 | 16.25 hrs |
| **Combined** | **72,750** | **63.09** | **1,645** | **131** | **516** | **111** | **7.04** | **8.43** | **25.96** | **15,451** | **21.24%** | **1,189** | **18.25 hrs** |

---

## 13. Important Data-Quality Limitations

1. **Equipment Zero Artifacts:** Sensor resets and outages generate exact `0.0` readings (e.g., in `ara2_3ed4df8e` and `ara2_ca187575`). These must not be treated as real water measurements.
2. **Duplicate Timestamps:** 258 excess duplicate rows exist (`224` in `ara2_3ed4df8e`, `34` in `ara2_6ca69422`), involving 512 total colliding records (444 and 68 rows respectively) with conflicting sensor values.
3. **Multi-Day Gaps from Farm Operations:** Devices were temporarily uninstalled during fish harvests or experienced communication outages (up to 9.7 days in `ara2_3b2f3973`).
4. **Sensor Disturbance and Rapid Jumps:** 3,338 readings exhibit DO jumps > 2 mg/L in 15 minutes, often linked to aerator activation, probe cleaning, or probe movement.
5. **Persistent Sensor Failure:** In pond `ara2_ca187575`, the pH probe failed completely upon re-installation on January 25, recording 495 consecutive `0.0` pH values while DO continued operating.

---

## 14. Dataset Suitability for Phase 2

**Verdict: SUITABLE FOR PHASE 2 (WITH MANDATORY DATA CLEANING)**

The dataset is well-suited for developing an early warning predictive model because:
1. **Sufficient Sample Size:** 72,750 observations across 17 distinct aquaculture ponds provide rich real-world behavioral variety.
2. **Consistent Sampling Cadence:** 97.37% of readings follow the 15-minute expected frequency, enabling structured lag features and rolling aggregations.
3. **Balanced Event Prevalence:** Valid DO < 3.0 mg/L events occur across all 17 ponds (21.24% prevalence), offering enough minority-class positive examples without extreme imbalance.

---

## 15. Issues to Resolve Before Label Generation

Before generating prediction labels (`DO < 3.0 mg/L within next 2 hours`), the following data cleaning steps must be implemented:

1. **Filter Equipment Artifact Zeros:** Exclude `DO == 0.0`, `pH == 0.0`, and `Temperature == 0.0` from training and target evaluation.
2. **Deduplicate Timestamps:** Implement a deterministic deduplication strategy for `ara2_3ed4df8e` and `ara2_6ca69422` (e.g., retaining the latest entry or averaging identical timestamps).
3. **Respect Boundary Gaps in Rolling Windows:** When computing future 2-hour labels and historical lag features, do not allow rolling windows or forward labels to span across gaps > 30 minutes. Gaps must break the time-series sequence to prevent data leakage across outages.
4. **Isolate Probe Cleaning Periods:** Inspect `Key_Events.csv` cleaning dates to ensure probe cleaning events do not introduce artificial spike labels.
