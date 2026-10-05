# Independent External Dataset Audit: Oman Nile Tilapia Aquaculture

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Document:** External Dataset Audit Certification  
**Dataset Name:** Dissolved Oxygen Forecasting Dataset for Nile Tilapia Aquaculture in Oman  
**Source Repository:** [https://github.com/AhmedTheNetCoder/DO-Forecasting-Tilapia-Dataset](https://github.com/AhmedTheNetCoder/DO-Forecasting-Tilapia-Dataset)  
**Academic Citation:** Al-Khaldi, A.M.; Dhandapani, R.; Al-Badri, M.A. *Sensors* 2026, 26, 4242. DOI: [10.3390/s26134242](https://doi.org/10.3390/s26134242)  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Audit Date:** October 2026  

---

## 1. Executive Summary & Purpose

In accordance with mentor review requirements for **objective external validation**, ShinerAI identified and independently audited a publicly available, peer-reviewed aquaculture water quality monitoring dataset from North Al Sharqiyah, Oman. 

The purpose of this external audit is to establish an uncompromised baseline for evaluating whether the **frozen ShinerAI XGBoost Config C model** transfers to an independent geographical, biological, and sensor environment without any retraining, parameter tuning, or threshold modification.

---

## 2. 18-Point Dataset Audit Verification

| # | Audit Item | Findings & Verification | Integrity Status |
|---|---|---|---|
| **1** | **File Names** | `raw_readings_new.csv` (live validation raw)<br>`raw_readings.csv` (offline open pond raw)<br>`aggregated_data.csv` (5-min aggregated)<br>`validation_log_new.csv` (forecast validation log)<br>`event_logs_new.csv` (operational events)<br>`feature_importance.csv` (author feature importances)<br>`calibration_do.json`, `calibration_ph.json`, `calibration_temp.json`<br>`data_dictionary.md`, `README.md`, `LICENSE` | Verified complete matching upstream repository |
| **2** | **Row Counts** | `raw_readings_new.csv`: **102,670 rows**<br>`raw_readings.csv`: **41,663 rows**<br>`aggregated_data.csv`: **897 rows**<br>`validation_log_new.csv`: **9,472 rows**<br>`event_logs_new.csv`: **69 rows** | 100% row conservation verified |
| **3** | **Date Range** | Primary Live Validation: **2026-04-01 13:57:04.226656** to **2026-04-10 07:49:34.631639** (8.74 continuous days)<br>Offline Testing: **2026-03-26 19:56:13.394171** to **2026-04-27 22:15:35.227306** | Strict chronological bounds established |
| **4** | **Timestamp Format** | ISO 8601 strings with microsecond resolution: `YYYY-MM-DDTHH:MM:SS.ffffff` | Fully parseable via `pd.to_datetime` |
| **5** | **Sampling Interval** | High-frequency telemetry: mean interval = **7.36 seconds**; median = **6.56 seconds**; 75th percentile = **6.66 seconds**; standard deviation = 224.78s (due to 9 discrete network gaps > 60s) | Conforms to documented IoT hardware specifications |
| **6** | **Duplicate Timestamps** | `raw_readings_new.csv`: **0 duplicates** (102,670 unique timestamps)<br>`raw_readings.csv`: 541 duplicates (deduplicated during audit) | Deduplication verified |
| **7** | **Missing Timestamps** | **0 null timestamps** across all audited files | Zero missing values |
| **8** | **Missing DO Values** | **0 NaN / null entries** in the raw sensor streams | Zero missing values |
| **9** | **Invalid DO Values** | Negative DO values: **0**.<br>Exact zero readings: **2 readings** in `raw_readings_new.csv` at `2026-04-09 11:27:57` and `11:28:58` (representing momentary sensor disconnect/dropout); **7 readings** in `raw_readings.csv`. Treated as sensor dropouts (`np.nan`). | Verified and handled |
| **10** | **DO Minimum** | Raw live stream minimum: **0.000 mg/L** (momentary dropout); valid sensor telemetry minimum: **6.076 mg/L**; resampled 15-minute minimum: **6.076 mg/L** | Verified non-hypoxic throughout |
| **11** | **DO Maximum** | Raw live stream maximum: **14.138 mg/L**; 15-minute resampled maximum: **12.254 mg/L** | Verified physically plausible |
| **12** | **Temperature Availability** | Available for 100% of raw rows (`temp_c`). Mean temperature = **26.63°C** (min 23.56°C, max 29.81°C) | Fully available |
| **13** | **pH Availability** | Available for 100% of raw rows (`ph`). Mean pH = **8.32** (min 7.82, max 8.91) | Fully available |
| **14** | **Usable 15-min Observations** | Resampling the 8.74-day continuous live monitoring period to a fixed 15-minute grid produces **841 grid intervals**, of which **761 intervals** contain active raw sensor telemetry (coverage = 90.49%) | Documented |
| **15** | **Usable 2-hour Histories** | **749 intervals** possess a complete, uninterrupted sequence of 8 historical 15-minute lags (`do_t_minus_15` through `do_t_minus_120`) | Documented |
| **16** | **Eligible Prediction Points** | **742 intervals** satisfy all three eligibility criteria:<br>1. Operational condition: `current_do >= 3.0 mg/L`<br>2. History completeness: all 8 past lags available<br>3. Ground-truth completeness: all 8 future 2-hour intervals available | Strict invariant enforcement |
| **17** | **SAFE Labels (Class 0)** | **742 intervals (100.0%)** remain $\ge 3.0\text{ mg/L}$ across the entire 2-hour forward horizon | Verified ground truth |
| **18** | **AT_RISK Labels (Class 1)** | **0 intervals (0.0%)** fall below $3.0\text{ mg/L}$ within the 2-hour horizon | Verified ground truth |

---

## 3. Critical Scientific Finding on Ground-Truth Class Distribution

The audit reveals a fundamental environmental characteristic of the external Oman dataset:
- In the controlled live validation experiment, the 180L tilapia tank was equipped with continuous aeration to maintain healthy fish welfare for Nile tilapia (*Oreochromis niloticus*).
- Dissolved oxygen concentrations consistently fluctuated between **6.08 mg/L and 12.25 mg/L**, with a mean of **7.78 mg/L**.
- **Not a single true low-DO event ($< 3.0\text{ mg/L}$) occurred during the entire 8.74-day live validation period.**

### Scientific Consequences for Metric Calculation:
1. **Specificity (True Negative Rate) and Accuracy** are rigorously and meaningfully evaluable on all 742 true negative instances.
2. **Recall (Sensitivity)** is mathematically undefined ($0/0$) due to the absence of true positive events in the ground truth.
3. **Precision (PPV)** is evaluable: $0 / (0 + \text{FP})$.
4. **ROC-AUC and PR-AUC** are mathematically undefined for a single-class dataset (scikit-learn raises an `UndefinedMetricWarning` / `ValueError` because rank-order discrimination requires both positive and negative examples).

Under strict scientific guidelines, ShinerAI **refuses to fabricate positive samples** or adjust the operational threshold merely to manufacture artificial metrics. The audit confirms this dataset provides a rigorous test of **model false-alarm rate (Specificity)** under stable, well-aerated aquaculture conditions.

---

## 4. Independent External Dataset 2 Audit: Nigeria Aquaponics (`IoTPond10.csv`)

In October 2026, ShinerAI audited a secondary candidate external dataset: **Sensor-Based Aquaponics Fish Pond Datasets** (`IoTPond10.csv`) from the High Performance Intelligent Computing (HiPIC) lab at the University of Nigeria, Nsukka (Lacuna Fund; Kaggle).

### Key Audit Findings & NO-GO Determination:
1. **File Metadata:** 620 raw rows, 11 columns, SHA256: `7cd9f8b98a6edf6f2111046f35b19a1304f840c4eaef0b182b5b8922cd601690`.
2. **Structural Bifurcation:** The file contains only 310 unique sensor readings (Rows 0–309, with `CET` timestamps); Rows 310–619 are an exact 100% duplicate copy pasted with date-only strings (`D/M/YYYY`) to pair with catfish growth measurements.
3. **Severe Hardware Artifacts:**
   - 20 occurrences of `-127.0°C` (Dallas DS18B20 hardware disconnect code).
   - 338 rows (54.5%) of `0.00 mg/L` DO dropouts.
   - 9 rows with `inf` ammonia (firmware division-by-zero error).
   - 316 rows with negative turbidity (uncalibrated zero offset down to -97 NTU).
   - 28 rows with pH outside physical 0–14 scale (-1.98 to 15.45).
4. **DO Chattering:** DO oscillates between 0.00 mg/L and 21.65–32.41 mg/L within 20 to 60 seconds, representing electronic noise rather than biological hypoxia.
5. **Temporal Infeasibility:** ShinerAI requires a 4-hour continuous window (2h lookback + 2h lookahead = 240 minutes). The longest continuous session in `IoTPond10.csv` is only 174.6 minutes (June 25, 2021). The number of eligible evaluation samples is **strictly zero (0 samples, 0.0%)**.
6. **Scientific Decision:** **DEFINITIVE NO-GO**. In accordance with scientific integrity guidelines, ShinerAI strictly rejects synthetic temporal interpolation or fabricating artificial evaluation samples. Full audit report documented in [`docs/EXTERNAL_NIGERIA_DATASET_AUDIT.md`](../../docs/EXTERNAL_NIGERIA_DATASET_AUDIT.md).

