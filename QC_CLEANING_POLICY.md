# Quality Control & Data Cleaning Policy
**Project:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Phase:** Phase 2 — Data Cleaning & Supervised Label Generation  
**Policy Version:** 2.0 (Authoritative)  
**Applicability:** FWI Continuous Water Quality Monitoring Dataset (`data/raw/csv/`)  

---

## 1. Principles of Data Governance & Hygiene

1. **Raw Data Invariance:** Original raw CSV files in `data/raw/csv/` are **read-only**. They must never be overwritten, modified, trimmed, or renamed.
2. **Explicit Data Quality Accounting:** No rows are silently dropped or removed without cataloging. All 72,750 original records are preserved in `data/processed/cleaned_pond_data.csv` with their designated `data_quality_status` and exclusion rationale.
3. **No Blind Imputation:** Missing measurements and gaps are **not** artificially fabricated, smoothed, or linearly interpolated. Time-series continuity is enforced via segment partitioning.
4. **Preservation of Contextual QC Flags:** The original dataset QC flags (`QC_Flag_DateTime`, `QC_Flag_DO`, `QC_Flag_pH`) are preserved in both the cleaned time-series dataset and the final ML-ready dataset.
5. **Separation of History and Target (Zero Data Leakage):** Historical input features strictly represent measurements at or prior to the prediction time $T$. Future observations in $(T, T + 2\text{ hours}]$ are strictly quarantined for ground-truth label generation.

---

## 2. Data Quality Status Categories

In `data/processed/cleaned_pond_data.csv`, every record is assigned a definitive `data_quality_status`:

| Status Value | Meaning | Action in Feature & Label Pipeline | Count | % of Raw Data |
|---|---|---|---|---|
| `usable` | Sensor readings are valid, positive numbers with unique timestamps. | Eligible for segment grouping, historical feature windows, and target evaluation. | 71,609 | 98.43% |
| `excluded_sensor_artifact` | Exact zero reading (`0.0`) in DO, pH, or Temperature indicating sensor reset/power cut/probe initialization. | Excluded from feature inputs and target evaluations; cannot serve as prediction time $T$. | 629 | 0.86% |
| `excluded_duplicate` | Identical timestamp shared by multiple rows with conflicting measurements. | Excluded from feature inputs and window construction to prevent ambiguous time alignment. | 512 | 0.70% |

*(Note: Several rows exhibited dual issues—e.g. conflicting duplicate timestamps containing zero sensor artifacts; each is classified under its primary exclusion rule in the mutually exclusive hierarchy).*

### Complete Row Accounting (Zero Discrepancy)
The full dataset transitions from 72,750 raw observations to 41,277 ML-ready examples according to [`results/reports/phase2_row_accounting.csv`](file:///D:/FISH/results/reports/phase2_row_accounting.csv):

| Accounting Category | Count | % of Raw Data | Stage | Rationale |
|---|---|---|---|---|
| **Raw Ingested Observations** | **72,750** | **100.00%** | Raw Ingestion | Total continuous telemetry across 17 ponds. |
| **Excluded: Sensor Artifacts (Zeros)** | **629** | **0.86%** | Cleaning | Equipment reset/failure zeros (`DO==0`, `pH==0`, `Temp==0`). |
| **Excluded: Conflicting Duplicate Timestamps** | **512** | **0.70%** | Cleaning | All colliding records with conflicting measurements. |
| *Usable Baseline Observations* | *71,609* | *98.43%* | Baseline Filter | Valid unique positive sensor readings. |
| **Excluded: Current DO Already < 3.0 mg/L** | **15,234** | **20.94%** | Windowing | Pond is already in hypoxia; cannot serve as early-warning candidate. |
| **Excluded: Insufficient Past History (<2h)** | **8,700** | **11.96%** | Windowing | Fewer than 8 prior readings in contiguous segment. |
| **Excluded: Insufficient Future Coverage** | **6,398** | **8.79%** | Windowing | Monitoring stopped before complete 2-hour horizon ($T+120\text{m}$) without low DO. |
| **Final ML-Ready: SAFE (`target = 0`)** | **36,101** | **49.62%** | Final Dataset | Maintained DO $\ge 3.0$ mg/L continuously across complete next 2 hours. |
| **Final ML-Ready: AT_RISK (`target = 1`)** | **5,176** | **7.11%** | Final Dataset | Dropped below 3.0 mg/L within next 2 hours. |
| **Total Accounted Rows** | **72,750** | **100.00%** | Reconciliation | **Sum: 629 + 512 + 15,234 + 8,700 + 6,398 + 36,101 + 5,176 = 72,750.** |
| **Unexplained Discrepancy** | **0** | **0.00%** | Reconciliation | **Strictly zero unexplained rows.** |

---

## 3. Detailed Treatment of Specific Data Quality Issues

### 3.1. Exact Zero Sensor Values (`DO == 0.0`, `pH == 0.0`, `Temperature == 0.0`)
- **What the issue is:** The dataset documentation explicitly identifies exact zero values as equipment artifacts caused by probe startup initialization, microcontroller resets, or telemetry outages (e.g., site-wide calibration on January 8, 2026; multi-day power outage in `ara2_3ed4df8e`; pH bulb initialization failure after harvest in `ara2_ca187575`).
- **Policy Decision:** **Excluded from ML inputs and target evaluation.**
- **Rationale:** An exact `0.0 mg/L DO` or `0.0 pH` does not reflect real pond physiology. Treating an equipment reset zero as a biological drop would generate false-positive early warnings, while using zero as an input feature would distort model feature scaling.
- **Original QC Flag Preserved:** Yes. Original flags and raw zero values are preserved in `cleaned_pond_data.csv`.

### 3.2. Duplicate Timestamps & Resolution of Discrepancy
- **What the issue is:** Ponds `ara2_3ed4df8e` (220 collision timestamps, 444 rows) and `ara2_6ca69422` (34 collision timestamps, 68 rows) contain multiple records with identical timestamps.
- **Discrepancy Resolution:** Phase 1 reported **258 duplicate rows** using `keep='first'` (which counts only the excess redundant records: $224 + 34 = 258$). Phase 2 evaluated **all 512 rows involved in timestamp collisions** ($444 + 68 = 512$).
- **Policy Decision:** **All 512 colliding records are quarantined under `excluded_duplicate`.**
- **Rationale:** Detailed inspection confirmed that colliding rows possess conflicting sensor values (e.g. at 2025-12-23 09:30, one record logged DO = 2.16 mg/L while the other logged DO = 2.57 mg/L). Because the dataset provides no objective ground truth to decide which sensor packet is valid, selecting one arbitrarily would inject unverified telemetry.
- **Original QC Flag Preserved:** Yes. All 512 colliding duplicate rows are preserved in `cleaned_pond_data.csv`.

### 3.3. Time Gaps > 20 Minutes & Segment Partitioning
- **What the issue is:** Expected monitoring interval is 15 minutes. Gaps $> 20$ minutes occur due to communication packet loss, sensor uninstallation for harvest, battery swaps, or cleaning.
- **Policy Decision:** **Do not interpolate.** Instead, partition the usable time series into **contiguous segments** where consecutive readings are $\le 20$ minutes apart.
- **Rationale:** A contiguous segment guarantees that every consecutive time step is within normal logging cadence. Rolling windows and historical lags ($t-15\text{m}, \dots, t-120\text{m}$) are strictly restricted to within the same segment. This prevents constructing false historical baselines across outages.
- **Original QC Flag Preserved:** Yes (`time_gap_>20min`).

### 3.4. Large Dissolved Oxygen Jumps (`DO_jump_>2`)
- **What the issue is:** The DO reading shifted by $> 2.0\text{ mg/L}$ over approximately 15 minutes (3,338 occurrences).
- **Policy Decision:** **Retained as usable data.**
- **Rationale:** In commercial aquaculture, sudden DO spikes and drops can be genuine physical events resulting from emergency paddlewheel aerator activation, sudden cloud cover, or heavy feeding activity. They should not be censored unless accompanied by an equipment reset zero.
- **Original QC Flag Preserved:** Yes (`DO_jump_>2`).

### 3.5. Dissolved Oxygen < 0.1 mg/L (`DO_<0.1`)
- **What the issue is:** Near-zero DO reading (272 occurrences).
- **Policy Decision:** **Retained as usable if $> 0.0\text{ mg/L}$.**
- **Rationale:** Severe hypoxia is a real biological phenomenon in poorly aerated or high-density ponds, particularly right before dawn. Provided the reading is not an exact `0.0` equipment artifact, it represents a valid crisis event.
- **Original QC Flag Preserved:** Yes (`DO_<0.1`).

### 3.6. DO > 10 mg/L Before Noon (`DO_>10_before_noon`)
- **What the issue is:** Very high morning DO readings (1,193 occurrences).
- **Policy Decision:** **Retained as usable data.**
- **Rationale:** In shallow, algae-rich tropical ponds, high sunlight combined with dense phytoplankton blooms can produce extreme supersaturation early in the day. These are real environmental features of the pond.
- **Original QC Flag Preserved:** Yes (`DO_>10_before_noon`).

### 3.7. pH Outside 6.0–10.0 (`pH_out_of_range`) & pH Jumps (`pH_jump_>1`)
- **What the issue is:** pH reading $< 6.0$ or $> 10.0$ (541 occurrences, of which 495 were exact `0.0` artifacts in `ara2_ca187575`) or rapid pH shifts $> 1.0$ unit in 15 minutes (75 occurrences).
- **Policy Decision:** 
  - If $\text{pH} == 0.0$: excluded under the sensor artifact rule.
  - If $\text{pH} > 0.0$: retained as usable environmental context unless accompanied by other invalidating sensor flags.
- **Original QC Flag Preserved:** Yes (`pH_out_of_range`, `pH_jump_>1`).

### 3.8. Probe Cleaning Events & Satellite Flyovers
- **What the issue is:** Monitors were cleaned approximately every 5 days before satellite flyovers. Probe handling can cause transient 15-minute sensor disturbances.
- **Policy Decision:** Retained in cleaned time series with QC flags preserved. The contiguous segmentation ensures that any prolonged cleaning outage $> 20$ min breaks the sequence cleanly.

---

## 4. Supervised Target Label Generation Rules

### 4.1. Formal Problem Definition
> *"Given an individual fish pond currently at a SAFE dissolved oxygen level ($\text{DO} \ge 3.0\text{ mg/L}$), will its dissolved oxygen fall below $3.0\text{ mg/L}$ at any point during the subsequent 2 hours?"*

### 4.2. Candidate Row Eligibility
A observation at timestamp $T$ is an eligible prediction candidate if and only if:
1. `data_quality_status == 'usable'` (valid sensors, no zero artifacts, no duplicate timestamps).
2. `current_do >= 3.0 mg/L`. Rows where current DO is already $< 3.0\text{ mg/L}$ are **already in crisis**; labeling them as early warnings would defeat the purpose of an advance predictive system.
3. The past 2-hour window $[T - 2\text{ hours}, T]$ is fully populated with at least 8 prior contiguous readings in the same segment ($t-15\text{m}, t-30\text{m}, \dots, t-120\text{m}$).

### 4.3. Target Interval Definition
- **Prediction Time:** $T$
- **Target Interval:** $(T, T + 2\text{ hours}]$ (strictly future observations)

### 4.4. Binary Label Logic
- **`AT_RISK = 1`:** Assigned if at least one valid measurement within $(T, T + 2\text{ hours}]$ records $\text{DO} < 3.0\text{ mg/L}$. AT_RISK examples remain valid even if sensor telemetry ceases later in the window, because the hypoxia event has been directly observed.
- **`SAFE = 0`:** Assigned if and only if:
  1. No valid measurement in $(T, T + 2\text{ hours}]$ records $\text{DO} < 3.0\text{ mg/L}$, **AND**
  2. The future monitoring window covers the **complete 2-hour interval**: the latest observation extends through approximately $T + 120\text{ minutes}$ ($\ge T + 119\text{m}$, allowing a small documented 60-second tolerance for second-level logger clock drift) with at least 8 recorded readings.

### 4.5. Insufficient Future Coverage Exclusion
If no low-DO reading is observed, but the sensor went offline before the full 2-hour horizon was completed (e.g. sensor outage, device uninstallation, segment break), we **cannot** verify whether DO remained safe during the unmonitored window. Assigning `SAFE` would introduce false-negative label noise. Therefore, these rows are categorized as:
```text
label_status = insufficient_future_coverage
```
and strictly excluded from `ml_ready_dataset.csv`.

---

## 5. Provisional Threshold Caveat

The threshold of **3.0 mg/L Dissolved Oxygen** is adopted as a **provisional project standard** grounded in general warm-water teleost aquaculture guidelines (e.g., *Labeo rohita*, *Catla catla*, *Pangasianodon hypophthalmus*). It is not claimed to be universally optimal for all aquatic species or developmental stages. Species-specific physiological justifications must be validated before formal academic publication.
