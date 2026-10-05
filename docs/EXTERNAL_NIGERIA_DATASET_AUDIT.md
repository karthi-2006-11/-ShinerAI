# Independent External Dataset Audit: Nigeria Aquaponics IoT Fish Pond Dataset (`IoTPond10.csv`)

**Project:** ShinerAI — AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Document:** External Dataset Audit Certification  
**Dataset Name:** Sensor-Based Aquaponics Fish Pond Datasets — Pond 10 (`IoTPond10.csv`)  
**Lead Investigator / Authors:** Prof. Collins N. Udanor et al. (High Performance Intelligent Computing Lab / HiPIC, University of Nigeria, Nsukka)  
**Grant / Funder:** Lacuna Fund (AI for Agriculture and Food Security)  
**Kaggle Source:** `afolabisalawu/sensor-based-aquaponics-fish-pond-datasets`  
**License:** Open Database / Creative Commons  
**Audit Date:** October 2026  
**Auditor:** ShinerAI Research Team  
**Audit Mandate:** Feasibility and Scientific Integrity Audit for IEEE Peer Review  

---

## 1. Executive Summary & Audit Mandate

In response to mentor review feedback (*"Make a Independent external validation for this work"*), ShinerAI audited the public Nigerian aquaculture dataset (`IoTPond10.csv`) as a candidate for evaluating model transferability to genuine low-dissolved-oxygen conditions.

The active ShinerAI production model (`models/xgboost_config_c.joblib`, 11 features, 2-hour lookback, threshold $\tau = 0.50$, target DO $< 3.0\text{ mg/L}$ in $[T+15\text{m}, T+120\text{m}]$) is **strictly frozen**. In accordance with the scientific guidelines:
- **No model retraining is allowed.**
- **No threshold tuning is allowed.**
- **No feature modifications are allowed.**
- **No synthetic interpolation or imputation is permitted.**

### Definitive Audit Finding: **DEFINITIVE NO-GO**
The dataset is **fundamentally unsuitable** for evaluating the ShinerAI early-warning model. The dataset contains **0 eligible evaluation samples (0.0%)** that can satisfy ShinerAI's frozen 2-hour lookback and 2-hour lookahead contract without fabricating synthetic observations. Furthermore, over 54% of dissolved oxygen readings represent severe hardware disconnects ($0.00\text{ mg/L}$) and uncalibrated analog noise rather than genuine limnological hypoxia.

---

## 2. Dataset Identity & Source Information

* **Repository / Origin:** High Performance Intelligent Computing (HiPIC) Research Group, Department of Computer Science, University of Nigeria, Nsukka (UNN), Enugu State, Nigeria.
* **Target Species:** African Catfish (*Clarias gariepinus*).
* **Culture System:** High-density experimental indoor/covered recirculating aquaponics tanks.
* **Hardware Platform:** Microcontroller-based multi-sensor IoT node (analog probe interface, Dallas DS18B20 digital temperature sensor, ESP-based telemetry).
* **Acquisition Campaign:** June 2021 – December 2021.
* **Local Raw Archive:** `data/external/nigeria_iotpond/raw/IoTPond10.csv`.

---

## 3. Raw File Integrity & Cryptographic Hash

To guarantee zero accidental data corruption and strict reproducibility, the raw CSV file was preserved in the repository archive and cryptographically fingerprinted:

| File Property | Verified Audit Value | Status |
|---|---|:---:|
| **File Path** | `data/external/nigeria_iotpond/raw/IoTPond10.csv` | Immutable Raw Copy |
| **File Size** | **45,352 bytes** | Verified |
| **SHA256 Hash** | `7cd9f8b98a6edf6f2111046f35b19a1304f840c4eaef0b182b5b8922cd601690` | Exact Match |
| **Total Ingested Rows** | **620 rows** | Complete |
| **Total Ingested Columns** | **11 columns** | Complete |

---

## 4. Dataset Dimensions & Structural Bifurcation Discovery

An initial structural audit uncovered a critical, undocumented characteristic of `IoTPond10.csv`: **the file is split into two completely distinct halves**:

```text
Rows 0 to 309   (Part A, N = 310): Genuine IoT telemetry with second-level timestamps ('YYYY-MM-DD HH:MM:SS CET').
Rows 310 to 619 (Part B, N = 310): Exact duplicate copy of Part A sensor data with date-only strings ('D/M/YYYY').
```

### Empirical Proof of Duplicate Structure:
Comparing Part A (Rows 0–309) against Part B (Rows 310–619):
* `TEMPERATURE`: **310 / 310 identical (100.0%)**
* `DISOLVED OXYGEN`: **310 / 310 identical (100.0%)**
* `pH`: **310 / 310 identical (100.0%)**
* `TURBIDITY`: **310 / 310 identical (100.0%)**
* `NITRATE`: **310 / 310 identical (100.0%)**
* `Population`: **310 / 310 identical (100.0%)**
* `AMMONIA`: **301 / 310 identical (97.1%)** (residual 9 rows differ only by floating `inf`)

**Root Cause:** The data collectors duplicated the 310 sensor observations to pair them with 5 discrete fish growth sampling dates (`10/10/2021`, `24/10/2021`, `7/11/2021`, `21/11/2021`, `5/12/2021`) where catfish length ($31.7$ to $44.5\text{ cm}$) and weight ($263$ to $689\text{ g}$) were manually recorded. Consequently, the file contains only **310 unique physical sensor observations**, not 620.

---

## 5. Comprehensive Column Inventory & Statistical Profiling

Original terminology and spelling from the raw CSV have been preserved verbatim:

| Column Name | Raw Data Type | Missing Count | Unique Values | Minimum | Maximum | Mean | Median | Standard Dev. | Operational Evaluation |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `created_at` | `object (str)` | 0 | 315 | N/A | N/A | N/A | N/A | N/A | Bifurcated: 310 CET timestamps + 310 date-only strings |
| `entry_id` | `int64` | 0 | 620 | 1 | 620 | 310.5 | 310.5 | 179.12 | Sequential transmission counter |
| `TEMPERATURE` | `float64` | 0 | 57 | **-127.0°C** | 31.69°C | 22.16°C | 27.13°C | 27.27°C | 20 occurrences of -127.0°C hardware disconnect code |
| `TURBIDITY` | `int64` | 0 | 12 | **-97 NTU** | 100 NTU | 40.36 NTU | -9.0 NTU | 58.23 NTU | 316 negative values; uncalibrated analog ADC drift |
| `DISOLVED OXYGEN` | `float64` | 0 | 132 | **0.00 mg/L** | **32.41 mg/L** | 2.36 mg/L | **0.00 mg/L** | 4.40 mg/L | 338 rows (54.5%) are 0.00 mg/L; extreme chattering |
| `pH` | `float64` | 0 | 104 | **-1.98** | **15.45** | 5.66 | 6.04 | 2.10 | 28 values outside physical 0–14 pH scale |
| `AMMONIA` | `float64` | 0 | 127 | 0.0 mg/L | **inf** | N/A | 0.0 mg/L | N/A | 9 occurrences of `inf` (firmware division-by-zero) |
| `NITRATE` | `int64` | 0 | 176 | 0 mg/L | 2,473 mg/L | 311.56 mg/L | 193.0 mg/L | 391.18 mg/L | High baseline nitrate in aquaponics filter |
| `Population` | `int64` | 0 | 1 | 50 | 50 | 50.0 | 50.0 | 0.0 | Constant stocking density (50 fingerlings) |
| `Length` | `float64` | 0 | 19 | 13.45 cm | 44.55 cm | 27.21 cm | 24.11 cm | 13.05 cm | Bimodal fish growth sampling (Part A vs B) |
| `Weight` | `float64` | 0 | 19 | 27.60 g | 689.14 g | 282.57 g | 158.53 g | 281.11 g | Bimodal fish growth sampling (Part A vs B) |

---

## 6. Timestamp & Temporal Cadence Audit

Focusing on the 310 genuine timestamped observations (Part A, Rows 0–309):

* **Timezone:** Central European Time (`CET` / UTC+1).
* **Earliest Timestamp:** `2021-06-25 15:48:59 CET`
* **Latest Timestamp:** `2021-10-06 12:52:52 CET`
* **Calendar Duration:** **102 days, 21 hours, 3 minutes, 53 seconds** (102.88 days).
* **Chronological Monotonicity:** Strictly increasing (`is_monotonic_increasing == True`).
* **Duplicate Timestamps in Part A:** **0 duplicates** (all 310 rows have distinct seconds).

### Sampling Interval Distribution within Part A:
* **Minimum Sampling Interval:** **19.0 seconds**
* **Median Sampling Interval:** **20.0 seconds**
* **Mode:** **20.0 seconds**
* **25th Percentile:** **20.0 seconds**
* **75th Percentile:** **48.0 seconds**
* **95th Percentile:** **167.6 seconds**
* **Maximum Sampling Interval:** **3,101,659.0 seconds** (**35.90 days** / 861.6 hours)

### Active Monitoring Sessions by Date:
The 310 observations are severely fragmented across **13 discrete dates**:

| Date | Obs Count | Start Time (CET) | End Time (CET) | Session Duration | DO Min | DO Max | DO == 0 Count | DO < 3.0 Count | Temp -127°C |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2021-06-25** | **112** | 15:48:59 | 18:43:33 | **174.6 min (2.91 h)** | 0.00 | 21.65 | 66 | 87 | 2 |
| **2021-06-26** | 1 | 16:37:15 | 16:37:15 | 0.0 min (Instant) | 12.44 | 12.44 | 0 | 0 | 0 |
| **2021-06-27** | 1 | 00:05:12 | 00:05:12 | 0.0 min (Instant) | 0.00 | 0.00 | 1 | 1 | 0 |
| **2021-06-28** | 1 | 09:35:23 | 09:35:23 | 0.0 min (Instant) | 0.00 | 0.00 | 1 | 1 | 0 |
| **2021-06-29** | 3 | 17:26:25 | 18:43:12 | 76.8 min | 0.00 | 2.27 | 1 | 3 | 0 |
| **2021-06-30** | 1 | 01:09:29 | 01:09:29 | 0.0 min (Instant) | 0.00 | 0.00 | 1 | 1 | 0 |
| **2021-07-01** | 1 | 09:04:13 | 09:04:13 | 0.0 min (Instant) | 0.00 | 0.00 | 1 | 1 | 0 |
| **2021-07-07** | 26 | 15:15:34 | 15:24:32 | 9.0 min | 0.00 | 30.48 | 9 | 24 | 1 |
| **2021-08-12** | 96 | 12:58:51 | 13:36:45 | 37.9 min | 0.00 | 32.41 | 63 | 89 | 0 |
| **2021-08-22** | 14 | 21:25:00 | 21:30:41 | 5.7 min | 0.00 | 0.00 | 14 | 14 | 0 |
| **2021-09-24** | 4 | 13:31:02 | 13:32:46 | 1.7 min | 0.00 | 0.00 | 4 | 4 | 0 |
| **2021-10-01** | 6 | 11:28:43 | 11:31:32 | 2.8 min | 0.00 | 0.00 | 6 | 6 | 5 |
| **2021-10-06** | 44 | 12:34:22 | 12:52:52 | 18.5 min | 0.00 | 12.34 | 2 | 8 | 2 |

---

## 7. Sensor Quality Audit & Hardware Artifact Analysis

A forensic analysis of the physical sensor values revealed severe electronic, communication, and firmware artifacts:

### 1. Temperature Hardware Disconnect Code (`-127.0°C`):
* **Occurrences:** 20 across dataset (10 in Part A, 10 in Part B).
* **Engineering Ground Truth:** In Dallas Semiconductor / Maxim Integrated OneWire digital temperature sensors (e.g., DS18B20), the sensor returns an 85.0°C power-on reset state and a **-127.0°C return code** whenever the data bus fails, the pull-up resistor disconnects, or the device loses VCC/GND connection.
* **Classification:** **Definite hardware sensor disconnect artifact.**

### 2. Firmware Division-by-Zero (`AMMONIA = inf`):
* **Occurrences:** 9 rows.
* **Engineering Cause:** The analog gas/ion sensor calibration equation in microcontroller firmware contains a denominator calculation (e.g. $R_s / R_0$ or logarithmic curve fit) that divided by zero when the raw analog ADC returned 0 or 1023.
* **Classification:** **Definite firmware calculation artifact.**

### 3. Turbidity Uncalibrated Negative Offset (`TURBIDITY < 0 NTU`):
* **Occurrences:** 316 rows (51.0% of the entire dataset).
* **Range:** Reaches as low as **-97 NTU**.
* **Limnological Reality:** Physical turbidity cannot be negative. This indicates an uncalibrated baseline offset on the optical turbidity sensor amplifier.
* **Classification:** **Definite calibration offset error.**

### 4. pH Extreme Scale Violations (`pH < 0` or `pH > 14`):
* **Occurrences:** 28 rows (14 in Part A, 14 in Part B).
* **Range:** **-1.98 to +15.45**.
* **Limnological Reality:** In commercial aquaculture, water pH must remain between 6.5 and 8.5. Catfish experience acute mortality below pH 4.0. Readings of -1.98 and +15.45 violate the fundamental chemical pH scale ($[0, 14]$) and represent electrical open-circuit spikes on the glass probe.
* **Classification:** **Definite analog probe electrical fault.**

---

## 8. Dissolved Oxygen Quality, Distribution & Chattering Analysis

```text
Part A (N = 310) Dissolved Oxygen Statistics:
  Minimum:    0.0000 mg/L
  Maximum:   32.4110 mg/L
  Mean:       2.3576 mg/L
  Median:     0.0000 mg/L
  Std Dev:    4.4026 mg/L
  DO == 0.00: 169 observations (54.5%)
  DO < 3.00:  239 observations (77.1%)
  DO >= 3.00:  71 observations (22.9%)
```

### The "Chattering Dropout" Phenomenon:
In natural earthen ponds or aquaponics systems, dissolved oxygen depletion is governed by nocturnal biological oxygen demand (BOD) and respiration. Under Fickian diffusion, oxygen transitions smoothly over hours.

In `IoTPond10.csv`, DO exhibits violent high-frequency oscillation:
* **June 25, 2021:**
  * 17:30:59: $\text{DO} = \mathbf{13.56\text{ mg/L}}$
  * 17:31:19: $\text{DO} = \mathbf{0.00\text{ mg/L}}$ (plunges to zero in **20 seconds**)
  * 17:32:15: $\text{DO} = \mathbf{4.46\text{ mg/L}}$ (jumps in **56 seconds**)
  * 17:33:12: $\text{DO} = \mathbf{12.30\text{ mg/L}}$ (jumps in **57 seconds**)
  * 18:04:31: $\text{DO} = \mathbf{0.00\text{ mg/L}}$
  * 18:06:05: $\text{DO} = \mathbf{21.65\text{ mg/L}}$ (spikes in **94 seconds**)
  * 18:07:01: $\text{DO} = \mathbf{0.00\text{ mg/L}}$ (drops to zero in **56 seconds**)
* **August 12, 2021:**
  * 13:03:43: $\text{DO} = \mathbf{3.64\text{ mg/L}}$
  * 13:04:02: $\text{DO} = \mathbf{0.00\text{ mg/L}}$ (drops to zero in **19 seconds**)
  * 13:04:22: $\text{DO} = \mathbf{32.41\text{ mg/L}}$ (spikes to supersaturation in **20 seconds**)
  * 13:04:42: $\text{DO} = \mathbf{3.21\text{ mg/L}}$

### Empirical Conclusion on Low-DO Observations:
**The low-DO observations ($\text{DO} < 3.0\text{ mg/L}$) in this dataset do NOT represent genuine biological hypoxia.**  
They represent rapid hardware signal loss (dropouts to 0.00 mg/L) alternating with impossible supersaturation spikes (up to 32.41 mg/L) caused by faulty galvanic/optical membrane grounding or analog ADC float.

---

## 9. Low-DO Episode Identification

Applying ShinerAI's standard episode grouping rule (contiguous sequence of $\text{DO} < 3.0\text{ mg/L}$ separated by $> 30\text{ minutes}$):

* **Total Apparent Low-DO Clusters:** 10 clusters.
* **Sustained Real Hypoxia:** **0 episodes**.
* *Reason:* In every single cluster, either the readings are literal 0.00 mg/L hardware dropouts (e.g. August 22, September 24, October 1), or they are violently interrupted by supersaturation readings (> 12 to 32 mg/L) within 20 to 60 seconds. There is not a single sustained, physiologically credible nocturnal oxygen crash in the entire dataset.

---

## 10. ShinerAI 11-Feature Contract Feasibility (2-Hour Lookback)

The frozen ShinerAI model requires 11 specific features:
$$\mathbf{x} = [\text{current\_do}, \text{hour}, \text{minute}, \text{do}_{t-15}, \text{do}_{t-30}, \text{do}_{t-45}, \text{do}_{t-60}, \text{do}_{t-75}, \text{do}_{t-90}, \text{do}_{t-105}, \text{do}_{t-120}]$$

To extract 8 historical lags ($t-15\text{m} \dots t-120\text{m}$), an observation at timestamp $T$ requires at least **120 minutes of continuous preceding telemetry**.

* **Maximum continuous monitoring session in Part A:** **174.6 minutes** (June 25, 2021).
* **Second longest continuous session:** **76.8 minutes** (June 29, 2021).
* **All other 11 sessions:** $< 38\text{ minutes}$.
* **Rows with $\ge 120$ minutes of continuous pre-history:** Only **71 rows** (all occurring after 17:48:59 on June 25).

---

## 11. Future 2-Hour Target Feasibility

The frozen ShinerAI early warning task is:
$$\text{Given current DO } \ge 3.0\text{ mg/L and a 2-hour lookback, predict if DO } < 3.0\text{ mg/L within } [T+15\text{m}, T+120\text{m}].$$

This mandates that **after prediction time $T$, continuous telemetry must extend for an additional 120 minutes** into the future.

### The Mathematical Impossibility:
To satisfy both requirements, a single valid ShinerAI sample requires a continuous, uninterrupted monitoring window of:
$$\text{Window Length} = 120\text{ minutes (past)} + 120\text{ minutes (future)} = \mathbf{240\text{ minutes (4.0 hours)}}.$$

* **Longest continuous window available in `IoTPond10.csv`:** **174.6 minutes**.
* **Difference:** The entire dataset is **65.4 minutes too short** to support even a single 4-hour window.
* **Eligible ShinerAI samples with complete past and complete future coverage:** **EXACTLY ZERO (0 samples, 0.0%)**.

---

## 12. 15-Minute Resampling Feasibility

Can `IoTPond10.csv` be downsampled into non-overlapping 15-minute intervals $(T-15\text{m}, T]$?

* Over the 174.6-minute June 25 session, downsampling produces only **11 discrete 15-minute intervals**.
* ShinerAI requires:
  $$\text{8 past bins} + \text{1 current bin} + \text{8 future bins} = \mathbf{17\text{ consecutive 15-minute intervals (4.25 hours)}}.$$
* Because only 11 bins exist consecutively:
  $$\mathbf{11\text{ bins} < 17\text{ bins}}.$$
* **Conclusion:** It is mathematically impossible to produce even one valid 15-minute ShinerAI instance without fabricating synthetic data.

---

## 13. Dataset Independence Assessment

| Dimension | Internal Benchmark (FWI Arkansas) | External Validation 1 (Oman Tilapia) | Candidate 2 (`IoTPond10.csv`, Nigeria) |
|---|---|---|---|
| **Geography** | Lonoke County, Arkansas, USA (Subtropical) | North Al Sharqiyah, Oman (Arid Desert) | Nsukka, Enugu State, Nigeria (Tropical) |
| **Culture System** | 17 commercial earthen production ponds | 180L recirculating indoor aquarium tank | High-density aquaponics tank |
| **Target Species** | Golden Shiner (*Notemigonus crysoleucas*) | Nile Tilapia (*Oreochromis niloticus*) | African Catfish (*Clarias gariepinus*) |
| **Sensor Platform** | Optical/photometer multiparameter sonde | Analog probe + ESP32 IoT node | Uncalibrated analog sensors + OneWire |
| **Sampling Cadence** | Continuous 15-minute intervals (97.4% regular) | High-frequency 7-second continuous IoT | Fragmented 20-second bursts (< 3h sessions) |
| **Dataset Continuity** | 63.1 continuous days (41,277 ML examples) | 8.74 continuous days (742 valid examples) | 13 fragmented bursts over 102 days (0 examples) |

The dataset represents an independent geographical and biological system. However, independence is irrelevant when the physical time-series continuity is insufficient to form a single sample.

---

## 14. Ten-Criteria External Validation Evaluation

| # | Evaluation Criterion | Scientific Requirement | Audit Finding on `IoTPond10.csv` | Compliance |
|:---:|---|---|---|:---:|
| **1** | **Timestamp Quality** | Continuous, non-duplicated, parseable timestamps | Half the file has date-only duplicates; Part A has multi-week gaps | **FAIL** |
| **2** | **DO Reliability** | Plausible limnological oxygen dynamics | DO chatters 0 to 21 mg/L in 20s; 54.5% are 0.00 dropouts | **FAIL** |
| **3** | **Artifact Isolation** | Artifacts cleanly quarantined without destroying data | Quarantining -127°C, 0.0 DO, and spikes leaves sparse fragments | **FAIL** |
| **4** | **Historical Coverage** | Continuous $\ge 120$-minute pre-history for 8 lags | Only 71 rows have 2-hour pre-history in entire file | **FAIL** |
| **5** | **Future Coverage** | Continuous $\ge 120$-minute lookahead for target | 0 rows have 2-hour lookahead after 2-hour lookback | **FAIL** |
| **6** | **Hypoxia Ground Truth** | Genuine sustained biological hypoxia events | 0 genuine biological hypoxia events; only sensor dropouts | **FAIL** |
| **7** | **Zero Synthetic Interpolation** | No synthetic data fabrication permitted | Cannot form a single sample without synthetic imputation | **FAIL** |
| **8** | **Dataset Independence** | Independent species, facility, geography | Fully independent (Nigeria / Catfish / Aquaponics) | **PASS** |
| **9** | **Feature Contract Integrity** | Respects exact 11 Config C frozen features | Cannot extract 8 historical lags | **FAIL** |
| **10** | **Model & Threshold Freeze** | Zero retraining, zero threshold tuning | Evaluated under strict freeze | **PASS** |

---

## 15. External Validation Decision & Exact Next Step

### Formal Decision: **DEFINITIVE NO-GO**

### Methodological Justification:
1. **Mathematical Infeasibility:** ShinerAI mandates a 4-hour continuous window (2 hours past + 2 hours future). The longest continuous session in `IoTPond10.csv` is 2.91 hours (174.6 minutes). The total number of eligible evaluation samples is **strictly zero (0)**.
2. **Scientific Integrity Guardrail:** Creating evaluation samples from this dataset would require forward-filling multi-week gaps, fabricating synthetic timestamps, or interpolating 0.00 mg/L sensor disconnects. In accordance with strict scientific integrity standards, ShinerAI **refuses to fabricate synthetic data to manufacture artificial external validation**.
3. **Hardware Corruption:** The "low-DO" readings are electronic open-circuit dropouts and ADC saturation spikes (up to 32.41 mg/L), not water quality hypoxia.

### Exact Next Steps for Research:
1. **Retain and Document the Audit:** Archive this audit report in `docs/EXTERNAL_NIGERIA_DATASET_AUDIT.md` and `results/reports/nigeria_iotpond10_audit_summary.csv` as proof of methodological rigor for IEEE reviewers.
2. **Preserve External Validation Integrity:** Maintain the **Oman Nile Tilapia Validation** as ShinerAI's primary certified external validation benchmark (742 clean 15-minute intervals, 100.0% Specificity, zero false alarms).
3. **Pursue Continuous Open Hypoxia Datasets:** For future validation of external *Recall/Sensitivity*, continue searching public repositories for open-source datasets that provide uninterrupted multi-day monitoring with verified hypoxic crashes.
