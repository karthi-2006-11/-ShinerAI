# ShinerAI

**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**GitHub Repository:** [https://github.com/karthi-2006-11/-ShinerAI.git](https://github.com/karthi-2006-11/-ShinerAI.git)  
**Formal Reference:** [`PROJECT_DOCUMENTATION.md`](file:///d:/FISH/PROJECT_DOCUMENTATION.md)

DataSet Link - (https://github.com/fish-welfare-initiative/Data-Campaign-Data?utm_source=chatgpt.com).
---

## What is ShinerAI?

**ShinerAI** is an artificial intelligence and machine learning early-warning solution designed to protect commercial freshwater aquaculture ponds from critical water quality degradation.

In aquaculture, **Dissolved Oxygen (DO)** is the single most volatile and life-critical environmental parameter. Under nocturnal microbial respiration and biological oxygen demand, oxygen levels can plummet rapidly overnight. When dissolved oxygen falls below critical biological thresholds—provisionally set at **3.0 mg/L** (pending species-specific biological validation)—fish experience severe respiratory distress (hypoxia) and rapid mortality.

### The Core Problem Solved by ShinerAI:
> *"Given an aquaculture pond whose current dissolved oxygen is healthy ($\text{DO} \ge 3.0\text{ mg/L}$), will DO fall below $3.0\text{ mg/L}$ at any point during the subsequent 2 hours?"*

By forecasting impending hypoxia up to **2 hours in advance**, ShinerAI gives fish farmers ample lead time to power up mechanical aerators, start freshwater exchange pumps, or adjust feed management before fish sustain biological damage.

---

## Project Status

- **Phase 1: Environment Setup & Dataset Audit** — **COMPLETED**
  - Audited 17 pond continuous time-series CSV files (72,750 continuous 15-minute readings).
  - Confirmed 15-minute nominal sampling cadence (97.37% adherence) and identified multi-day operational gaps.
  - Cataloged equipment artifact zeros and QC flags.
  - Proved empirical feasibility: 15,451 low-DO readings occur across all 17 ponds (21.24% of raw dataset).
  - Detailed report: [`DATASET_AUDIT.md`](file:///d:/FISH/DATASET_AUDIT.md).

- **Phase 2: Data Cleaning, Label Generation & Final Reconciliation** — **COMPLETED**
  - Established formal QC cleaning policy ([`QC_CLEANING_POLICY.md`](file:///d:/FISH/QC_CLEANING_POLICY.md)).
  - Cleaned and segmented time series, isolating equipment artifacts and conflicting duplicate records.
  - Constructed leak-free supervised learning dataset ([`data/processed/ml_ready_dataset.csv`](file:///d:/FISH/data/processed/ml_ready_dataset.csv)) with 41,277 examples across 37 columns.
  - Completed strict numerical reconciliation: accounted for all 72,750 raw rows with exactly zero unexplained rows ([`results/reports/phase2_row_accounting.csv`](file:///d:/FISH/results/reports/phase2_row_accounting.csv)).
  - Resolved duplicate discrepancy: 258 excess rows via `keep='first'` vs. 512 total colliding records via `keep=False`.
  - Reconciled class distribution: 36,101 SAFE (87.46%), 5,176 AT_RISK (12.54%), 6.97 : 1 ratio ("moderately imbalanced class distribution").
  - Enforced zero data leakage: 24 past historical features ($t-120\text{m} \dots t$) and quarantined future 2-hour target ($t+15\text{m} \dots t+120\text{m}$).
  - All 24 automated unit and data leakage tests passing.
  - Detailed report: [`PHASE_2_REPORT.md`](file:///d:/FISH/PHASE_2_REPORT.md).

- **Formal Project Documentation** — **COMPLETED**
  - Comprehensive 40-section technical specification: [`PROJECT_DOCUMENTATION.md`](file:///d:/FISH/PROJECT_DOCUMENTATION.md).

- **Phase 3: Feature Engineering & Baseline Modeling** — **PENDING USER APPROVAL**

---

## Dataset Health & Reconciliation Summary

| Metric | Value | Rationale / Detail |
|---|---|---|
| **Raw Observations** | **72,750** | Total readings across 17 ponds in `data/raw/csv/` |
| **Monitored Ponds** | **17** | All 17 ponds tracked and represented |
| **Date Range** | **2025-11-28 to 2026-01-30** | ~9 weeks (63.09 days) continuous telemetry |
| **Sampling Cadence** | **~15 minutes** | 97.37% adherence within 13.5–16.5 min window |
| **Excluded Sensor Artifacts (Zeros)** | **629** | Hardware reset/probe disconnect (`DO=0`, `pH=0`, or `Temp=0`) |
| **Excluded Conflicting Duplicates** | **512** | Timestamp collisions with conflicting measurements |
| **Usable Baseline Observations** | **71,609** | Valid non-zero sensor readings eligible for windowing |
| **Excluded Current DO < 3.0 mg/L** | **15,234** | Pond already in hypoxia at time $T$; non-candidate for early warning |
| **Excluded Insufficient Past History** | **8,700** | Fewer than 8 prior readings in contiguous segment ($< 2$h) |
| **Excluded Insufficient Future Coverage** | **6,398** | Sensor stopped before $T+120\text{m}$ without drop; prevents false SAFE |
| **Final ML-Ready Examples** | **41,277** | Supervised learning examples in `ml_ready_dataset.csv` |
| **SAFE Examples (`target = 0`)** | **36,101 (87.46%)** | DO remains $\ge 3.0$ mg/L across full 2-hour future window |
| **AT_RISK Examples (`target = 1`)** | **5,176 (12.54%)** | DO drops $< 3.0$ mg/L within 2-hour future window |
| **Class Imbalance Ratio** | **6.97 : 1** | Moderately imbalanced class distribution |
| **ML-Ready Columns** | **37** | 34 predictor features + 3 quality/target columns |
| **Unexplained Rows** | **0 (0.00%)** | Mathematical identity strictly verified |
| **Automated Tests Passing** | **24 / 24 (100%)** | `pytest -v` across data, cleaning, and leakage suites |
| **Label Logic Validation** | **40 / 40 (100%)** | 20 SAFE + 20 AT_RISK samples verified against raw telemetry |
| **Leakage Audit Status** | **PASS (0 violations)** | Zero future-reading leakage in predictor features |

---

## Data Flow Architecture

The ShinerAI pipeline enforces strict temporal separation: **past data forms predictor features**, while **future data is used exclusively to assign target labels**.

```text
FWI Continuous Pond Telemetry (17 Ponds, 72,750 Rows)
                    │
                    ▼
          Integrity & Quality Audit
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
Quarantine Artifacts     Quarantine Duplicates
      (629 rows)              (512 rows)
         │                     │
         └──────────┬──────────┘
                    ▼
          71,609 Usable Observations
                    │
                    ▼
       Continuous Time Segmentation (Gaps > 20 min)
                    │
                    ▼
     Candidate Window Generation at Timestamp T
                    │
         ┌──────────┴─────────────────────────┐
         │                                    │
         ▼                                    ▼
[PAST ONLY: T-120m to T]             [CURRENT STATE CHECK AT T]
Contiguous >= 8 prior steps?         Is current DO >= 3.0 mg/L?
   NO -> Exclude Past (8,700)           NO -> Exclude Already Hypoxic (15,234)
   YES -> 34 Predictor Features         YES -> Eligible Candidate
         │                                    │
         └──────────────────┬─────────────────┘
                            ▼
              [FUTURE ONLY: T to T+120m]
              Does DO fall below 3.0 mg/L?
                    │
         ┌──────────┴─────────────────────────┐
         ▼                                    ▼
       YES                                   NO
Label: AT_RISK (1)                    Did sensor monitor full 2h?
  (5,176 examples)                       NO -> Exclude Future (6,398)
         │                               YES -> Label: SAFE (0)
         │                                        (36,101 examples)
         │                                    │
         └──────────────────┬─────────────────┘
                            ▼
       Final ML-Ready Dataset: 41,277 Examples (37 Columns)
                    │
                    ▼
         Phase 3 Baseline Modeling (Pending Approval)
```

> **Detailed Architecture Diagram:** Saved at [`results/figures/data_flow_diagram.png`](file:///d:/FISH/results/figures/data_flow_diagram.png).

---

## Project Structure

```text
fish-farm-early-warning/
│
├── data/
│   ├── raw/
│   │   └── csv/                     # Original 17 pond CSVs + 5 metadata CSVs (READ-ONLY)
│   └── processed/
│       ├── cleaned_pond_data.csv    # Cleaned time series with QC status & segment IDs
│       └── ml_ready_dataset.csv     # 41,277 supervised learning examples (37 columns)
│
├── notebooks/
│   └── 01_dataset_audit.ipynb       # Interactive walkthrough of Phase 1 audit
│
├── src/
│   ├── __init__.py                  # Package marker
│   ├── data_loader.py               # Reusable data discovery, parsing, and combination
│   ├── data_quality.py              # Quality audits, interval checks, QC & feasibility logic
│   ├── audit.py                     # Automated Phase 1 audit pipeline
│   └── cleaning.py                  # Phase 2 cleaning, segmentation, and ML dataset pipeline
│
├── results/
│   ├── figures/                     # Standalone Matplotlib figures
│   │   ├── data_flow_diagram.png    # End-to-end pipeline architecture
│   │   ├── do_time_series_*.png     # 17 individual pond DO plots
│   │   ├── example_at_risk_event.png# Example AT_RISK window (past + future drop)
│   │   ├── example_safe_event.png   # Example SAFE window (past + stable future)
│   │   ├── safe_vs_at_risk_distribution.png
│   │   └── at_risk_percentage_by_pond.png
│   │
│   └── reports/                     # Tabular CSV and JSON reports
│       ├── phase2_row_accounting.csv# Strict numerical row reconciliation (0 unexplained)
│       ├── label_logic_validation.csv# Audit of random SAFE and AT_RISK prediction windows
│       ├── leakage_audit_report.json# Formal verification of zero future leakage
│       ├── pond_summary.csv         # Descriptive stats per pond
│       ├── qc_summary.csv           # QC flag counts per pond
│       ├── feasibility_summary.csv  # DO < 3.0 mg/L feasibility
│       ├── label_summary.csv        # Dataset-wide label counts & exclusion metrics
│       ├── label_summary.json       # Machine-readable label summary
│       ├── label_summary_by_pond.csv# Class counts per pond
│       └── early_warning_event_analysis.csv # AT_RISK event lead times & hours
│
├── tests/
│   ├── test_data_pipeline.py        # Phase 1 loader & audit tests (10 tests)
│   ├── test_cleaning_pipeline.py    # Phase 2 cleaning & dataset schema tests (9 tests)
│   └── test_no_data_leakage.py      # Phase 2 future leakage & target window tests (5 tests)
│
├── requirements.txt                 # Lightweight Python dependencies
├── pytest.ini                       # Pytest path configuration
├── README.md                        # Project landing page & quickstart
├── PROJECT_DOCUMENTATION.md         # Formal 40-section project reference
├── DATASET_AUDIT.md                 # In-depth Phase 1 audit report
├── QC_CLEANING_POLICY.md            # Phase 2 data cleaning governance policy
├── PHASE_2_REPORT.md                # Comprehensive Phase 2 execution report
└── PHASE_2_BEGINNER_GUIDE.md        # Beginner guide to features, labels & leakage
```

---

## Setup & Execution Guide (Windows)

### 1. Virtual Environment Activation
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run Phase 1 Audit
```powershell
python src/audit.py
```

### 3. Run Phase 2 Cleaning & Dataset Generation
```powershell
python src/cleaning.py
```

### 4. Run the Full Automated Test Suite (24 Tests)
```powershell
pytest -v
```

### 5. Launch JupyterLab
```powershell
jupyter lab
```
Navigate to `notebooks/01_dataset_audit.ipynb` to view the interactive audit.

---

## Methodological Guardrails

1. **Read-Only Raw Telemetry:** Raw files in `data/raw/csv/` are strictly read-only and never modified.
2. **Zero Forward Leakage:** Feature matrices never include future observations ($t > T$).
3. **No Synthetic Data:** Missing temporal gaps are partitioned into new segments rather than interpolated with synthetic numbers.
4. **Physiological Caveat:** The 3.0 mg/L threshold is provisional pending species-specific biological validation; ShinerAI forecasts water oxygen depletion, not biological diseases.
