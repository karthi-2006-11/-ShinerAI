# ShinerAI

**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**GitHub Repository:** [https://github.com/karthi-2006-11/-ShinerAI.git](https://github.com/karthi-2006-11/-ShinerAI.git)  
**Formal Reference:** [`PROJECT_DOCUMENTATION.md`](file:///d:/FISH/PROJECT_DOCUMENTATION.md)

DataSet Link - https://github.com/fish-welfare-initiative/Data-Campaign-Data.git.
---

## What is ShinerAI?

**ShinerAI** is an artificial intelligence and machine learning early-warning solution designed to protect commercial freshwater aquaculture ponds from critical water quality degradation.

In aquaculture, **Dissolved Oxygen (DO)** is the single most volatile and life-critical environmental parameter. Under nocturnal microbial respiration and biological oxygen demand, oxygen levels can plummet rapidly overnight. When dissolved oxygen falls below critical biological thresholds—provisionally set at **3.0 mg/L** (pending species-specific biological validation)—fish experience severe respiratory distress (hypoxia) and rapid mortality.

### The Core Problem Solved by ShinerAI:
> *"Given an aquaculture pond whose current dissolved oxygen is healthy ($\text{DO} \ge 3.0\text{ mg/L}$), will DO fall below $3.0\text{ mg/L}$ at any point during the subsequent 2 hours?"*

By forecasting impending hypoxia up to **2 hours in advance**, ShinerAI gives fish farmers ample lead time to power up mechanical aerators, start freshwater exchange pumps, or adjust feed management before fish sustain biological damage.

---

## Open & Run ShinerAI

### GitHub Repository

[Open ShinerAI on GitHub](https://github.com/karthi-2006-11/-ShinerAI)

### Clone the Repository

```bash
git clone https://github.com/karthi-2006-11/-ShinerAI.git
cd -ShinerAI
```

### Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Flask Application

Launch the Flask REST backend and interactive web dashboard:

```bash
python -m flask --app backend.app run
```

*Or directly:*
```bash
python -m backend.app
```

### Open the Dashboard in the Browser

Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```
*(or [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard))*

The dashboard will open with live API connectivity, preloaded holdout demonstration scenarios (SAFE and AT_RISK), 2-hour DO trajectory charts, and local SHAP feature explanations.

### Launch the Master Research & Reproducibility Notebook

To inspect, run, or reproduce the complete end-to-end machine learning research pipeline in a single self-contained notebook:

```bash
jupyter notebook notebooks/ShinerAI_Complete_ML_Pipeline.ipynb
```
*(or open [`notebooks/ShinerAI_Complete_ML_Pipeline.ipynb`](file:///d:/FISH/notebooks/ShinerAI_Complete_ML_Pipeline.ipynb) directly in VS Code / Jupyter Lab)*

---

## Project Status

- **Research Reproducibility & Master Pipeline Notebook** — **COMPLETED**
  - **Single Source of Truth Notebook:** [`notebooks/ShinerAI_Complete_ML_Pipeline.ipynb`](file:///d:/FISH/notebooks/ShinerAI_Complete_ML_Pipeline.ipynb) covers 33 sequentially structured sections unifying data loading, audit, cleaning, label logic verification, leak-free temporal splitting with 2-hour purge gap, baseline modeling, candidate training (Logistic Regression, Random Forest, XGBoost across Configs A, B, C), 5-fold GroupKFold unseen-pond generalization, temporal holdout evaluation, forensic error analysis, high-precision wall-time profiling, global/local SHAP explainability, and domain-specific contribution experiments.
  - **Master Model Evaluation Table:** [`results/reports/MASTER_MODEL_EVALUATION.csv`](file:///d:/FISH/results/reports/MASTER_MODEL_EVALUATION.csv) comparing all 11 model configurations and 2 baselines across 12 standardized classification metrics (PR-AUC, ROC-AUC, F1, Recall, Precision, Specificity, Accuracy, TP, FP, TN, FN).
  - **Pipeline Wall-Time Benchmark:** [`results/timing/pipeline_wall_time.csv`](file:///d:/FISH/results/timing/pipeline_wall_time.csv) profiling total end-to-end execution (~20.5 seconds) and confirming sub-millisecond inference latency (< 0.05 ms per sample).
  - **Publication Figures:** 8 publication-grade research figures saved in [`results/figures/`](file:///d:/FISH/results/figures/).

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

- **Phase 3: Machine Learning Training & Evaluation** — **COMPLETED**
  - Evaluated baselines (Majority PR-AUC: 0.1142; Current-DO Boolean Threshold F1: 0.6257; Current-DO Ranking PR-AUC: 0.6149).
  - Evaluated 3 feature configurations (Config A: Current, Config B: Current + History, Config C: DO History).
  - Executed leak-free temporal holdout (80% train / 20% test per pond with 2h purge gap).
  - Evaluated 5-fold GroupKFold unseen-pond generalization across all 17 ponds.
  - Evaluated operational trade-offs: XGBoost Config C achieved highest PR-AUC (0.7574); Random Forest Config C achieved highest F1 (0.6602) and specificity (93.11%), producing fewer false alarms (504 FPs) at default threshold; Logistic Regression Config B provided highest recall (89.93%).
  - Serialized model artifacts under `models/` with metadata specification.
  - Detailed report: [`PHASE_3_REPORT.md`](file:///d:/FISH/PHASE_3_REPORT.md).
  - Beginner guide: [`PHASE_3_BEGINNER_GUIDE.md`](file:///d:/FISH/PHASE_3_BEGINNER_GUIDE.md).

- **Phase 4: Model Explainability & Flask Backend REST API** — **COMPLETED**
  - Implemented SHAP TreeExplainer pipeline (`src/explainability_pipeline.py`) computing global feature importance (1,000 test observations) and local waterfall/bar explanations for representative SAFE and AT_RISK cases.
  - Confirmed `current_do` as rank #1 driver, diurnal markers (`minute_of_day`, `hour_of_day`) encoding time-of-day patterns as #2 and #3, and recent trajectory lags as #4 and #5.
  - Transparent artifact provenance: Config C model artifacts were reproduced using the frozen Phase 3 training procedure solely to create dedicated explainability/API artifacts; no model architecture, dataset, split, feature set, or training procedure was changed.
  - Built production Flask REST API (`backend/`) with strict input validation, boundary condition enforcement ($\text{DO} \ge 3.0\text{ mg/L}$), canonical public input naming (`do_t_minus_15` to `do_t_minus_120`), and four endpoints (`/health`, `/model-info`, `/predict`, `/explain`).
  - Supported configurable model artifact loading (`MODEL_ARTIFACT_PATH`) between XGBoost Config C (highest PR-AUC among tested models: 0.7574, highest recall among Config C tree models: 79.75%) and Random Forest Config C (highest Specificity: 93.11%, producing 194 fewer false positives than XGBoost Config C at the default threshold; this could reduce unnecessary interventions in a deployment where alerts trigger aeration).
  - Detailed report: [`PHASE_4_REPORT.md`](file:///d:/FISH/PHASE_4_REPORT.md).
  - Beginner guide: [`PHASE_4_BEGINNER_GUIDE.md`](file:///d:/FISH/PHASE_4_BEGINNER_GUIDE.md).
  - REST API specification: [`docs/API.md`](file:///d:/FISH/docs/API.md).

- **Phase 5: Interactive Dashboard & UI + End-to-End Integration** — **COMPLETED**
  - Developed a lightweight, accessible, and responsive web dashboard in `frontend/` using pure semantic HTML5, vanilla CSS3, and native JavaScript (ES6+).
  - Designed a photorealistic, interactive procedural aquarium theme background (`frontend/aquarium.js`) with 42 schooling goldfish, sinusoidal spine undulation, dynamic group escape physics on cursor approach, swaying aquatic plants, and rising aerator bubbles clearly visible through translucent glassmorphic cards (`backdrop-filter: blur(22px) saturate(115%)`, subtle readability veil `rgba(235, 248, 255, 0.06)`, translucent gradient `rgba(255, 255, 255, 0.20)` to `0.12`), a floating frosted glass navbar with custom vector "ShinerAI" wordmark, and an aquatic pearl-white typography system.
  - Flask backend seamlessly serves the static dashboard directly at `GET /` and `GET /dashboard` while maintaining standard REST API discovery endpoints.
  - Implemented dynamic 2-hour DO trajectory visualization in pure SVG with a prominent $3.0\text{ mg/L}$ provisional hypoxia threshold reference line.
  - Built-in demonstration scenarios (SAFE Case Study — Rising DO During Daytime vs. AT_RISK Case Study — Declining DO During Nighttime) loaded directly from the evaluated temporal holdout set.
  - Interactive prediction (`POST /predict`) and local SHAP explainability (`POST /explain`) with direction indicators (`↑ Toward AT_RISK`, `↓ Toward SAFE`).
  - Automated test suite expanded to 78 tests with 100% pass rate.
  - Detailed report: [`PHASE_5_REPORT.md`](file:///d:/FISH/PHASE_5_REPORT.md).
  - Beginner guide: [`PHASE_5_BEGINNER_GUIDE.md`](file:///d:/FISH/PHASE_5_BEGINNER_GUIDE.md).

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
| **ML-Ready Columns** | **37** | 34 input/metadata columns. The actual model predictors will be selected during Phase 3. (+ 3 quality/target columns) |
| **Unexplained Rows** | **0 (0.00%)** | Mathematical identity strictly verified |
| **Automated Tests Passing** | **24 / 24 (100%)** | `pytest -v` across data, cleaning, and leakage suites |
| **Label Logic Validation** | **40 / 40 (100%)** | 20 SAFE + 20 AT_RISK samples verified against raw telemetry |
| **Leakage Audit Status** | **PASS (0 violations)** | Zero future-reading leakage in past input features |

---

## Data Flow Architecture

The ShinerAI pipeline enforces strict temporal separation: **past data forms input features**, while **future data is used exclusively to assign target labels**.

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
   YES -> 34 Input/Metadata Columns     YES -> Eligible Candidate
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
├── backend/                         # Phase 4 Modular Flask REST API
│   ├── __init__.py                  # Package marker & app export
│   ├── app.py                       # Application factory & REST routes (/health, /predict, /explain)
│   ├── config.py                    # Environment configuration & feature schema
│   ├── validation.py                # Payload validation & boundary condition enforcement
│   ├── model_service.py             # Read-only model artifact loading & inference
│   └── explainability.py            # Local SHAP TreeExplainer service
│
├── data/
│   ├── raw/
│   │   └── csv/                     # Original 17 pond CSVs + 5 metadata CSVs (READ-ONLY)
│   └── processed/
│       ├── cleaned_pond_data.csv    # Cleaned time series with QC status & segment IDs
│       └── ml_ready_dataset.csv     # 41,277 supervised learning examples (37 columns)
│
├── docs/
│   └── API.md                       # Formal REST API specification
│
├── models/
│   ├── logistic_regression.joblib   # Linear high-recall candidate
│   ├── random_forest.joblib         # Random Forest candidate (Config B)
│   ├── xgboost.joblib               # XGBoost candidate (Config B)
│   ├── random_forest_config_c.joblib# Config C Random Forest (highest specificity: 93.11%)
│   ├── xgboost_config_c.joblib      # Config C XGBoost (highest PR-AUC: 0.7574, default serving)
│   └── model_metadata.json          # Complete hyperparameters, metadata & metrics
│
├── notebooks/
│   └── 01_dataset_audit.ipynb       # Interactive walkthrough of Phase 1 audit
│
├── src/
│   ├── __init__.py                  # Package marker
│   ├── data_loader.py               # Reusable data discovery, parsing, and combination
│   ├── data_quality.py              # Quality audits, interval checks, QC & feasibility logic
│   ├── audit.py                     # Automated Phase 1 audit pipeline
│   ├── cleaning.py                  # Phase 2 cleaning, segmentation, and ML dataset pipeline
│   ├── model_utils.py               # Phase 3 feature configs, temporal splitting, metrics
│   ├── train.py                     # Phase 3 model training & serialization pipeline
│   ├── evaluate.py                  # Phase 3 per-pond & GroupKFold evaluation pipeline
│   └── explainability_pipeline.py   # Phase 4 global & local SHAP explainability pipeline
│
├── results/
│   ├── figures/                     # Standalone Matplotlib figures
│   │   ├── explainability/          # Phase 4 global and local SHAP plots
│   │   │   ├── global_feature_importance_xgb_config_c.png
│   │   │   ├── global_feature_importance_rf_config_c.png
│   │   │   ├── local_explanation_safe_xgb_config_c.png
│   │   │   ├── local_explanation_at_risk_xgb_config_c.png
│   │   │   ├── local_explanation_safe_rf_config_c.png
│   │   │   └── local_explanation_at_risk_rf_config_c.png
│   │   ├── data_flow_diagram.png    # End-to-end Phase 1-2 pipeline architecture
│   │   ├── ml_pipeline_diagram.png  # Phase 3 ML workflow architecture
│   │   ├── roc_curve_comparison.png # ROC curves across models
│   │   ├── pr_curve_comparison.png  # Precision-Recall curves across models
│   │   ├── confusion_*.png          # Confusion matrices for candidate models
│   │   ├── do_time_series_*.png     # 17 individual pond DO plots
│   │   ├── example_at_risk_event.png# Example AT_RISK window (past + future drop)
│   │   ├── example_safe_event.png   # Example SAFE window (past + stable future)
│   │   ├── safe_vs_at_risk_distribution.png
│   │   └── at_risk_percentage_by_pond.png
│   │
│   └── reports/                     # Tabular CSV and JSON reports
│       ├── explainability/          # Phase 4 SHAP rankings & case study JSON summaries
│       │   ├── global_importance_xgb_config_c.csv
│       │   ├── global_importance_rf_config_c.csv
│       │   ├── local_explanations_summary.json
│       │   └── explainability_report.md
│       ├── final_model_selection.csv# Neutral model selection & trade-off matrix
│       ├── model_comparison.csv     # Phase 3 model benchmark results across configurations
│       ├── model_comparison.json    # Machine-readable model benchmarks
│       ├── per_pond_model_performance.csv # Pond-by-pond performance breakdown
│       ├── group_kfold_performance.csv    # 5-Fold GroupKFold unseen-pond results
│       ├── temporal_split_accounting.csv  # Train/purge/test row accounting per pond
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
│   ├── test_no_data_leakage.py      # Phase 2 future leakage & target window tests (5 tests)
│   ├── test_phase3_splits.py        # Phase 3 temporal holdout & purge tests (3 tests)
│   ├── test_phase3_models.py        # Phase 3 model, artifact & metric tests (5 tests)
│   ├── test_phase4_api.py           # Phase 4 REST API & validation tests (23 tests)
│   └── test_phase4_explainability.py# Phase 4 SHAP artifact & service tests (4 tests)
│
├── requirements.txt                 # Lightweight Python dependencies
├── pytest.ini                       # Pytest path configuration
├── README.md                        # Project landing page & quickstart
├── PROJECT_DOCUMENTATION.md         # Formal 42-section project reference
├── DATASET_AUDIT.md                 # In-depth Phase 1 audit report
├── QC_CLEANING_POLICY.md            # Phase 2 data cleaning governance policy
├── PHASE_2_REPORT.md                # Comprehensive Phase 2 execution report
├── PHASE_2_BEGINNER_GUIDE.md        # Beginner guide to features, labels & leakage
├── PHASE_3_REPORT.md                # Comprehensive Phase 3 ML execution report
├── PHASE_3_BEGINNER_GUIDE.md        # Beginner guide to machine learning & metrics
├── PHASE_4_REPORT.md                # Comprehensive Phase 4 Explainability & API report
└── PHASE_4_BEGINNER_GUIDE.md        # Beginner guide to SHAP explainability & API serving
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

### 4. Run Phase 3 Model Training & Evaluation
```powershell
python src/train.py
python src/evaluate.py
```

### 5. Run Phase 4 Explainability Pipeline
```powershell
python src/explainability_pipeline.py
```

### 6. Launch the Flask REST Backend API & Web Dashboard
```powershell
.venv\Scripts\python.exe -m flask --app backend.app run
# Or directly:
.venv\Scripts\python.exe -m backend.app
```
*Access the interactive dashboard in your browser at `http://127.0.0.1:5000/` or `http://127.0.0.1:5000/dashboard`.*  
*(By default runs on `http://127.0.0.1:5000` serving `models/xgboost_config_c.joblib`)*

### 7. Run the Full Automated Test Suite (78 Tests)
```powershell
.venv\Scripts\python.exe -m pytest -v
```

### 8. Launch JupyterLab
```powershell
jupyter lab
```
Navigate to `notebooks/01_dataset_audit.ipynb` to view the interactive audit.

---

## Methodological Guardrails

1. **Read-Only Raw Telemetry:** Raw files in `data/raw/csv/` are strictly read-only and never modified.
2. **Zero Forward Leakage:** Feature matrices never include future observations ($t > T$).
3. **No Synthetic Data:** Missing temporal gaps are partitioned into new segments rather than interpolated with synthetic numbers.
4. **Scope & Physiological Caveat:** The 3.0 mg/L threshold is provisional pending species-specific biological validation; ShinerAI forecasts impending water oxygen depletion events ($\text{DO} < 3.0\text{ mg/L}$ within 2 hours), not fish disease or fish mortality.
5. **Statistical Association vs. Causality:** SHAP attributions represent statistical predictive associations within this dataset; they do not prove biological causality.

