# Scientific Audit Report: Andhra Pradesh Aquaculture Dataset

**Dataset Title:** Aquaculture Water Quality  
**Associated Study:** *Quantum LSTM-driven IoT framework for real-time monitoring of dissolved oxygen in aquaculture systems: a case study in Andhra Pradesh, India*  
**Authors:** Nanduri Ashok Kumar and Jeevanantham Vellaichamy  
**Journal:** *Water Quality Research Journal* (IWA Publishing, March 2026)  
**DOI:** [10.2166/wqrj.2026.010](https://doi.org/10.2166/wqrj.2026.010) (Open Access)  
**Host Platform:** Kaggle (`https://www.kaggle.com/datasets/ashokkumarnandri/aquaculture-water-quality`)  
**Audit Date:** October 4, 2026  
**Auditor:** ShinerAI Scientific Review Team  

---

## 1. Audit Objective

Mentor review requested investigating the Andhra Pradesh aquaculture dataset as a candidate for a second independent external validation of ShinerAI:
> *"After the audit, investigate the previously identified Andhra Pradesh aquaculture dataset: 'Aquaculture Water Quality'... DO NOT assume it is suitable. FIRST AUDIT IT... If the dataset passes the audit: create an external validation pipeline. If it does not pass: DO NOT force it into the project."*

This audit rigorously evaluates the dataset against ShinerAI's operational task preconditions, temporal sampling requirements, data access terms, and scientific reproducibility standards.

---

## 2. Technical Audit Findings

| Audit Dimension | Evaluation Finding | Compliance Status | Scientific Impact on ShinerAI |
|---|---|:---:|---|
| **1. Primary Source & Citation** | Published in *Water Quality Research Journal* (March 2026), peer-reviewed academic article. | **PASS** | Valid academic provenance. |
| **2. Geographic & Facility Domain** | Monitored 3 aquaponic ponds in Andhra Pradesh, India under natural operating conditions. | **PASS** | Represents a distinct geographical location (South Asia) and farming system. |
| **3. Measured Water Parameters** | Recorded Dissolved Oxygen (mg/L), pH, Temperature (°C), Nitrate (ppm), Ammonia (mg/L), Turbidity (NTU), and Manganese (mg/L). | **PASS** | Contains Dissolved Oxygen, pH, and Temperature. |
| **4. Temporal Sampling Cadence** | The published paper explicitly documents a temporal resolution of **20 minutes per measurement** (3 readings per hour). | **FAIL** | **CRITICAL INCOMPATIBILITY:** ShinerAI's Config C feature architecture is strictly defined on **15-minute nominal intervals** ($t-15\text{m}, t-30\text{m}, \dots, t-120\text{m}$). A 20-minute time-series cannot construct these 8 discrete lag features without synthetic interpolation. |
| **5. Prohibition of Synthetic Interpolation** | ShinerAI's fundamental methodological rule states: *"No synthetic data or arbitrary interpolation across missing time intervals."* Mapping a 20-min grid onto a 15-min grid would require artificial cubic/spline resampling, corrupting empirical validation. | **FAIL** | Methodological rule violation if forced. |
| **6. Task Formulation Alignment** | The Andhra Pradesh study formulates a 3-class instantaneous categorization (Anoxia: DO < 2 mg/L; Hypoxia: 2–4 mg/L; Normoxia: > 4 mg/L) rather than a **2-hour forward-looking predictive early warning**. | **FAIL** | Different learning formulation requiring non-trivial relabeling. |
| **7. Platform Access & Downloadability** | Kaggle endpoint (`ashokkumarnandri/aquaculture-water-quality`) is protected by Cloudflare and Google reCAPTCHA Enterprise challenges. Automated unauthenticated HTTP requests return HTTP 404 or redirect to interactive browser challenge pages. | **FAIL** | Fails automated CI/CD programmatic download without interactive user login session. |
| **8. Official Data Availability Statement** | Article data availability statement: *"All relevant data are included in the paper or its Supplementary Information"* and available from corresponding author (`drjeevananthamv@veltech.edu.in`) upon request. Supplementary files are not bundled in an open Git repository (unlike the Oman Tilapia dataset). | **PARTIAL** | Available upon author request, not public API. |

---

## 3. In-Depth Technical Analysis of the 20-Minute Cadence Incompatibility

The central reason this dataset cannot be integrated into ShinerAI's frozen model evaluation pipeline is the fundamental mismatch in time-series geometry:

```
ShinerAI Fixed 15-Min Grid:
  [T-120m]  [T-105m]  [T-90m]  [T-75m]  [T-60m]  [T-45m]  [T-30m]  [T-15m]   [T]
     │         │        │        │        │        │        │        │        │
     ▼         ▼        ▼        ▼        ▼        ▼        ▼        ▼        ▼
  Lag 8     Lag 7    Lag 6    Lag 5    Lag 4    Lag 3    Lag 2    Lag 1    Current

Andhra Pradesh 20-Min Telemetry:
  [T-120m]       [T-100m]       [T-80m]        [T-60m]        [T-40m]        [T-20m]        [T]
     │              │              │              │              │              │            │
     ▼              ▼              ▼              ▼              ▼              ▼            ▼
   Matches      MISMATCH       MISMATCH        Matches       MISMATCH       MISMATCH      Matches
```

Only three points in a 2-hour window coincide ($T$, $T-60\text{m}$, $T-120\text{m}$). The remaining intermediate lags ($T-15\text{m}, T-30\text{m}, T-45\text{m}, T-75\text{m}, T-90\text{m}, T-105\text{m}$) would have to be fabricated through interpolation. 

In predictive modeling for life-critical early warning, evaluating a frozen model on interpolated synthetic features invalidates the scientific guarantee of out-of-distribution transferability.

---

## 4. Formal Audit Verdict

**VERDICT: REJECTED FOR EXTERNAL VALIDATION (NON-COMPLIANT).**

### Justification:
1. **Temporal Incompatibility:** 20-minute cadence violates ShinerAI's 15-minute feature contract and cannot be converted without synthetic interpolation.
2. **Access Limitation:** Interactive reCAPTCHA gating prevents unauthenticated reproducible pipelining.
3. **Scientific Integrity:** Forcing this dataset by interpolating data would violate ShinerAI's core methodological standards.

### Recommendation:
Document this audit transparently in the repository. Retain the **Oman Nile Tilapia dataset (*Sensors* 2026)** as the primary, verified, leakage-free external validation benchmark, while maintaining open documentation of its specificity-only verification and the need for future open 15-minute continuous hypoxia datasets.
