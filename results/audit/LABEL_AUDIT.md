# ShinerAI: Ground-Truth Label Integrity Audit

**Audit Date:** October 3, 2026  
**Auditor:** Scientific Audit Agent  
**Dataset:** `data/processed/ml_ready_dataset.csv` ($N = 41,277$)  

---

## 1. Executive Summary

This audit examines the mathematical and physical soundness of the ground-truth prediction target in ShinerAI. We verify:
1. The mathematical definition of the early-warning label.
2. The operational filtering rule (restricting analysis to $\text{current_do} \ge 3.0\text{ mg/L}$).
3. The independence and correctness of the forward-looking label computation.
4. The exact class balance across training, purged, and test partitions.
5. The scientific framing of the $3.0\text{ mg/L}$ operational threshold.

We confirm **zero label logic discrepancies across all 41,277 observations**.

---

## 2. Mathematical Definition of the Target Label

The ShinerAI system formulates dissolved oxygen risk as a binary early-warning classification task:

$$\text{target}(t) = \begin{cases} 
1 & \text{if } \min_{m \in \{15, 30, 45, 60, 75, 90, 105, 120\}} \text{DO}(t + m) < 3.0\text{ mg/L} \\[6pt]
0 & \text{if } \min_{m \in \{15, 30, 45, 60, 75, 90, 105, 120\}} \text{DO}(t + m) \ge 3.0\text{ mg/L}
\end{cases}$$

Subject to the entry precondition:
$$\text{current_do}(t) \ge 3.0\text{ mg/L}$$

### Operational Rationale for the Precondition:
If dissolved oxygen is already below $3.0\text{ mg/L}$ at observation time $t$, hypoxia is an **active ongoing emergency**. Detecting current hypoxia is a trivial threshold alert, not an *early warning*. ShinerAI is designed to give farm managers actionable lead time (up to 2 hours) while water conditions are still normoxic or acceptable, allowing aeration systems to spin up before fish experience asphyxiation.

---

## 3. Independent Label Recomputation & Validation

The label generation was audited against raw 15-minute sensor trajectories across all 17 commercial ponds.
- **Total Validated Observations:** 41,277
- **Label Mismatches:** 0
- **False Negative Ground Truth Inconsistencies:** 0
- **False Positive Ground Truth Inconsistencies:** 0
- **Verification Result:** **100% Deterministic Reproducibility**

### Per-Pond Label Summary Table:
| Pond ID | Total Rows | SAFE ($y=0$) | AT_RISK ($y=1$) | Risk Prevalence (%) |
| :--- | :---: | :---: | :---: | :---: |
| `ara2_0677080b` | 1,484 | 1,243 | 241 | 16.24% |
| `ara2_0f143c64` | 3,145 | 3,036 | 109 | 3.47% |
| `ara2_148f1633` | 1,328 | 1,191 | 137 | 10.32% |
| `ara2_176528d3` | 1,872 | 1,467 | 405 | 21.63% |
| `ara2_29660d32` | 2,462 | 2,053 | 409 | 16.61% |
| `ara2_3b2f3973` | 3,648 | 3,396 | 252 | 6.91% |
| `ara2_3eab831e` | 2,857 | 2,375 | 482 | 16.87% |
| `ara2_3ed4df8e` | 2,347 | 2,069 | 278 | 11.84% |
| `ara2_45e1cde5` | 2,425 | 2,166 | 259 | 10.68% |
| `ara2_4f79fc8d` | 2,298 | 2,040 | 258 | 11.23% |
| `ara2_573a2826` | 2,324 | 1,756 | 568 | 24.44% |
| `ara2_6ca69422` | 2,318 | 2,020 | 298 | 12.86% |
| `ara2_858b914c` | 1,821 | 1,600 | 221 | 12.14% |
| `ara2_b3d128ac` | 2,184 | 1,891 | 293 | 13.42% |
| `ara2_c9cdacda` | 4,170 | 3,890 | 280 | 6.71% |
| `ara2_ca187575` | 2,045 | 1,764 | 281 | 13.74% |
| `ara2_d52ddb31` | 2,549 | 2,144 | 405 | 15.89% |

---

## 4. Class Balance Summary

Across the full dataset and experimental partitions:

| Partition | Total Examples | SAFE ($y=0$) | AT_RISK ($y=1$) | Class Ratio (Negative : Positive) | Imbalance Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Complete Dataset** | 41,277 | 36,101 | 5,176 | 6.98 : 1 (12.54% positive) | Reference distribution |
| **Training Partition** | 32,908 | 28,689 | 4,219 | 6.80 : 1 (12.82% positive) | `scale_pos_weight = 6.80`, balanced weights |
| **Purged Boundary** | 108 | 94 | 14 | 6.71 : 1 (12.96% positive) | Excised from training to eliminate leakage |
| **Holdout Test Set** | 8,261 | 7,318 | 943 | 7.76 : 1 (11.41% positive) | Evaluated with PR-AUC as primary metric |

---

## 5. Scientific Clarification on the 3.0 mg/L Threshold

> [!IMPORTANT]
> **Operational vs Biological Threshold:**
> The $3.0\text{ mg/L}$ threshold is an **operational engineering threshold** established for the ShinerAI deployment to provide a standardized safety buffer. It must **NOT** be described as an absolute universal biological threshold for all aquaculture species.
> - Coldwater teleosts (e.g., *Oncorhynchus mykiss*, Rainbow Trout) require $\text{DO} > 5.0–6.0\text{ mg/L}$ to prevent physiological stress.
> - Warmwater species (e.g., *Oreochromis niloticus*, Nile Tilapia, and *Ictalurus punctatus*, Channel Catfish) exhibit higher hypoxic tolerance, surviving short acute exposures below $2.0\text{ mg/L}$ but suffering feeding cessation and immune suppression below $3.0–3.5\text{ mg/L}$.
> - Penaeid shrimp (*Litopenaeus vannamei*) require $\ge 3.0\text{ mg/L}$ for commercial molt cycles.
> 
> Therefore, $3.0\text{ mg/L}$ represents an actionable management intervention line, rather than a universal biological constant.
