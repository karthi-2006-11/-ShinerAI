# ShinerAI: Temporal Train/Test Split & Leakage Audit

**Audit Date:** October 3, 2026  
**Auditor:** Scientific Audit Agent  
**Dataset:** `data/processed/ml_ready_dataset.csv` ($N = 41,277$)  

---

## 1. Executive Summary

This audit rigorously verifies the temporal partition methodology used to evaluate all models in ShinerAI. Time-series forecasting and early-warning tasks in aquaculture are uniquely susceptible to lookahead bias and target leakage if cross-validation or random splits are applied naively. 

We confirm that ShinerAI applies a **strict per-pond chronological holdout split (80% train / 20% test)** augmented by a **2.0-hour boundary purge gap**. The split is 100% leak-free.

---

## 2. Mathematical Definition of the Temporal Purge Gap

For each unique pond $p \in \{1, \dots, 17\}$:
1. All valid telemetry observations are ordered strictly chronologically:
   $$\mathcal{D}_p = \{(x_i, y_i, t_i) \mid t_1 < t_2 < \dots < t_{N_p}\} $$
2. The 80th percentile timestamp is designated as the cutoff boundary:
   $$t_{\text{cutoff}}^{(p)} = \text{quantile}_{0.80}(\{t_i\})$$
3. The test set $\mathcal{D}_{\text{test}, p}$ is defined as all observations at or after cutoff:
   $$\mathcal{D}_{\text{test}, p} = \{(x_i, y_i, t_i) \mid t_i \ge t_{\text{cutoff}}^{(p)}\}$$
4. Because the target label $y_i$ is defined across the future 2-hour window $[t_i, t_i + 2\text{ hr}]$:
   $$y_i = \mathbb{I}\left(\min_{m \in \{15, \dots, 120\}} \text{DO}_{t_i + m} < 3.0\text{ mg/L}\right)$$
   Any training observation occurring in the window $[t_{\text{cutoff}}^{(p)} - 2\text{ hr}, t_{\text{cutoff}}^{(p)})$ would evaluate sensor readings occurring *after* $t_{\text{cutoff}}^{(p)}$ (inside the test period).
5. Therefore, a mandatory 2.0-hour purge interval is excised completely from training:
   $$\mathcal{D}_{\text{purge}, p} = \{(x_i, y_i, t_i) \mid t_{\text{cutoff}}^{(p)} - 2\text{ hr} \le t_i < t_{\text{cutoff}}^{(p)}\}$$
6. The valid, leak-free training set is strictly bounded:
   $$\mathcal{D}_{\text{train}, p} = \{(x_i, y_i, t_i) \mid t_i < t_{\text{cutoff}}^{(p)} - 2\text{ hr}\} $$

---

## 3. Dataset-Level Accounting Verification

| Partition | Row Count | Target = 0 (SAFE) | Target = 1 (AT_RISK) | Prevalence (% AT_RISK) |
| :--- | :---: | :---: | :---: | :---: |
| **Complete Dataset** | 41,277 | 36,101 | 5,176 | 12.54% |
| **Training Partition** | 32,908 | 28,689 | 4,219 | 12.82% |
| **Purged Partition (2h)**| 108 | 94 | 14 | 12.96% |
| **Test Partition** | 8,261 | 7,318 | 943 | 11.42% |

### Conservation Identity Check:
$$\text{Train} (32,908) + \text{Purged} (108) + \text{Test} (8,261) = 41,277 \equiv 41,277 \quad (\textbf{PASSED: 100% RECONCILED})$$

---

## 4. Per-Pond Split Accounting Breakdown

| Pond ID | Total Rows | Train Rows | Purged (2h) | Test Rows | Train Positive (%) | Test Positive (%) | Start Date | Cutoff Date/Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ara2_0677080b` | 1,484 | 1,179 | 8 | 297 | 197 (16.7%) | 44 (14.8%) | 2025-12-14 | 2026-01-26 00:21 |
| `ara2_0f143c64` | 3,145 | 2,513 | 3 | 629 | 107 (4.3%) | 2 (0.3%) | 2025-12-14 | 2026-01-21 12:48 |
| `ara2_148f1633` | 1,328 | 1,055 | 7 | 266 | 85 (8.1%) | 45 (16.9%) | 2026-01-04 | 2026-01-25 12:48 |
| `ara2_176528d3` | 1,872 | 1,489 | 8 | 375 | 331 (22.2%) | 74 (19.7%) | 2025-12-14 | 2026-01-20 19:27 |
| `ara2_29660d32` | 2,462 | 1,961 | 8 | 493 | 326 (16.6%) | 83 (16.8%) | 2025-11-28 | 2026-01-20 13:42 |
| `ara2_3b2f3973` | 3,648 | 2,910 | 8 | 730 | 252 (8.7%) | 0 (0.0%) | 2025-11-30 | 2026-01-22 22:54 |
| `ara2_3eab831e` | 2,857 | 2,277 | 8 | 572 | 414 (18.2%) | 68 (11.9%) | 2025-11-29 | 2026-01-21 18:12 |
| `ara2_3ed4df8e` | 2,347 | 1,877 | 0 | 470 | 231 (12.3%) | 47 (10.0%) | 2025-12-13 | 2026-01-23 03:21 |
| `ara2_45e1cde5` | 2,425 | 1,932 | 8 | 485 | 235 (12.2%) | 24 (4.9%) | 2025-12-14 | 2026-01-21 01:03 |
| `ara2_4f79fc8d` | 2,298 | 1,830 | 8 | 460 | 210 (11.5%) | 48 (10.4%) | 2025-12-14 | 2026-01-21 13:39 |
| `ara2_573a2826` | 2,324 | 1,852 | 7 | 465 | 468 (25.3%) | 97 (20.9%) | 2025-12-14 | 2026-01-22 23:51 |
| `ara2_6ca69422` | 2,318 | 1,846 | 8 | 464 | 281 (15.2%) | 17 (3.7%) | 2025-12-17 | 2026-01-25 09:09 |
| `ara2_858b914c` | 1,821 | 1,448 | 8 | 365 | 158 (10.9%) | 59 (16.2%) | 2025-11-29 | 2025-12-19 06:00 |
| `ara2_b3d128ac` | 2,184 | 1,747 | 0 | 437 | 230 (13.2%) | 63 (14.4%) | 2025-12-14 | 2026-01-21 05:39 |
| `ara2_c9cdacda` | 4,170 | 3,328 | 8 | 834 | 178 (5.3%) | 102 (12.2%) | 2025-12-03 | 2026-01-18 17:33 |
| `ara2_ca187575` | 2,045 | 1,629 | 7 | 409 | 191 (11.7%) | 90 (22.0%) | 2025-12-14 | 2026-01-13 11:48 |
| `ara2_d52ddb31` | 2,549 | 2,035 | 4 | 510 | 325 (16.0%) | 80 (15.7%) | 2025-12-14 | 2026-01-19 14:51 |

---

## 5. Verification of Zero Temporal Leakage

For every individual pond $p \in \{1, \dots, 17\}$, we verify the interval non-overlap condition:
$$\Delta t = \min(t_{\text{test}}) - \max(t_{\text{train}}) \ge 2.0\text{ hours}$$

All 17 ponds strictly satisfy $\Delta t \ge 2.0\text{ hours}$, proving that:
1. No future feature values leak into the past.
2. No future target evaluation horizons overlap between train and test sets.
3. Model evaluation represents a faithful simulation of real-world operational deployment.
