"""
Appends Sections 35, 36, 37, and 38 to notebooks/ShinerAI_Complete_ML_Pipeline.ipynb
Covers:
- Section 35: Stronger Domain Baselines (Persistence & 120-min Linear Trend)
- Section 36: Operational Event-Level Evaluation (Hypoxia Episodes, Lead-Time Distribution, False Alarm Burden, Hysteresis Filter)
- Section 37: Probability Calibration & Cost-Sensitive Threshold Analysis
- Section 38: External Dataset Audits & Oman Discrepancy Reconciliation
"""

import json
from pathlib import Path

notebook_path = Path("notebooks/ShinerAI_Complete_ML_Pipeline.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Check existing sections to prevent duplicate insertion
source_all = "".join(
    "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "markdown"
)
if "### Section 35: Stronger Domain Baselines" in source_all:
    print("Mentor sections already present in notebook.")
    exit(0)

new_cells = []

# ==============================================================================
# SECTION 35: STRONGER DOMAIN BASELINES
# ==============================================================================
md_sec35 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Section 35: Stronger Domain Baselines (Domain-Standard Non-ML Baselines)\n",
        "\n",
        "#### 35.1 Scientific Rationale\n",
        "In response to IEEE mentor review feedback, machine learning models must be benchmarked not only against a naive majority-class baseline and a single static threshold, but also against **domain-standard time-series baselines**:\n",
        "1. **Strict Persistence Baseline:** Assumes dissolved oxygen remains constant at its current reading $\\text{DO}(T)$ throughout the 2-hour future window. Under the ShinerAI operational invariant (current $\\text{DO} \\ge 3.0$ mg/L), strict persistence always forecasts that DO will remain safe ($\\hat{y} = 0$).\n",
        "2. **120-Minute Linear Trend Extrapolation Baseline:** Fits an ordinary least squares (OLS) linear slope over the 9 historical readings $[T-120\\text{m}, \\dots, T]$ and extrapolates linearly over the forward 2 hours. If the extrapolated trajectory breaches $3.0$ mg/L, it triggers an alert ($\\hat{y} = 1$).\n"
    ]
}

code_sec35 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# SECTION 35: DOMAIN BASELINES EVALUATION\n",
        "# ==============================================================================\n",
        "from src.baselines import evaluate_persistence_baseline, evaluate_trend_baseline\n",
        "\n",
        "print(\"=\" * 75)\n",
        "print(\"EVALUATING DOMAIN-STANDARD BASELINES ON HELD-OUT TEST SET (N = 8,261)\")\n",
        "print(\"=\" * 75)\n",
        "\n",
        "p_metrics = evaluate_persistence_baseline(test_df)\n",
        "t_metrics = evaluate_trend_baseline(test_df)\n",
        "\n",
        "print(f\"Strict Persistence Baseline:  PR-AUC = {p_metrics['pr_auc']:.4f}, Recall = {p_metrics['recall']:.4f}, Specificity = {p_metrics['specificity']:.4f}\")\n",
        "print(f\"120-min Linear Trend Baseline: PR-AUC = {t_metrics['pr_auc']:.4f}, ROC-AUC = {t_metrics['roc_auc']:.4f}, F1 = {t_metrics['f1']:.4f}, Recall = {t_metrics['recall']:.4f}, Precision = {t_metrics['precision']:.4f}\")\n",
        "\n",
        "# Benchmark lift of Champion XGBoost Config C over Linear Trend\n",
        "xgb_pr_auc = 0.7574\n",
        "trend_pr_auc = t_metrics['pr_auc']\n",
        "print(f\"\\nChampion XGBoost Lift over Linear Trend: +{xgb_pr_auc - trend_pr_auc:.4f} PR-AUC (+{(xgb_pr_auc - trend_pr_auc)/trend_pr_auc * 100:.1f}% relative)\")\n"
    ]
}

# ==============================================================================
# SECTION 36: OPERATIONAL EVENT-LEVEL EVALUATION
# ==============================================================================
md_sec36 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Section 36: Operational Event-Level Evaluation (Primary Novelty Contribution)\n",
        "\n",
        "#### 36.1 Bridging Point-Wise ML to Operational Farm Reality\n",
        "Conventional ML benchmarks evaluate point-wise binary classification on 15-minute sensor rows. However, in commercial fish farming:\n",
        "- Hypoxia occurs in multi-hour **nocturnal episodes** (typically 1 to 4 hours).\n",
        "- A single true early warning 1 to 2 hours before DO breaches 3.0 mg/L enables aerator deployment and prevents mortality.\n",
        "- Frequent isolated 15-minute false alarms cause operator alert fatigue.\n",
        "\n",
        "Here we group the 8,261 held-out test intervals into **136 contiguous impending hypoxia episodes** and quantify:\n",
        "1. **Event Detection Rate (EDR)** and Missed Event Rate.\n",
        "2. **Advance Warning Lead Time Distribution** (minutes before hypoxia crossing).\n",
        "3. **Daily False Alarm Burden** (alerts and false episodes per pond per day).\n",
        "4. **Operational Hysteresis Filtering** (2-consecutive interval confirmation to suppress chattering).\n"
    ]
}

code_sec36 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# SECTION 36: OPERATIONAL EVENT-LEVEL EVALUATION\n",
        "# ==============================================================================\n",
        "from src.operational_evaluation import (\n",
        "    extract_hypoxia_episodes,\n",
        "    compute_event_level_metrics,\n",
        "    compute_false_alarms_per_pond_day,\n",
        "    compute_alert_stability_metrics,\n",
        "    evaluate_hysteresis_filter,\n",
        ")\n",
        "\n",
        "test_eval_df = test_df.copy()\n",
        "test_eval_df[\"prob\"] = probs\n",
        "test_eval_df[\"pred\"] = (probs >= 0.50).astype(int)\n",
        "\n",
        "df_episodes = extract_hypoxia_episodes(test_eval_df)\n",
        "ev_metrics = compute_event_level_metrics(df_episodes)\n",
        "fa_metrics = compute_false_alarms_per_pond_day(test_eval_df)\n",
        "st_metrics = compute_alert_stability_metrics(test_eval_df)\n",
        "hy_metrics = evaluate_hysteresis_filter(test_eval_df, consecutive_steps_required=2)\n",
        "\n",
        "print(\"=\" * 75)\n",
        "print(\"OPERATIONAL & EVENT-LEVEL EARLY WARNING SUMMARY\")\n",
        "print(\"=\" * 75)\n",
        "print(f\"Total Hypoxia Episodes:       {ev_metrics['total_hypoxia_episodes']}\")\n",
        "print(f\"Detected Episodes:             {ev_metrics['detected_episodes']} ({ev_metrics['event_detection_rate']:.2%})\")\n",
        "print(f\"Missed Episodes:               {ev_metrics['missed_episodes']} ({ev_metrics['event_miss_rate']:.2%})\")\n",
        "print(f\"Mean Warning Lead Time:        {ev_metrics['mean_lead_time_minutes']:.1f} minutes\")\n",
        "print(f\"Median Warning Lead Time:      {ev_metrics['median_lead_time_minutes']:.1f} minutes\")\n",
        "print(f\"Daily False Alarm Burden:      {fa_metrics['mean_false_alarms_per_pond_day']:.2f} alerts/pond/day\")\n",
        "print(f\"False Alarm Chattering Rate:   {st_metrics['chattering_rate']*100:.1f}%\")\n",
        "print(f\"Hysteresis Filter (k=2):       Reduces FP intervals by {hy_metrics['fp_reduction_percent']:.1f}% (698 -> {hy_metrics['fp']})\")\n"
    ]
}

# ==============================================================================
# SECTION 37: PROBABILITY CALIBRATION & COST-SENSITIVE THRESHOLDS
# ==============================================================================
md_sec37 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Section 37: Probability Calibration & Cost-Sensitive Analysis\n",
        "\n",
        "#### 37.1 Calibration Quality & Brier Score\n",
        "Tree ensemble outputs on imbalanced classification often exhibit probability distortion. We evaluate:\n",
        "- **Brier Score Loss** and Expected Calibration Error (ECE).\n",
        "- **Post-Hoc Calibrators:** Platt Scaling (logistic sigmoid) and Isotonic Regression fitted strictly on the training partition.\n",
        "- **Cost-Sensitive Loss Optimization:** Demonstrating that default threshold $\\tau = 0.50$ is the exact empirical minimum for standard aquaculture asymmetric loss ($C_{FN} : C_{FP} = 5:1$).\n"
    ]
}

code_sec37 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# SECTION 37: CALIBRATION & COST-SENSITIVE OPTIMIZATION\n",
        "# ==============================================================================\n",
        "from src.calibration import fit_and_evaluate_calibrators, analyze_cost_sensitive_thresholds\n",
        "\n",
        "X_train = train_df[CONFIG_C_FEATURES].values\n",
        "y_train = train_df[\"target\"].values\n",
        "train_probs = model.predict_proba(X_train)[:, 1]\n",
        "\n",
        "cal_results = fit_and_evaluate_calibrators(y_train, train_probs, y_test, probs)\n",
        "cost_results = analyze_cost_sensitive_thresholds(y_train, train_probs, y_test, probs)\n",
        "\n",
        "brier_uncal = cal_results[\"uncalibrated\"][\"brier_score\"]\n",
        "brier_platt = cal_results[\"platt_scaling\"][\"brier_score\"]\n",
        "brier_iso = cal_results[\"isotonic_regression\"][\"brier_score\"]\n",
        "\n",
        "print(\"=\" * 75)\n",
        "print(\"PROBABILITY CALIBRATION RESULTS\")\n",
        "print(\"=\" * 75)\n",
        "print(f\"Uncalibrated Brier Score: {brier_uncal:.4f}\")\n",
        "print(f\"Platt Scaling Brier Score: {brier_platt:.4f} (-{(brier_uncal - brier_platt)/brier_uncal * 100:.1f}% error reduction)\")\n",
        "print(f\"Isotonic Regression Brier: {brier_iso:.4f} (-{(brier_uncal - brier_iso)/brier_uncal * 100:.1f}% error reduction)\")\n",
        "print(f\"Optimal Operating Threshold (5:1 loss): tau = {cost_results['optimal_thresholds_on_train']['5:1']:.2f}\")\n"
    ]
}

# ==============================================================================
# SECTION 38: EXTERNAL VALIDATION AUDIT SUMMARY
# ==============================================================================
md_sec38 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Section 38: External Datasets Audit Summary\n",
        "\n",
        "#### 38.1 Oman Nile Tilapia Discrepancy Reconciliation\n",
        "A formal investigation resolved the variance between reported zero false alarms vs. 5 false alarms:\n",
        "- **Clean QC Telemetry ($DO > 0$):** 0 False Positives, **100.0% Specificity** across all 3,808 non-hypoxic 15-minute test intervals.\n",
        "- **Unfiltered Raw Telemetry:** 5 False Positives, 99.87% Specificity, caused by two raw $0.0\\text{ mg/L}$ hardware sensor disconnect spikes.\n",
        "\n",
        "#### 38.2 Andhra Pradesh Aquaculture Dataset Feasibility Audit\n",
        "An audit of the Andhra Pradesh dataset (*Water Quality Research Journal* 2026; DOI: 10.2166/wqrj.2026.010; Kaggle) revealed:\n",
        "- **Temporal Cadence:** 20-minute sampling cadence, conflicting directly with ShinerAI's 15-minute feature contract (8 discrete historical lags).\n",
        "- **Methodological Integrity:** Synthetic temporal interpolation was strictly rejected to prevent fabricated telemetry. The dataset was certified as methodologically non-compliant without modifying model specifications.\n"
    ]
}

new_cells.extend([
    md_sec35, code_sec35,
    md_sec36, code_sec36,
    md_sec37, code_sec37,
    md_sec38
])

nb["cells"].extend(new_cells)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully added Sections 35, 36, 37, and 38 to {notebook_path}")
