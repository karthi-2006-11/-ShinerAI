"""
ShinerAI: Generate Mentor-Ready Model Comparison Tables & Visualizations
Parts A, B, and C implementation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_comparisons():
    master_csv = Path("results/reports/MASTER_MODEL_EVALUATION.csv")
    assert master_csv.exists(), f"Missing master evaluation: {master_csv}"
    df = pd.read_csv(master_csv)

    reports_dir = Path("results/reports")
    figures_dir = Path("results/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART A: Consolidated Model Comparison Table
    # =========================================================================
    # Map friendly names
    part_a_rows = []
    for _, r in df.iterrows():
        model_name = str(r["model"])
        feat_raw = r["feature_set"]
        feat_name = "" if pd.isna(feat_raw) else str(feat_raw)
        if feat_name in ["", "None", "nan"]:
            feat_desc = "None (Class Distribution Only)"
        elif feat_name == "current_do only":
            feat_desc = "Current DO Only (Static Threshold <= 4.2 mg/L)"
        elif "Config A" in feat_name:
            feat_desc = "Config A — Current DO + Time (5 Predictors)"
        elif "Config B" in feat_name:
            feat_desc = "Config B — Current + DO/pH/Temp History (29 Predictors)"
        elif "Config C" in feat_name:
            feat_desc = "Config C — Current + Recent DO History (11 Predictors)"
        else:
            feat_desc = feat_name

        part_a_rows.append({
            "Model": model_name,
            "Feature Configuration": feat_desc,
            "PR-AUC": f"{float(r['pr_auc']):.4f}",
            "ROC-AUC": f"{float(r['roc_auc']):.4f}",
            "F1": f"{float(r['f1']):.4f}",
            "Recall": f"{float(r['recall']):.4f}",
            "Precision": f"{float(r['precision']):.4f}",
            "Specificity": f"{float(r['specificity']):.4f}",
            "Accuracy": f"{float(r['accuracy']):.4f}",
        })

    df_part_a = pd.DataFrame(part_a_rows)
    df_part_a.to_csv(reports_dir / "MODEL_COMPARISON.csv", index=False)

    # Markdown for Part A
    md_a = [
        "# ShinerAI: Consolidated Model Comparison (Internal Temporal Holdout)",
        "",
        "**Dataset:** Commercial Golden Shiner Aquaculture Dataset (17 Earthen Ponds, Lonoke County, AR, USA)  ",
        "**Evaluation Scheme:** Chronological 80/20 Holdout with a Strict 2.0-Hour Purge Gap ($N = 8,261$ test observations)  ",
        "**Positive Prevalence:** 11.42% (943 low-DO events $< 3.0$ mg/L)  ",
        "**Decision Threshold:** $\\tau = 0.50$ (applied across all classifiers)  ",
        "",
        "| Model | Feature Configuration | PR-AUC | ROC-AUC | F1 | Recall | Precision | Specificity | Accuracy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in part_a_rows:
        md_a.append(f"| {r['Model']} | {r['Feature Configuration']} | **{r['PR-AUC']}** | {r['ROC-AUC']} | {r['F1']} | {r['Recall']} | {r['Precision']} | {r['Specificity']} | {r['Accuracy']} |")
    
    md_a.append("")
    md_a.append("> **Note:** All values reflect exact verified metrics from `results/reports/MASTER_MODEL_EVALUATION.csv`. XGBoost Config C serves as the production frozen early warning model.")
    (reports_dir / "MODEL_COMPARISON.md").write_text("\n".join(md_a), encoding="utf-8")
    print("Saved Part A tables to results/reports/MODEL_COMPARISON.[csv/md]")

    # =========================================================================
    # PART B: Feature-Ablation Comparison
    # =========================================================================
    # Extract PR-AUC for LR, RF, XGB on Config A, B, C
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    ablation_rows = []

    for m in models:
        pr_a = float(df[(df["model"] == m) & (df["feature_set"].str.contains("Config A"))]["pr_auc"].values[0])
        pr_b = float(df[(df["model"] == m) & (df["feature_set"].str.contains("Config B"))]["pr_auc"].values[0])
        pr_c = float(df[(df["model"] == m) & (df["feature_set"].str.contains("Config C"))]["pr_auc"].values[0])

        diff_b_minus_a = pr_b - pr_a
        diff_c_minus_a = pr_c - pr_a
        diff_c_minus_b = pr_c - pr_b

        ablation_rows.append({
            "Model": m,
            "Config A PR-AUC (Current Only)": f"{pr_a:.4f}",
            "Config B PR-AUC (Full History)": f"{pr_b:.4f}",
            "Config C PR-AUC (DO History Only)": f"{pr_c:.4f}",
            "Config B - Config A": f"{diff_b_minus_a:+.4f}",
            "Config C - Config A": f"{diff_c_minus_a:+.4f}",
            "Config C - Config B": f"{diff_c_minus_b:+.4f}",
        })

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(reports_dir / "FEATURE_ABLATION_COMPARISON.csv", index=False)

    md_b = [
        "# ShinerAI: Feature Ablation Comparison (Value of Historical Telemetry)",
        "",
        "This ablation analysis directly addresses mentor feedback by quantifying whether **recent dissolved-oxygen history adds predictive information** beyond current-time measurements alone.",
        "",
        "### Feature Configurations Evaluated:",
        "- **Config A (Current Only, 5 features):** `current_do`, `current_ph`, `current_temperature`, `hour_of_day`, `minute_of_day`.",
        "- **Config B (Current + History, 29 features):** Current values + 8 lags each of DO, pH, and Temperature ($t-15\\text{m}$ to $t-120\\text{m}$) + diurnal time features.",
        "- **Config C (DO History Only, 11 features):** `current_do` + 8 historical DO lags ($t-15\\text{m}$ to $t-120\\text{m}$) + diurnal time features.",
        "",
        "### Ablation Results (PR-AUC Metric):",
        "",
        "| Model | Config A PR-AUC | Config B PR-AUC | Config C PR-AUC | Config B - Config A | Config C - Config A | Config C - Config B |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ablation_rows:
        md_b.append(f"| **{r['Model']}** | {r['Config A PR-AUC (Current Only)']} | {r['Config B PR-AUC (Full History)']} | {r['Config C PR-AUC (DO History Only)']} | {r['Config B - Config A']} | **{r['Config C - Config A']}** | {r['Config C - Config B']} |")

    md_b.extend([
        "",
        "### Key Scientific Findings & Guardrails:",
        "1. **Recent DO History Adds Defensible Predictive Information:** For every evaluated algorithm, adding 2-hour DO history yields a decisive lift over current-only features alone (`Config C - Config A` lift: **+0.0531** for Logistic Regression, **+0.0364** for Random Forest, and **+0.0257** for XGBoost).",
        "2. **Model-Specific Behavior on Full vs DO History:** Config C does *not* strictly dominate Config B across all models. For linear Logistic Regression, auxiliary water parameters in Config B provide a higher lift (**0.6763** vs. **0.6550**), whereas for non-linear ensembles (Random Forest and XGBoost), isolating DO history eliminates collinear sensor noise and achieves the highest PR-AUC (**0.7471** and **0.7574** respectively).",
        "3. **No Explicit Derivative Engineered:** All historical features are discrete sensor observations ($t-15\\text{m}$ to $t-120\\text{m}$); no mathematical derivative or explicit 'DO velocity' feature was computed.",
    ])
    (reports_dir / "FEATURE_ABLATION_COMPARISON.md").write_text("\n".join(md_b), encoding="utf-8")
    print("Saved Part B tables to results/reports/FEATURE_ABLATION_COMPARISON.[csv/md]")

    # =========================================================================
    # PART C: Mentor Comparison Visualization
    # =========================================================================
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)

    # Color palette
    c_baseline = "#64748B"
    c_cfg_a = "#93C5FD"     # Light soft blue
    c_cfg_b = "#3B82F6"     # Medium blue
    c_cfg_c = "#0369A1"     # Strong deep navy blue
    c_champion = "#0284C7"  # Primary highlight

    # Ordered approaches
    labels = [
        "Majority\nBaseline",
        "Persistence\nBaseline",
        "120m Trend\nBaseline",
        "Current-DO\nHeuristic",
        "LogReg\n(Config A)",
        "LogReg\n(Config B)",
        "LogReg\n(Config C)",
        "RandomForest\n(Config A)",
        "RandomForest\n(Config B)",
        "RandomForest\n(Config C)",
        "XGBoost\n(Config A)",
        "XGBoost\n(Config B)",
        "XGBoost\n(Config C)*",
    ]
    praucs = [
        0.1142, 0.1142, 0.4656, 0.6149,
        0.6019, 0.6763, 0.6550,
        0.7107, 0.7420, 0.7471,
        0.7317, 0.7353, 0.7574,
    ]
    colors = [
        c_baseline, c_baseline, "#94A3B8", c_baseline,
        c_cfg_a, c_cfg_b, c_cfg_c,
        c_cfg_a, c_cfg_b, c_cfg_c,
        c_cfg_a, c_cfg_b, "#059669", # Emerald highlight for active champion
    ]

    x = np.arange(len(labels))
    bars = ax.bar(x, praucs, color=colors, width=0.68, edgecolor="#0F172A", linewidth=0.8, zorder=3)

    # Add numeric labels on top of bars
    for bar, val in zip(bars, praucs):
        y_pos = bar.get_height()
        font_weight = "bold" if val >= 0.75 else "normal"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos + 0.012,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight=font_weight,
            color="#0F172A",
        )

    # Reference lines for baselines
    ax.axhline(0.1142, color="#94A3B8", linestyle="--", linewidth=1.2, zorder=2, label="Majority Class Baseline (0.1142)")
    ax.axhline(0.6149, color="#E11D48", linestyle=":", linewidth=1.4, zorder=2, label="Current-DO Threshold Heuristic (0.6149)")

    # Grouping background bands
    ax.axvspan(-0.5, 3.5, color="#F1F5F9", alpha=0.5, zorder=1)
    ax.axvspan(3.5, 6.5, color="#F8FAFC", alpha=0.5, zorder=1)
    ax.axvspan(6.5, 9.5, color="#F1F5F9", alpha=0.5, zorder=1)
    ax.axvspan(9.5, 12.5, color="#F0FDF4", alpha=0.6, zorder=1)

    # Titles & labels
    ax.set_title(
        "ShinerAI: PR-AUC Model & Configuration Comparison\n(Internal Temporal Holdout — Original FWI Aquaculture Dataset)",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
        color="#0F172A",
    )
    ax.set_ylabel("PR-AUC (Primary Early Warning Metric)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color="#1E293B")
    ax.set_ylim(0, 0.86)

    # Grid & styling
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Annotation for Champion
    ax.annotate(
        "Production Frozen Model\n(PR-AUC = 0.7574)",
        xy=(10, 0.7574),
        xytext=(8.0, 0.81),
        arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=1),
    )

    # Custom legend for configurations
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c_baseline, edgecolor="#0F172A", label="Baselines (No ML / Static)"),
        Patch(facecolor=c_cfg_a, edgecolor="#0F172A", label="Config A (Current DO + Time)"),
        Patch(facecolor=c_cfg_b, edgecolor="#0F172A", label="Config B (Current + All Sensor History)"),
        Patch(facecolor=c_cfg_c, edgecolor="#0F172A", label="Config C (Current + DO History Only)"),
        Patch(facecolor="#059669", edgecolor="#0F172A", label="Active Champion (*XGBoost Config C)"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper left", fontsize=8.2, framealpha=0.92)
    leg.get_frame().set_edgecolor("#CBD5E1")

    fig.tight_layout()
    comp_fig_path = figures_dir / "model_comparison_prauc.png"
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"Saved Part C mentor visualization to: {comp_fig_path}")

if __name__ == "__main__":
    generate_comparisons()
