"""
ShinerAI: Model Evaluation & Visualizations
Phase 3 — Machine Learning Training + Evaluation

This script:
1. Loads trained candidate models from models/.
2. Performs comprehensive evaluation on the temporal holdout test partition.
3. Generates per-pond test performance breakdown (results/reports/per_pond_model_performance.csv).
4. Conducts 5-Fold GroupKFold unseen-pond generalization evaluation (results/reports/group_kfold_performance.csv).
5. Generates publication-ready figures under results/figures/:
   - confusion_logistic_regression.png
   - confusion_random_forest.png
   - confusion_xgboost.png
   - roc_curve_comparison.png
   - pr_curve_comparison.png
   - ml_pipeline_diagram.png
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
    f1_score,
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import (
    create_temporal_split,
    get_model,
    CONFIG_B_FEATURES,
)


def run_evaluation_pipeline():
    """Executes the full evaluation, cross-validation, and visualization suite."""
    data_path = os.path.join("data", "processed", "ml_ready_dataset.csv")
    models_dir = "models"
    reports_dir = os.path.join("results", "reports")
    figures_dir = os.path.join("results", "figures")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("================================================================================")
    print("ShinerAI: Phase 3 Model Evaluation & Visualizations")
    print("================================================================================")

    df = pd.read_csv(data_path)
    train_df, test_df, purged_df, _ = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)

    X_test = test_df[CONFIG_B_FEATURES].values
    y_test = test_df["target"].values

    # Load candidate models
    models = {
        "Logistic Regression": joblib.load(os.path.join(models_dir, "logistic_regression.joblib")),
        "Random Forest": joblib.load(os.path.join(models_dir, "random_forest.joblib")),
        "XGBoost": joblib.load(os.path.join(models_dir, "xgboost.joblib")),
    }

    # ==========================================================================
    # 1. GENERATE CONFUSION MATRIX FIGURES
    # ==========================================================================
    print("\n1. Generating Confusion Matrix Figures...")
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SAFE (0)", "AT_RISK (1)"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix: {name}\nTemporal Holdout (Threshold = 0.50)")

        key = name.lower().replace(" ", "_")
        fig_path = os.path.join(figures_dir, f"confusion_{key}.png")
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    # ==========================================================================
    # 2. GENERATE ROC & PR CURVE COMPARISONS
    # ==========================================================================
    print("\n2. Generating ROC and Precision-Recall Curves...")

    # ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6), dpi=150)
    # PR Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6), dpi=150)

    # Add no-skill baselines
    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="No Skill / Random (AUC = 0.5000)")
    pos_rate = float(y_test.mean())
    ax_pr.plot([0, 1], [pos_rate, pos_rate], linestyle="--", label=f"Majority Baseline (AP = {pos_rate:.4f})")

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.4f})")

        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap_score = average_precision_score(y_test, y_prob)
        ax_pr.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {ap_score:.4f})")

    ax_roc.set_title("ROC Curve Comparison on Temporal Test Holdout")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    roc_fig_path = os.path.join(figures_dir, "roc_curve_comparison.png")
    fig_roc.savefig(roc_fig_path)
    plt.close(fig_roc)
    print(f"  Saved: {roc_fig_path}")

    ax_pr.set_title("Precision-Recall Curve Comparison on Temporal Test Holdout")
    ax_pr.set_xlabel("Recall (Coverage of Low-DO Events)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    ax_pr.legend(loc="upper right")
    fig_pr.tight_layout()
    pr_fig_path = os.path.join(figures_dir, "pr_curve_comparison.png")
    fig_pr.savefig(pr_fig_path)
    plt.close(fig_pr)
    print(f"  Saved: {pr_fig_path}")

    # ==========================================================================
    # 3. PER-POND PERFORMANCE BREAKDOWN
    # ==========================================================================
    print("\n3. Calculating Per-Pond Performance Breakdown...")
    pond_records = []

    for pond_id, p_group in test_df.groupby("pond_id"):
        X_p = p_group[CONFIG_B_FEATURES].values
        y_p = p_group["target"].values
        total_p = len(p_group)
        safe_p = int((y_p == 0).sum())
        at_risk_p = int((y_p == 1).sum())

        for name, model in models.items():
            y_prob_p = model.predict_proba(X_p)[:, 1]
            y_pred_p = (y_prob_p >= 0.50).astype(int)

            if at_risk_p > 0:
                p_pr_auc = float(average_precision_score(y_p, y_prob_p))
                p_rec = float(recall_score(y_p, y_pred_p, zero_division=0))
                p_prec = float(precision_score(y_p, y_pred_p, zero_division=0))
                p_f1 = float(f1_score(y_p, y_pred_p, zero_division=0))
            else:
                p_pr_auc = 0.0
                p_rec = 0.0
                p_prec = 0.0
                p_f1 = 0.0

            pond_records.append({
                "pond_id": pond_id,
                "model": name,
                "test_examples": total_p,
                "safe_examples": safe_p,
                "at_risk_examples": at_risk_p,
                "precision": round(p_prec, 4),
                "recall": round(p_rec, 4),
                "f1": round(p_f1, 4),
                "pr_auc": round(p_pr_auc, 4),
            })

    pond_df = pd.DataFrame(pond_records)
    pond_report_path = os.path.join(reports_dir, "per_pond_model_performance.csv")
    pond_df.to_csv(pond_report_path, index=False)
    print(f"  Saved: {pond_report_path}")

    # ==========================================================================
    # 4. UNSEEN-POND GENERALIZATION (5-FOLD GROUPKFOLD)
    # ==========================================================================
    print("\n4. Running 5-Fold GroupKFold Unseen-Pond Generalization...")
    gkf = GroupKFold(n_splits=5)
    groups = df["pond_id"]

    gkf_records = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
        tr_sub = df.iloc[train_idx]
        val_sub = df.iloc[val_idx]

        test_ponds = sorted(val_sub["pond_id"].unique())
        X_tr = tr_sub[CONFIG_B_FEATURES].values
        y_tr = tr_sub["target"].values
        X_val = val_sub[CONFIG_B_FEATURES].values
        y_val = val_sub["target"].values

        spw = float((len(y_tr) - sum(y_tr)) / sum(y_tr))

        fold_clfs = {
            "Logistic Regression": get_model("logistic_regression", scale_pos_weight=spw, random_state=42),
            "Random Forest": get_model("random_forest", scale_pos_weight=spw, random_state=42),
            "XGBoost": get_model("xgboost", scale_pos_weight=spw, random_state=42),
        }

        for name, clf in fold_clfs.items():
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.50).astype(int)

            pr = float(average_precision_score(y_val, probs))
            roc = float(roc_auc_score(y_val, probs))
            f1 = float(f1_score(y_val, preds, zero_division=0))
            rec = float(recall_score(y_val, preds, zero_division=0))
            prec = float(precision_score(y_val, preds, zero_division=0))

            gkf_records.append({
                "fold": fold,
                "test_ponds": ", ".join(test_ponds),
                "model": name,
                "test_examples": len(val_sub),
                "test_at_risk": int(y_val.sum()),
                "pr_auc": round(pr, 4),
                "roc_auc": round(roc, 4),
                "f1": round(f1, 4),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
            })

    gkf_df = pd.DataFrame(gkf_records)
    gkf_report_path = os.path.join(reports_dir, "group_kfold_performance.csv")
    gkf_df.to_csv(gkf_report_path, index=False)
    print(f"  Saved: {gkf_report_path}")

    # ==========================================================================
    # 5. PHASE 3 ML PIPELINE DIAGRAM
    # ==========================================================================
    print("\n5. Generating Phase 3 ML Pipeline Architecture Diagram...")
    fig_diag, ax_diag = plt.subplots(figsize=(10, 12), dpi=150)
    ax_diag.set_xlim(0, 10)
    ax_diag.set_ylim(0, 12)
    ax_diag.axis("off")

    def draw_diag_box(x, y, w, h, text, color="#EBF3FB", edgecolor="#2B6CB0", bold=False, fontsize=10):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      facecolor=color, edgecolor=edgecolor, linewidth=1.5)
        ax_diag.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax_diag.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)

    def draw_diag_arrow(x1, y1, x2, y2, label=""):
        ax_diag.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))
        if label:
            ax_diag.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label, fontsize=9, color="#2D3748", fontweight="bold")

    ax_diag.text(5, 11.5, "ShinerAI: Phase 3 Machine Learning Training & Evaluation Architecture",
                 ha="center", va="center", fontsize=12, fontweight="bold", color="#1A365D")

    draw_diag_box(2.5, 10.3, 5.0, 0.7, "ML-Ready Dataset (41,277 Examples, 37 Columns)\nStrictly Formatted from Phase 2", "#EBF8FF", "#3182CE", bold=True)
    draw_diag_arrow(5.0, 10.3, 5.0, 9.6)

    draw_diag_box(2.5, 8.9, 5.0, 0.7, "Feature Selection & Omission of Redundancies\n(Omit do_t, ph_t, temp_t; Configs A, B, C)", "#FEFCBF", "#D69E2E")
    draw_diag_arrow(5.0, 8.9, 5.0, 8.2)

    draw_diag_box(2.5, 7.3, 5.0, 0.9, "Leakage-Safe Temporal Holdout Split\n80% Train (32,908) | 2h Purge (108) | 20% Test (8,261)\nStrictly Evaluated per Pond Chronology", "#C6F6D5", "#2F855A", bold=True)

    draw_diag_arrow(3.5, 7.3, 2.5, 6.2)
    draw_diag_arrow(6.5, 7.3, 7.5, 6.2)

    draw_diag_box(0.5, 4.8, 4.0, 1.4, "Model Training (Train Set Only)\n- Logistic Regression (StandardScaler)\n- Random Forest (Balanced Weights)\n- XGBoost (scale_pos_weight = 6.80)\nTested on Configs A, B, C", "#EBF3FB", "#2B6CB0")
    draw_diag_box(5.5, 4.8, 4.0, 1.4, "Generalization Testing\n- Temporal Holdout Test (8,261 rows)\n- 5-Fold GroupKFold Unseen Ponds\n- Per-Pond Breakdown across 17 Ponds", "#FAF5FF", "#6B46C1")

    draw_diag_arrow(2.5, 4.8, 4.0, 3.8)
    draw_diag_arrow(7.5, 4.8, 6.0, 3.8)

    draw_diag_box(2.0, 2.8, 6.0, 1.0, "Model Evaluation & Selection\nPrimary Metric: PR-AUC (Average Precision)\nSecondary: F1, Recall, Specificity, Confusion Matrices", "#EDF2F7", "#4A5568", bold=True)
    draw_diag_arrow(5.0, 2.8, 5.0, 2.0)

    draw_diag_box(2.0, 1.1, 6.0, 0.9, "Selected Candidate Model Artifacts (models/*.joblib)\nReady for Phase 4 Farm Advisory Integration", "#BEE3F8", "#2B6CB0", bold=True)

    fig_diag.tight_layout()
    diag_path = os.path.join(figures_dir, "ml_pipeline_diagram.png")
    fig_diag.savefig(diag_path, bbox_inches="tight")
    plt.close(fig_diag)
    print(f"  Saved: {diag_path}")

    print("\nPhase 3 Evaluation Complete!")


if __name__ == "__main__":
    run_evaluation_pipeline()
