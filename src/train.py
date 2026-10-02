"""
ShinerAI: Model Training Pipeline
Phase 3 — Machine Learning Training + Evaluation

This script:
1. Verifies data/processed/ml_ready_dataset.csv integrity.
2. Performs a leak-free temporal holdout split (80% train, 20% test) with a 2-hour purge gap.
3. Evaluates Majority and Current-DO baselines.
4. Trains and evaluates Logistic Regression, Random Forest, and XGBoost across:
   - Configuration A: Current Only (5 features)
   - Configuration B: Current + Full History (29 features)
   - Configuration C: DO History Only (11 features)
5. Saves trained model artifacts under models/.
6. Generates results/reports/model_comparison.csv and results/reports/model_comparison.json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import (
    verify_dataset_preconditions,
    create_temporal_split,
    compute_classification_metrics,
    get_model,
    FEATURE_CONFIGS,
    CONFIG_A_FEATURES,
    CONFIG_B_FEATURES,
    CONFIG_C_FEATURES,
)


def run_training_pipeline() -> pd.DataFrame:
    """Executes the full Phase 3 training and evaluation pipeline."""
    # 1. Paths (project-relative)
    data_path = os.path.join("data", "processed", "ml_ready_dataset.csv")
    models_dir = "models"
    reports_dir = os.path.join("results", "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("================================================================================")
    print("ShinerAI: Phase 3 Machine Learning Training Pipeline")
    print("================================================================================")

    # 2. Ingestion & Preconditions
    print(f"Ingesting dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at {data_path}. Run Phase 2 cleaning first.")

    df = pd.read_csv(data_path)
    audit_res = verify_dataset_preconditions(df)
    print(f"Precondition checks PASSED: {audit_res['rows']} rows, {audit_res['columns']} columns, "
          f"17 ponds, {audit_res['safe_count']} SAFE, {audit_res['at_risk_count']} AT_RISK.")

    # 3. Leak-Safe Temporal Split with 2-Hour Purge
    print("\nExecuting leak-safe temporal split (80% train / 20% test per pond with 2h purge)...")
    train_df, test_df, purged_df, accounting_df = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)

    # Save accounting report
    accounting_path = os.path.join(reports_dir, "temporal_split_accounting.csv")
    accounting_df.to_csv(accounting_path, index=False)
    print(f"Saved temporal split accounting to {accounting_path}")

    y_train = train_df["target"].values
    y_test = test_df["target"].values
    n_train_pos = int(y_train.sum())
    n_test_pos = int(y_test.sum())

    print(f"Split Summary:")
    print(f"  Train Set:  {len(train_df):,} examples | SAFE: {len(train_df)-n_train_pos:,} | "
          f"AT_RISK: {n_train_pos:,} ({y_train.mean():.2%})")
    print(f"  Purged Set: {len(purged_df):,} examples (2-hour boundary purge to prevent label leakage)")
    print(f"  Test Set:   {len(test_df):,} examples | SAFE: {len(test_df)-n_test_pos:,} | "
          f"AT_RISK: {n_test_pos:,} ({y_test.mean():.2%})")

    scale_pos_weight = float((len(y_train) - n_train_pos) / n_train_pos)
    print(f"  Calculated scale_pos_weight for training partition: {scale_pos_weight:.2f}")

    comparison_records = []

    # 4. Baselines
    print("\n--- Evaluating Baseline Models ---")

    # 4.1 Majority Class Baseline
    maj_prob = np.full_like(y_test, y_train.mean(), dtype=float)
    maj_metrics = compute_classification_metrics(y_test, maj_prob, threshold=0.50)
    comparison_records.append({
        "model": "Majority Baseline",
        "feature_set": "None",
        "evaluation_scheme": "Temporal Holdout (2h purge)",
        **maj_metrics,
    })
    print(f"Majority Baseline        -> PR-AUC: {maj_metrics['pr_auc']:.4f} | Recall: {maj_metrics['recall']:.4f} | F1: {maj_metrics['f1']:.4f} | Accuracy: {maj_metrics['accuracy']:.4f}")

    # 4.2 Current-DO Baseline
    # Search threshold strictly on training data that maximizes F1
    best_th = 3.5
    best_f1 = 0.0
    for th in np.arange(3.0, 6.0, 0.1):
        th_pred = (train_df["current_do"].values <= th).astype(int)
        f = f1_score(y_train, th_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = round(float(th), 2)

    # 1D logistic regression strictly on current_do to generate well-calibrated probabilities
    lr_1d = LogisticRegression(class_weight="balanced", random_state=42)
    lr_1d.fit(train_df[["current_do"]], y_train)
    do_base_prob = lr_1d.predict_proba(test_df[["current_do"]])[:, 1]

    do_base_metrics = compute_classification_metrics(y_test, do_base_prob, threshold=0.50)
    comparison_records.append({
        "model": f"Current-DO Baseline (DO <= {best_th:.1f})",
        "feature_set": "current_do only",
        "evaluation_scheme": "Temporal Holdout (2h purge)",
        **do_base_metrics,
    })
    print(f"Current-DO Baseline      -> PR-AUC: {do_base_metrics['pr_auc']:.4f} | Recall: {do_base_metrics['recall']:.4f} | F1: {do_base_metrics['f1']:.4f} | Accuracy: {do_base_metrics['accuracy']:.4f} (Threshold <= {best_th:.1f} mg/L)")

    # 5. Machine Learning Models across Configurations
    configs = {
        "Config A (Current Only)": CONFIG_A_FEATURES,
        "Config B (Current + History)": CONFIG_B_FEATURES,
        "Config C (DO History Only)": CONFIG_C_FEATURES,
    }

    trained_models = {}

    for config_name, feature_list in configs.items():
        print(f"\n--- Training on {config_name} ({len(feature_list)} features) ---")
        X_train = train_df[feature_list].values
        X_test = test_df[feature_list].values

        for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            clf = get_model(m_name, scale_pos_weight=scale_pos_weight, random_state=42)
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_test)[:, 1]
            metrics = compute_classification_metrics(y_test, y_prob, threshold=0.50)

            comparison_records.append({
                "model": m_name,
                "feature_set": config_name,
                "evaluation_scheme": "Temporal Holdout (2h purge)",
                **metrics,
            })

            print(f"{m_name:<24} -> PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f} | Specificity: {metrics['specificity']:.4f}")

            # Keep candidate models trained on Config B (full inputs) for disk serialization
            if config_name == "Config B (Current + History)":
                key = m_name.lower().replace(" ", "_")
                trained_models[key] = {
                    "model": clf,
                    "features": feature_list,
                    "metrics": metrics,
                }

    # 6. Save Model Comparison Reports
    comparison_df = pd.DataFrame(comparison_records)
    csv_report_path = os.path.join(reports_dir, "model_comparison.csv")
    json_report_path = os.path.join(reports_dir, "model_comparison.json")

    comparison_df.to_csv(csv_report_path, index=False)
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    print(f"\nSaved model comparison reports to:\n  - {csv_report_path}\n  - {json_report_path}")

    # 7. Serialize Final Candidate Models & Metadata
    print("\n--- Serializing Candidate Model Artifacts ---")
    for key, data in trained_models.items():
        artifact_path = os.path.join(models_dir, f"{key}.joblib")
        joblib.dump(data["model"], artifact_path)
        print(f"Saved model artifact: {artifact_path}")

    # Model metadata
    metadata = {
        "project": "ShinerAI",
        "phase": 3,
        "problem_statement": "Predict whether DO < 3.0 mg/L within next 2 hours given current DO >= 3.0 mg/L",
        "primary_metric": "PR-AUC (Average Precision)",
        "train_data": {
            "path": data_path,
            "total_examples": len(df),
            "train_examples": len(train_df),
            "test_examples": len(test_df),
            "purged_examples": len(purged_df),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
            "ponds_count": 17,
        },
        "split_method": {
            "type": "Temporal holdout per pond",
            "train_ratio": 0.80,
            "purge_gap_hours": 2.0,
            "rationale": "Prevents training labels from using future sensor data from test period",
        },
        "feature_configurations": {
            "config_a": {"description": "Current values and time of day", "feature_count": len(CONFIG_A_FEATURES), "features": CONFIG_A_FEATURES},
            "config_b": {"description": "Current values, time of day, and all historical lags", "feature_count": len(CONFIG_B_FEATURES), "features": CONFIG_B_FEATURES},
            "config_c": {"description": "DO history and time of day", "feature_count": len(CONFIG_C_FEATURES), "features": CONFIG_C_FEATURES},
        },
        "library_versions": {
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "models": {
            key: {
                "features": data["features"],
                "metrics": data["metrics"],
            }
            for key, data in trained_models.items()
        },
        "random_seed": 42,
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata: {metadata_path}")

    print("\nPhase 3 Training Complete!")
    return comparison_df


if __name__ == "__main__":
    run_training_pipeline()
