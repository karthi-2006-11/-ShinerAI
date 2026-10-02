"""
Unit tests for Phase 3 machine learning models, artifacts, and evaluation metrics.
Verifies no feature leakage, valid probability boundaries, artifact loading, and metric integrity.
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.model_utils import (
    FEATURE_CONFIGS,
    CONFIG_A_FEATURES,
    CONFIG_B_FEATURES,
    CONFIG_C_FEATURES,
    NON_PREDICTOR_COLUMNS,
    REDUNDANT_CURRENT_COLUMNS,
    compute_classification_metrics,
)


def test_feature_sets_exclude_targets_and_metadata():
    """Verify that none of the feature configurations include targets, metadata, or redundant columns."""
    for cfg_name, features in FEATURE_CONFIGS.items():
        # No target or metadata columns
        for col in NON_PREDICTOR_COLUMNS:
            assert col not in features, f"{col} erroneously present in {cfg_name}"

        # No redundant current columns
        for col in REDUNDANT_CURRENT_COLUMNS:
            assert col not in features, f"{col} erroneously present in {cfg_name}"


def test_feature_counts():
    """Verify exact feature counts for Configs A, B, and C."""
    assert len(CONFIG_A_FEATURES) == 5
    assert len(CONFIG_B_FEATURES) == 29
    assert len(CONFIG_C_FEATURES) == 11


def test_saved_model_artifacts_load_and_predict():
    """Verify that saved joblib models can be loaded and generate valid probabilities on test data."""
    models_dir = "models"
    data_path = os.path.join("data", "processed", "ml_ready_dataset.csv")
    df = pd.read_csv(data_path, nrows=50)

    X_sample = df[CONFIG_B_FEATURES].values

    for model_file in ["logistic_regression.joblib", "random_forest.joblib", "xgboost.joblib"]:
        model_path = os.path.join(models_dir, model_file)
        assert os.path.exists(model_path), f"Artifact {model_path} missing"

        clf = joblib.load(model_path)
        probs = clf.predict_proba(X_sample)[:, 1]
        preds = (probs >= 0.50).astype(int)

        # Probabilities strictly in [0.0, 1.0]
        assert np.all((probs >= 0.0) & (probs <= 1.0)), f"{model_file} produced probabilities outside [0, 1]"
        # Predictions strictly in {0, 1}
        assert set(preds).issubset({0, 1}), f"{model_file} produced invalid class labels"


def test_metrics_computation():
    """Verify that compute_classification_metrics outputs correct bounds and keys."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.7, 0.15])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(y_true)


def test_evaluation_reports_and_figures_exist():
    """Verify that all Phase 3 report CSVs and PNG figures exist and are non-empty."""
    expected_reports = [
        os.path.join("results", "reports", "model_comparison.csv"),
        os.path.join("results", "reports", "model_comparison.json"),
        os.path.join("results", "reports", "per_pond_model_performance.csv"),
        os.path.join("results", "reports", "group_kfold_performance.csv"),
        os.path.join("results", "reports", "temporal_split_accounting.csv"),
    ]
    for r in expected_reports:
        assert os.path.exists(r), f"Missing report: {r}"
        assert os.path.getsize(r) > 50, f"Report {r} appears empty"

    expected_figures = [
        os.path.join("results", "figures", "confusion_logistic_regression.png"),
        os.path.join("results", "figures", "confusion_random_forest.png"),
        os.path.join("results", "figures", "confusion_xgboost.png"),
        os.path.join("results", "figures", "roc_curve_comparison.png"),
        os.path.join("results", "figures", "pr_curve_comparison.png"),
        os.path.join("results", "figures", "ml_pipeline_diagram.png"),
    ]
    for f in expected_figures:
        assert os.path.exists(f), f"Missing figure: {f}"
        assert os.path.getsize(f) > 1000, f"Figure {f} appears empty or corrupt"
