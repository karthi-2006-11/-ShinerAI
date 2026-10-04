"""
Unit tests for src/calibration.py
Verifies:
1. Calibration diagnostics (Brier score, ECE) on synthetic predictions.
2. Platt scaling and Isotonic regression calibrators.
3. Substantial Brier score reduction on held-out test predictions.
4. Cost-sensitive decision threshold optimization.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.model_utils import create_temporal_split, CONFIG_C_FEATURES
from src.calibration import (
    compute_calibration_diagnostics,
    fit_and_evaluate_calibrators,
    analyze_cost_sensitive_thresholds,
)


@pytest.fixture(scope="module")
def model_and_splits():
    data_path = Path("data/processed/ml_ready_dataset.csv")
    model_path = Path("models/xgboost_config_c.joblib")
    assert data_path.exists() and model_path.exists()

    df = pd.read_csv(data_path)
    train_df, test_df, _, _ = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)
    model = joblib.load(model_path)

    X_train = train_df[CONFIG_C_FEATURES].values
    y_train = train_df["target"].values
    train_probs = model.predict_proba(X_train)[:, 1]

    X_test = test_df[CONFIG_C_FEATURES].values
    y_test = test_df["target"].values
    test_probs = model.predict_proba(X_test)[:, 1]

    return {
        "y_train": y_train,
        "train_probs": train_probs,
        "y_test": y_test,
        "test_probs": test_probs,
    }


def test_synthetic_calibration_diagnostics():
    """Verify Brier score on perfect vs terrible predictions."""
    y_true = np.array([0, 0, 1, 1])
    # Perfect predictions
    perfect_probs = np.array([0.0, 0.0, 1.0, 1.0])
    diag_perf = compute_calibration_diagnostics(y_true, perfect_probs)
    assert diag_perf["brier_score"] == 0.0

    # Completely wrong predictions
    inverted_probs = np.array([1.0, 1.0, 0.0, 0.0])
    diag_inv = compute_calibration_diagnostics(y_true, inverted_probs)
    assert diag_inv["brier_score"] == 1.0


def test_real_model_calibration_reduction(model_and_splits):
    """Verify that Platt scaling and Isotonic regression substantially improve calibration."""
    data = model_and_splits
    res = fit_and_evaluate_calibrators(
        data["y_train"], data["train_probs"], data["y_test"], data["test_probs"]
    )

    uncal_brier = res["uncalibrated"]["brier_score"]
    platt_brier = res["platt_scaling"]["brier_score"]
    iso_brier = res["isotonic_regression"]["brier_score"]

    assert uncal_brier == pytest.approx(0.0863, abs=1e-3)
    assert platt_brier == pytest.approx(0.0503, abs=1e-3)
    assert iso_brier == pytest.approx(0.0503, abs=1e-3)

    # Reduction should be >= 40%
    reduction = (uncal_brier - platt_brier) / uncal_brier
    assert reduction >= 0.40


def test_cost_sensitive_threshold_optimization(model_and_splits):
    """Verify cost-sensitive threshold search confirms tau=0.50 under 5:1 penalty ratio."""
    data = model_and_splits
    cost_res = analyze_cost_sensitive_thresholds(
        data["y_train"], data["train_probs"], data["y_test"], data["test_probs"]
    )

    opt_train = cost_res["optimal_thresholds_on_train"]
    assert "5:1" in opt_train
    # Confirms tau=0.50 is the optimal threshold on training set for 5:1 loss
    assert opt_train["5:1"] == 0.50
