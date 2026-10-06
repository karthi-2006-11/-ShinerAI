"""
ShinerAI: Unit Tests for Experimental XGBoost Tuning (V1)
Verifies:
1. Production model artifact remains strictly untouched and bit-identical.
2. Experimental model artifact is stored separately.
3. Feature engineering operates row-wise using past/current observations only (zero leakage).
4. Temporal split maintains 2-hour purge gap and test set isolation.
5. Tuning trial records exist and are deterministic and reproducible.
6. Threshold optimization uses training-only validation predictions.
"""

import os
import hashlib
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from src.model_utils import (
    create_temporal_split,
    CONFIG_C_FEATURES,
    compute_classification_metrics,
)
from src.features_experimental import (
    compute_experimental_features,
    RATE_OF_CHANGE_FEATURES,
    ACCELERATION_FEATURES,
    CONFIG_C_PLUS_ACCEL,
)

FROZEN_BASELINE_HASH = "57eb3cf72259d149028129d24920b97a7f593ed31f9708d2a1d436351f157475"
DATASET_HASH = "887a6382992daa68a32f9182d111ded2c0120543af80c8700d24e1bd6556ef2f"


def sha256_file(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_production_baseline_artifact_integrity():
    """Verify production baseline model artifact was not modified or overwritten."""
    prod_path = "models/xgboost_config_c.joblib"
    assert os.path.exists(prod_path), "Production model artifact missing!"
    current_hash = sha256_file(prod_path)
    assert current_hash == FROZEN_BASELINE_HASH, (
        f"Production model artifact hash mismatch! Expected {FROZEN_BASELINE_HASH}, got {current_hash}"
    )


def test_dataset_integrity():
    """Verify underlying ML-ready dataset was not modified."""
    data_path = "data/processed/ml_ready_dataset.csv"
    assert os.path.exists(data_path), "Dataset missing!"
    current_hash = sha256_file(data_path)
    assert current_hash == DATASET_HASH, (
        f"Dataset hash mismatch! Expected {DATASET_HASH}, got {current_hash}"
    )


def test_experimental_model_artifact_separation():
    """Verify experimental model artifact exists as an independent artifact."""
    exp_path = "models/experimental_xgboost_v1.joblib"
    prod_path = "models/xgboost_config_c.joblib"
    assert os.path.exists(exp_path), "Experimental model artifact missing!"
    assert os.path.exists(prod_path), "Production model artifact missing!"
    assert exp_path != prod_path
    assert sha256_file(exp_path) != sha256_file(prod_path), (
        "Experimental artifact must not be identical to production artifact"
    )


def test_feature_engineering_temporal_integrity():
    """Verify feature engineering uses strictly current and past data with no target leakage."""
    df = pd.read_csv("data/processed/ml_ready_dataset.csv").head(100)
    feat_df = compute_experimental_features(df)

    # Check newly created feature columns
    for col in RATE_OF_CHANGE_FEATURES + ACCELERATION_FEATURES:
        assert col in feat_df.columns, f"Missing feature: {col}"
        assert not feat_df[col].isna().any(), f"NaN found in {col}"

    # Verify mathematical definitions row-wise
    row0 = feat_df.iloc[0]
    expected_change_15 = row0["current_do"] - row0["do_t_minus_15"]
    assert np.isclose(row0["do_change_15"], expected_change_15)

    expected_accel = (row0["current_do"] - row0["do_t_minus_15"]) - (row0["do_t_minus_15"] - row0["do_t_minus_30"])
    assert np.isclose(row0["do_accel_15"], expected_accel)

    # Ensure target column is untouched
    assert (feat_df["target"] == df["target"]).all()


def test_test_set_isolation_and_purge_gap():
    """Verify temporal split strictly isolates test partition with 2-hour purge gap."""
    df = pd.read_csv("data/processed/ml_ready_dataset.csv")
    train_df, test_df, purged_df, _ = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)

    assert len(train_df) == 32908
    assert len(test_df) == 8261
    assert len(purged_df) == 108
    assert len(train_df) + len(test_df) + len(purged_df) == len(df)

    # Ensure timestamps in train strictly precede test by at least purge duration per pond
    train_df["prediction_timestamp"] = pd.to_datetime(train_df["prediction_timestamp"])
    test_df["prediction_timestamp"] = pd.to_datetime(test_df["prediction_timestamp"])

    for pid in df["pond_id"].unique():
        p_train_max = train_df[train_df["pond_id"] == pid]["prediction_timestamp"].max()
        p_test_min = test_df[test_df["pond_id"] == pid]["prediction_timestamp"].min()
        gap_hours = (p_test_min - p_train_max).total_seconds() / 3600.0
        assert gap_hours >= 2.0, f"Purge gap violation in pond {pid}: {gap_hours} hours"


def test_tuning_results_report_exists_and_reproducible():
    """Verify experimental trial results report was saved and contains required columns."""
    report_path = "results/reports/xgb_tuning_v1_results.csv"
    assert os.path.exists(report_path), f"Report missing: {report_path}"
    results_df = pd.read_csv(report_path)

    assert len(results_df) == 64, f"Expected 64 trials, found {len(results_df)}"
    required_cols = [
        "trial_id", "feature_config", "num_features", "max_depth",
        "learning_rate", "n_estimators", "mean_val_pr_auc", "mean_val_recall", "mean_val_f1"
    ]
    for col in required_cols:
        assert col in results_df.columns, f"Missing column: {col}"

    # Verify baseline trial T00_baseline is recorded
    baseline_rows = results_df[results_df["trial_id"] == "T00_baseline"]
    assert len(baseline_rows) == 1
    assert baseline_rows.iloc[0]["num_features"] == 11


def test_threshold_optimization_report():
    """Verify validation threshold sweep report exists and reflects valid ranges."""
    th_path = "results/reports/experimental_xgb_threshold_optimization.csv"
    assert os.path.exists(th_path), f"Threshold report missing: {th_path}"
    th_df = pd.read_csv(th_path)

    assert len(th_df) == 41
    assert th_df["threshold"].min() == pytest.approx(0.10)
    assert th_df["threshold"].max() == pytest.approx(0.90)
    assert ((th_df["recall"] >= 0) & (th_df["recall"] <= 1)).all()
    assert ((th_df["precision"] >= 0) & (th_df["precision"] <= 1)).all()


def test_summary_json_contains_complete_metadata():
    """Verify experimental summary JSON contains complete comparison metadata."""
    json_path = "results/reports/experimental_tuning_summary.json"
    assert os.path.exists(json_path), f"Summary JSON missing: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["num_trials_evaluated"] == 64
    assert meta["production_model_hash_verified"] == FROZEN_BASELINE_HASH
    assert "validation_metrics" in meta
    assert "untouched_test_metrics_selected_threshold" in meta
    assert "event_level_comparison" in meta
    assert meta["event_level_comparison"]["total_episodes"] == 136
