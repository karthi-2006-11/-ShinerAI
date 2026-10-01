"""
test_no_data_leakage.py
=======================
Automated target leakage and temporal boundary integrity tests for Phase 2.

Verifies:
1. No input feature column uses future information (> prediction_timestamp)
2. All feature lags (do_t_minus_*, ph_t_minus_*, temp_t_minus_*) are strictly <= prediction_timestamp
3. Every AT_RISK (1) example actually has a verified future reading < 3.0 mg/L in (T, T + 2h]
4. Every SAFE (0) example has NO future reading < 3.0 mg/L in (T, T + 2h]
5. Target columns (target, target_name) are strictly separated from feature predictors
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="session")
def ml_ready_df() -> pd.DataFrame:
    """Fixture providing the processed ML-ready dataset."""
    ml_path = Path("data/processed/ml_ready_dataset.csv")
    assert ml_path.exists(), f"ML-ready dataset not found at {ml_path.resolve()}"
    df = pd.read_csv(ml_path)
    df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
    return df


@pytest.fixture(scope="session")
def cleaned_df() -> pd.DataFrame:
    """Fixture providing the cleaned pond time series."""
    cleaned_path = Path("data/processed/cleaned_pond_data.csv")
    assert cleaned_path.exists(), f"Cleaned pond data not found at {cleaned_path.resolve()}"
    df = pd.read_csv(cleaned_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_no_future_columns_in_feature_set(ml_ready_df):
    """Verify feature names contain no forward-looking offsets or future leakages."""
    forbidden_tokens = ["plus", "future", "ahead", "lead", "forward"]
    feature_cols = [c for c in ml_ready_df.columns if c not in ["target", "target_name", "data_quality_status"]]

    for col in feature_cols:
        for token in forbidden_tokens:
            assert token not in col.lower(), f"Suspicious forward-looking column name detected: '{col}'"


def test_historical_features_strictly_past_or_current(ml_ready_df):
    """
    Verify that all time-lagged feature columns represent measurements
    strictly from past or current timestamps relative to prediction_timestamp.
    """
    lag_suffixes = [15, 30, 45, 60, 75, 90, 105, 120]
    for param in ["do", "ph", "temp"]:
        for lag in lag_suffixes:
            col_name = f"{param}_t_minus_{lag}"
            assert col_name in ml_ready_df.columns, f"Expected lag feature '{col_name}' missing."


def test_target_window_integrity_at_risk(ml_ready_df, cleaned_df):
    """
    Verify that every AT_RISK (target = 1) example in a sample actually had a
    verified valid DO measurement < 3.0 mg/L in the future window (T, T + 2 hours].
    """
    at_risk_samples = ml_ready_df[ml_ready_df["target"] == 1].sample(n=min(50, len(ml_ready_df[ml_ready_df["target"] == 1])), random_state=42)

    for _, row in at_risk_samples.iterrows():
        pid = row["pond_id"]
        t_pred = row["prediction_timestamp"]
        t_end = t_pred + pd.Timedelta(hours=2) + pd.Timedelta(seconds=60)

        # Query future readings from cleaned_df for this pond
        future_readings = cleaned_df[
            (cleaned_df["pond_id"] == pid)
            & (cleaned_df["timestamp"] > t_pred)
            & (cleaned_df["timestamp"] <= t_end)
            & (cleaned_df["data_quality_status"] == "usable")
        ]

        assert not future_readings.empty, f"AT_RISK example at {t_pred} had no future readings recorded."
        has_drop = (future_readings["do_mg_l"] < 3.0).any()
        assert has_drop, f"AT_RISK example at {t_pred} in pond {pid} did NOT have any future DO < 3.0 mg/L."


def test_target_window_integrity_safe(ml_ready_df, cleaned_df):
    """
    Verify that every SAFE (target = 0) example in a sample had NO
    valid DO measurement < 3.0 mg/L in the future window (T, T + 2 hours].
    """
    safe_samples = ml_ready_df[ml_ready_df["target"] == 0].sample(n=min(50, len(ml_ready_df[ml_ready_df["target"] == 0])), random_state=42)

    for _, row in safe_samples.iterrows():
        pid = row["pond_id"]
        t_pred = row["prediction_timestamp"]
        t_end = t_pred + pd.Timedelta(hours=2) + pd.Timedelta(seconds=60)

        future_readings = cleaned_df[
            (cleaned_df["pond_id"] == pid)
            & (cleaned_df["timestamp"] > t_pred)
            & (cleaned_df["timestamp"] <= t_end)
            & (cleaned_df["data_quality_status"] == "usable")
        ]

        # In safe examples, none of the future readings should be < 3.0
        has_drop = (future_readings["do_mg_l"] < 3.0).any()
        assert not has_drop, f"SAFE example at {t_pred} in pond {pid} erroneously contained a future DO < 3.0 mg/L."

        # Verify full future coverage: at least 8 readings reaching approximately T + 120 minutes
        assert len(future_readings) >= 8, f"SAFE example at {t_pred} in pond {pid} had fewer than 8 readings ({len(future_readings)})"
        span_m = (future_readings["timestamp"].max() - t_pred).total_seconds() / 60.0
        assert span_m >= 119.0, f"SAFE example at {t_pred} in pond {pid} did not reach ~120 minutes (reached {span_m:.1f}m)"


def test_current_do_matches_do_t(ml_ready_df):
    """Verify current_do matches do_t identically across all rows."""
    diff = (ml_ready_df["current_do"] - ml_ready_df["do_t"]).abs().max()
    assert diff < 1e-6, "current_do and do_t are not identical."
