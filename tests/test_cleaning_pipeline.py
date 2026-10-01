"""
test_cleaning_pipeline.py
=========================
Unit and integration tests for Phase 2 data cleaning, segmentation,
and ML dataset generation.
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

from src.data_loader import find_csv_files
from src.cleaning import verify_raw_data


@pytest.fixture(scope="session")
def raw_data_dir() -> Path:
    """Fixture providing raw data directory."""
    return Path("data/raw/csv")


@pytest.fixture(scope="session")
def cleaned_df() -> pd.DataFrame:
    """Fixture providing cleaned pond DataFrame."""
    path = Path("data/processed/cleaned_pond_data.csv")
    assert path.exists(), f"Cleaned pond data not found at {path.resolve()}"
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def ml_ready_df() -> pd.DataFrame:
    """Fixture providing ML-ready dataset."""
    path = Path("data/processed/ml_ready_dataset.csv")
    assert path.exists(), f"ML-ready dataset not found at {path.resolve()}"
    return pd.read_csv(path)


def test_no_raw_files_modified(raw_data_dir):
    """Verify that all 22 raw CSV files remain present and intact in data/raw/csv/."""
    pond_files, meta_files = find_csv_files(raw_data_dir)
    assert len(pond_files) == 17, "Expected 17 pond files."
    assert len(meta_files) >= 5, "Expected at least 5 metadata files."

    # Total row count across raw files must match Phase 1 exactly (72,750 rows)
    total_raw_rows = sum(len(pd.read_csv(f)) for f in pond_files)
    assert total_raw_rows == 72750, f"Raw data row count altered: expected 72,750, got {total_raw_rows}"


def test_raw_verification_function():
    """Verify the verify_raw_data function runs and returns verified status."""
    res = verify_raw_data("data/raw/csv")
    assert res["status"] == "VERIFIED_OK"
    assert res["pond_count"] == 17
    assert res["total_rows"] == 72750


def test_cleaned_dataset_properties(cleaned_df):
    """Verify cleaned dataset contains all rows and required quality status columns."""
    assert len(cleaned_df) == 72750, f"Expected 72,750 rows in cleaned dataset, got {len(cleaned_df)}"
    assert "data_quality_status" in cleaned_df.columns
    assert "segment_id" in cleaned_df.columns
    assert "is_usable" in cleaned_df.columns

    # Verify quality statuses present
    statuses = set(cleaned_df["data_quality_status"].unique())
    assert "usable" in statuses
    assert "excluded_sensor_artifact" in statuses
    assert "excluded_duplicate" in statuses


def test_ml_ready_dataset_columns(ml_ready_df):
    """Verify ML-ready dataset contains all required feature, metadata, and target columns."""
    expected_cols = [
        "pond_id", "prediction_timestamp", "hour_of_day", "minute_of_day",
        "current_do", "current_ph", "current_temperature",
        "do_t", "do_t_minus_15", "do_t_minus_30", "do_t_minus_45", "do_t_minus_60",
        "do_t_minus_75", "do_t_minus_90", "do_t_minus_105", "do_t_minus_120",
        "ph_t", "ph_t_minus_15", "ph_t_minus_30", "ph_t_minus_45", "ph_t_minus_60",
        "ph_t_minus_75", "ph_t_minus_90", "ph_t_minus_105", "ph_t_minus_120",
        "temp_t", "temp_t_minus_15", "temp_t_minus_30", "temp_t_minus_45", "temp_t_minus_60",
        "temp_t_minus_75", "temp_t_minus_90", "temp_t_minus_105", "temp_t_minus_120",
        "target", "target_name", "data_quality_status"
    ]
    for col in expected_cols:
        assert col in ml_ready_df.columns, f"Column '{col}' missing from ml_ready_dataset.csv"


def test_no_exact_zeros_in_ml_ready_features(ml_ready_df):
    """Verify that no equipment artifact zeros (0.0) enter the ML feature columns."""
    sensor_cols = [c for c in ml_ready_df.columns if c.startswith(("do_", "ph_", "temp_", "current_"))]
    for col in sensor_cols:
        assert (ml_ready_df[col] == 0.0).sum() == 0, f"Found exact zero in feature column '{col}'"
        assert (ml_ready_df[col] > 0.0).all(), f"Found non-positive value in feature column '{col}'"


def test_current_do_greater_than_or_equal_to_three(ml_ready_df):
    """Verify that current DO is >= 3.0 mg/L for every single example in the ML dataset."""
    min_curr_do = ml_ready_df["current_do"].min()
    assert min_curr_do >= 3.0, f"Found example with current DO < 3.0 mg/L ({min_curr_do})"
    assert (ml_ready_df["current_do"] >= 3.0).all()


def test_both_classes_exist(ml_ready_df):
    """Verify that both SAFE (0) and AT_RISK (1) classes are represented in the target."""
    classes = set(ml_ready_df["target"].unique())
    assert classes == {0, 1}, f"Target classes must be {{0, 1}}, got {classes}"

    target_names = set(ml_ready_df["target_name"].unique())
    assert target_names == {"SAFE", "AT_RISK"}

    safe_count = (ml_ready_df["target"] == 0).sum()
    at_risk_count = (ml_ready_df["target"] == 1).sum()
    assert safe_count > 10000, f"Insufficient safe examples: {safe_count}"
    assert at_risk_count > 1000, f"Insufficient at-risk examples: {at_risk_count}"


def test_multiple_ponds_represented(ml_ready_df):
    """Verify that all 17 ponds are represented in the final ML-ready dataset."""
    unique_ponds = ml_ready_df["pond_id"].nunique()
    assert unique_ponds == 17, f"Expected 17 ponds, found {unique_ponds}"

    # Every individual pond must have both safe and at-risk examples
    for pid, group in ml_ready_df.groupby("pond_id"):
        assert (group["target"] == 0).sum() > 0, f"Pond {pid} has no SAFE examples."
        assert (group["target"] == 1).sum() > 0, f"Pond {pid} has no AT_RISK examples."


def test_timestamps_chronological_within_windows(ml_ready_df):
    """Verify prediction timestamps are strictly ordered per pond."""
    for pid, group in ml_ready_df.groupby("pond_id"):
        ts_series = pd.to_datetime(group["prediction_timestamp"])
        assert ts_series.is_monotonic_increasing, f"Pond {pid} prediction timestamps are not strictly increasing."
