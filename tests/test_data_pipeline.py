"""
test_data_pipeline.py
=====================
Pytest suite for Phase 1 data pipeline:
- File discovery
- Loading and timestamp parsing
- Schema validation and required columns
- Pond ID assignment
- Unified combination
- Numeric type conversion
- Quality audit & feasibility calculations
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pytest

from src.data_loader import (
    find_csv_files,
    extract_pond_id,
    load_pond_csv,
    load_all_ponds,
    combine_pond_data,
)
from src.data_quality import (
    audit_single_pond,
    audit_combined_dataset,
    analyze_qc_flags,
    check_do_feasibility,
)


@pytest.fixture(scope="session")
def raw_data_dir() -> Path:
    """Fixture providing path to raw CSV directory."""
    path = Path("data/raw/csv")
    assert path.exists(), f"Raw data directory does not exist at {path.resolve()}"
    return path


@pytest.fixture(scope="session")
def discovered_files(raw_data_dir: Path):
    """Fixture returning discovered pond and metadata files."""
    pond_files, meta_files = find_csv_files(raw_data_dir)
    return pond_files, meta_files


@pytest.fixture(scope="session")
def loaded_ponds(raw_data_dir: Path):
    """Fixture returning loaded list of pond DataFrames."""
    return load_all_ponds(raw_data_dir)


@pytest.fixture(scope="session")
def combined_df(loaded_ponds):
    """Fixture returning combined DataFrame."""
    return combine_pond_data(loaded_ponds)


def test_pond_csv_discovery(discovered_files):
    """Verify that all expected 17 pond files and metadata files are discovered."""
    pond_files, meta_files = discovered_files
    assert len(pond_files) == 17, f"Expected 17 pond files, found {len(pond_files)}"
    assert len(meta_files) >= 5, f"Expected at least 5 metadata files, found {len(meta_files)}"
    for f in pond_files:
        assert f.name.startswith("ara2_"), f"Unexpected pond file naming: {f.name}"


def test_extract_pond_id():
    """Verify pond_id extraction from file paths."""
    sample_path = Path("data/raw/csv/ara2_0677080b.csv")
    assert extract_pond_id(sample_path) == "ara2_0677080b"


def test_load_single_pond_csv(discovered_files):
    """Verify loading a single pond CSV file correctly preserves data and metadata."""
    pond_files, _ = discovered_files
    sample_file = pond_files[0]
    df = load_pond_csv(sample_file)

    assert not df.empty, "Loaded DataFrame should not be empty."
    assert "pond_id" in df.columns, "'pond_id' must be present in loaded DataFrame."
    assert df["pond_id"].iloc[0] == sample_file.stem
    assert "source_file" in df.columns
    assert df["source_file"].iloc[0] == sample_file.name


def test_timestamp_parsed_correctly(discovered_files):
    """Verify that Timestamp column is parsed to datetime64 and has no NaT values."""
    pond_files, _ = discovered_files
    for file_path in pond_files[:3]:  # Check sample of ponds
        df = load_pond_csv(file_path)
        assert "timestamp" in df.columns, "Standard 'timestamp' column missing."
        assert "Timestamp" in df.columns, "Aliased 'Timestamp' column missing."
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp must be datetime type."
        assert df["timestamp"].isna().sum() == 0, f"Found NaT timestamps in {file_path.name}"


def test_required_columns_exist(discovered_files):
    """Verify all required sensor and metadata columns exist in loaded data."""
    pond_files, _ = discovered_files
    df = load_pond_csv(pond_files[0])

    # Core required columns
    required_cols = [
        "timestamp",
        "Timestamp",
        "pond_id",
        "source_file",
        "do_mg_l",
        "ph",
        "temperature_c",
        "qc_flag_datetime",
        "qc_flag_do",
        "qc_flag_ph",
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' is missing from loaded DataFrame."


def test_combined_dataframe_properties(combined_df):
    """Verify that combined DataFrame is non-empty, contains all 17 ponds, and has expected rows."""
    assert not combined_df.empty, "Combined DataFrame must not be empty."
    assert combined_df["pond_id"].nunique() == 17, "Combined DataFrame must contain exactly 17 ponds."
    assert len(combined_df) == 72750, f"Expected 72,750 rows, got {len(combined_df)}"


def test_basic_numeric_columns_convertible(combined_df):
    """Verify numeric columns (DO, pH, Temperature) are valid numeric float types."""
    for col in ["do_mg_l", "ph", "temperature_c"]:
        assert pd.api.types.is_numeric_dtype(combined_df[col]), f"Column '{col}' is not numeric dtype."
        # Confirm numeric operations (min, max, mean) compute without exceptions
        col_min = combined_df[col].min()
        col_max = combined_df[col].max()
        assert not np.isnan(col_min)
        assert not np.isnan(col_max)
        assert col_max >= col_min


def test_audit_single_pond_metrics(discovered_files):
    """Verify single pond audit dictionary structure and calculation correctness."""
    pond_files, _ = discovered_files
    df = load_pond_csv(pond_files[0])
    metrics = audit_single_pond(df)

    expected_keys = [
        "filename", "pond_id", "rows", "earliest_timestamp", "latest_timestamp",
        "duration_days", "unique_timestamps", "duplicate_timestamps",
        "expected_interval_min", "median_interval", "pct_close_15min", "gaps_gt_20min",
        "max_gap", "missing_do", "missing_ph", "missing_temp",
        "zero_do", "zero_ph", "zero_temp",
        "min_do", "max_do", "mean_do", "median_do",
        "min_ph", "max_ph", "mean_ph", "median_ph",
        "min_temp", "max_temp", "mean_temp", "median_temp"
    ]
    for key in expected_keys:
        assert key in metrics, f"Key '{key}' missing from audit_single_pond output."


def test_check_do_feasibility(discovered_files):
    """Verify DO < 3 mg/L feasibility checks exclude 0 mg/L artifacts and produce valid counts."""
    pond_files, _ = discovered_files
    df = load_pond_csv(pond_files[0])
    feasibility = check_do_feasibility(df, threshold=3.0)

    assert "valid_below_3_count" in feasibility
    assert "pct_below_3" in feasibility
    assert "num_episodes" in feasibility
    assert "longest_period_str" in feasibility
    assert feasibility["valid_below_3_count"] >= 0
    assert 0.0 <= feasibility["pct_below_3"] <= 100.0


def test_qc_flags_parsing(combined_df):
    """Verify that analyze_qc_flags parses QC flags without error."""
    qc_stats = analyze_qc_flags(combined_df)
    assert "flag_counts" in qc_stats
    assert "multi_flag_rows" in qc_stats
    assert "total_flagged_rows" in qc_stats
    assert qc_stats["total_flagged_rows"] > 0
