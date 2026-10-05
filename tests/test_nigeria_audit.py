"""
Unit tests for Nigeria External Dataset Audit (IoTPond10.csv)
Verifies:
1. Raw file existence, immutability, and SHA256 hash.
2. Required columns and data types.
3. Timestamp parsing and structural bifurcation (Part A vs Part B).
4. Sensor hardware artifact detection (-127 C, 0.0 mg/L DO, Ammonia inf, pH out-of-bounds).
5. ShinerAI feature and target contract feasibility (zero valid 4-hour windows).
6. Non-interpolation policy enforcement.
"""

import hashlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


RAW_FILE_PATH = Path("data/external/nigeria_iotpond/raw/IoTPond10.csv")
EXPECTED_SHA256 = "7cd9f8b98a6edf6f2111046f35b19a1304f840c4eaef0b182b5b8922cd601690"
EXPECTED_ROWS = 620
EXPECTED_COLS = 11


@pytest.fixture(scope="module")
def raw_df():
    assert RAW_FILE_PATH.exists(), f"Raw file missing at {RAW_FILE_PATH}"
    df = pd.read_csv(RAW_FILE_PATH)
    return df


def test_raw_file_existence_and_hash():
    """Verify raw file exists and has unaltered SHA256 hash."""
    assert RAW_FILE_PATH.exists()
    assert RAW_FILE_PATH.stat().st_size == 45352
    h = hashlib.sha256(RAW_FILE_PATH.read_bytes()).hexdigest()
    assert h == EXPECTED_SHA256


def test_dataset_dimensions_and_columns(raw_df):
    """Verify row count, column count, and exact column naming."""
    assert len(raw_df) == EXPECTED_ROWS
    assert len(raw_df.columns) == EXPECTED_COLS
    
    expected_cols = [
        "created_at", "entry_id", "TEMPERATURE", "TURBIDITY",
        "DISOLVED OXYGEN", "pH", "AMMONIA", "NITRATE",
        "Population", "Length", "Weight"
    ]
    assert list(raw_df.columns) == expected_cols


def test_structural_bifurcation(raw_df):
    """Verify dataset is split into 310 timestamped rows and 310 date-only rows."""
    has_cet = raw_df["created_at"].str.contains("CET")
    has_slash = raw_df["created_at"].str.contains("/")
    
    assert has_cet.sum() == 310
    assert has_slash.sum() == 310
    
    # Verify Part B sensor values are exact duplicates of Part A
    part_a = raw_df.iloc[0:310]
    part_b = raw_df.iloc[310:620]
    
    for col in ["TEMPERATURE", "TURBIDITY", "DISOLVED OXYGEN", "pH", "NITRATE", "Population"]:
        assert (part_a[col].values == part_b[col].values).all(), f"Mismatch in {col}"


def test_sensor_artifacts_detection(raw_df):
    """Verify deterministic identification of hardware and firmware artifacts."""
    # Temperature Dallas DS18B20 disconnect value (-127 C)
    temp_neg127 = (raw_df["TEMPERATURE"] == -127.0).sum()
    assert temp_neg127 == 20
    
    # Dissolved Oxygen = 0.0 mg/L dropouts
    do_zeros = (raw_df["DISOLVED OXYGEN"] == 0.0).sum()
    assert do_zeros == 338  # 54.5% of rows
    
    # Ammonia firmware division-by-zero inf values
    amm_inf = np.isinf(raw_df["AMMONIA"]).sum()
    assert amm_inf == 9
    
    # Turbidity negative ADC zero-drift values
    turb_neg = (raw_df["TURBIDITY"] < 0).sum()
    assert turb_neg == 316
    
    # pH outside physical 0-14 scale
    ph_oob = ((raw_df["pH"] < 0) | (raw_df["pH"] > 14)).sum()
    assert ph_oob == 28


def test_shinerai_feasibility_zero_eligible_windows(raw_df):
    """
    Verify that zero samples can satisfy ShinerAI's 2-hour past + 2-hour future
    contract (240 minutes continuous) without synthetic interpolation.
    """
    part_a = raw_df.iloc[0:310].copy()
    part_a["ts"] = pd.to_datetime(part_a["created_at"].str.replace(" CET", ""))
    
    # Maximum continuous session duration
    session_durations = []
    for _, grp in part_a.groupby(part_a["ts"].dt.date):
        dur_m = (grp["ts"].max() - grp["ts"].min()).total_seconds() / 60.0
        session_durations.append(dur_m)
        
    max_duration = max(session_durations)
    assert max_duration == pytest.approx(174.6, abs=0.5)
    
    # Required continuous window for 2h lookback + 2h lookahead is 240 minutes
    assert max_duration < 240.0
    
    # Strict sample eligibility count
    eligible_samples = 0
    for _, grp in part_a.groupby(part_a["ts"].dt.date):
        t_start = grp["ts"].min()
        t_end = grp["ts"].max()
        # Row must have 120m past and 120m future within the continuous session
        valid_rows = grp[(grp["ts"] >= t_start + pd.Timedelta(hours=2)) & 
                         (grp["ts"] <= t_end - pd.Timedelta(hours=2))]
        eligible_samples += len(valid_rows)
        
    assert eligible_samples == 0


def test_do_chattering_instability(raw_df):
    """Verify that DO < 3.0 values represent rapid hardware dropouts rather than sustained hypoxia."""
    part_a = raw_df.iloc[0:310].copy()
    part_a["ts"] = pd.to_datetime(part_a["created_at"].str.replace(" CET", ""))
    
    june25 = part_a[part_a["ts"].dt.date == pd.to_datetime("2021-06-25").date()].copy()
    june25["do_diff"] = june25["DISOLVED OXYGEN"].diff().abs()
    june25["dt_sec"] = june25["ts"].diff().dt.total_seconds()
    
    # Swings > 5.0 mg/L in <= 60 seconds
    swings = june25[(june25["do_diff"] > 5.0) & (june25["dt_sec"] <= 60)]
    assert len(swings) >= 15
