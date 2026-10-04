"""
Unit tests for src/baselines.py
Verifies:
1. Strict persistence baseline logic and metrics.
2. Linear trend slope estimation on synthetic sequences (flat, rising, dropping).
3. Trend-based 120-minute forward extrapolation.
4. Exact metric reproducibility on held-out test split.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.model_utils import create_temporal_split
from src.baselines import (
    compute_linear_trend_slopes,
    evaluate_persistence_baseline,
    evaluate_trend_baseline,
    TREND_WINDOW_COLS,
    TREND_TIME_POINTS,
)


@pytest.fixture(scope="module")
def real_test_df():
    data_path = Path("data/processed/ml_ready_dataset.csv")
    assert data_path.exists(), "Dataset not found"
    df = pd.read_csv(data_path)
    _, test_df, _, _ = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)
    return test_df


def test_linear_trend_slope_synthetic():
    """Verify OLS slope calculation on known synthetic slopes."""
    # Flat DO: 5.0 at all lags
    flat_data = {col: [5.0] for col in TREND_WINDOW_COLS}
    df_flat = pd.DataFrame(flat_data)
    slopes_flat = compute_linear_trend_slopes(df_flat)
    assert pytest.approx(slopes_flat[0], abs=1e-6) == 0.0

    # Declining DO: drops 1.2 mg/L over 120 min (-0.01 mg/L per minute)
    # At t=-120: 6.2; at t=0: 5.0
    decline_data = {}
    for col, t in zip(TREND_WINDOW_COLS, TREND_TIME_POINTS):
        decline_data[col] = [5.0 + (-0.01 * t)]
    df_decline = pd.DataFrame(decline_data)
    slopes_decline = compute_linear_trend_slopes(df_decline)
    assert pytest.approx(slopes_decline[0], abs=1e-4) == -0.01

    # Rising DO: rises 1.2 mg/L over 120 min (+0.01 mg/L per minute)
    rising_data = {}
    for col, t in zip(TREND_WINDOW_COLS, TREND_TIME_POINTS):
        rising_data[col] = [5.0 + (0.01 * t)]
    df_rising = pd.DataFrame(rising_data)
    slopes_rising = compute_linear_trend_slopes(df_rising)
    assert pytest.approx(slopes_rising[0], abs=1e-4) == 0.01


def test_persistence_baseline_logic():
    """Verify strict persistence behavior: always predicts 0 (SAFE) under healthy invariant."""
    sample_df = pd.DataFrame({
        "current_do": [3.5, 4.0, 5.0, 3.2],
        "target": [0, 0, 1, 1],
    })
    res = evaluate_persistence_baseline(sample_df)
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 2
    assert res["tn"] == 2
    assert res["recall"] == 0.0
    assert res["specificity"] == 1.0
    assert res["accuracy"] == 0.5


def test_persistence_baseline_real_data(real_test_df):
    """Verify strict persistence metrics on the authoritative test set."""
    res = evaluate_persistence_baseline(real_test_df)
    assert res["test_examples"] == 8261
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["tn"] == 7318
    assert res["fn"] == 943
    assert res["recall"] == 0.0000
    assert res["specificity"] == 1.0000
    assert res["accuracy"] == pytest.approx(0.8858, abs=1e-4)
    assert res["pr_auc"] == pytest.approx(0.1142, abs=1e-4)


def test_trend_baseline_real_data(real_test_df):
    """Verify 120-minute linear trend baseline metrics on the authoritative test set."""
    res = evaluate_trend_baseline(real_test_df)
    assert res["test_examples"] == 8261
    assert res["pr_auc"] == pytest.approx(0.4656, abs=1e-3)
    assert res["roc_auc"] == pytest.approx(0.8870, abs=1e-3)
    assert res["f1"] == pytest.approx(0.5834, abs=1e-3)
    assert res["recall"] == pytest.approx(0.6267, abs=1e-3)
    assert res["precision"] == pytest.approx(0.5457, abs=1e-3)
    assert res["specificity"] == pytest.approx(0.9328, abs=1e-3)
    assert res["accuracy"] == pytest.approx(0.8978, abs=1e-3)
