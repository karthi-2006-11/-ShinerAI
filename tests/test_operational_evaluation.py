"""
Unit tests for src/operational_evaluation.py
Verifies:
1. Episode extraction on synthetic continuous time-series.
2. Event-level detection rate and lead time bounds.
3. False alarm burden calculations (daily rate per pond).
4. Alert stability and chattering metrics.
5. Hysteresis filter performance on operational holdout.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.model_utils import create_temporal_split, CONFIG_C_FEATURES
from src.operational_evaluation import (
    extract_hypoxia_episodes,
    compute_event_level_metrics,
    compute_false_alarms_per_pond_day,
    compute_alert_stability_metrics,
    evaluate_hysteresis_filter,
)


@pytest.fixture(scope="module")
def real_eval_df():
    data_path = Path("data/processed/ml_ready_dataset.csv")
    model_path = Path("models/xgboost_config_c.joblib")
    assert data_path.exists() and model_path.exists()

    df = pd.read_csv(data_path)
    _, test_df, _, _ = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)
    model = joblib.load(model_path)

    X_test = test_df[CONFIG_C_FEATURES].values
    probs = model.predict_proba(X_test)[:, 1]

    eval_df = test_df.copy()
    eval_df["prob"] = probs
    eval_df["pred"] = (probs >= 0.50).astype(int)
    return eval_df


def test_synthetic_episode_extraction():
    """Verify episode identification and lead time on controlled synthetic data."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=8, freq="15min")
    # Ponds: 4 safe rows, then 4 at-risk rows
    df_syn = pd.DataFrame({
        "pond_id": ["pond_1"] * 8,
        "prediction_timestamp": timestamps,
        "current_do": [5.0, 4.8, 4.5, 4.2, 3.8, 3.5, 3.2, 3.1],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
        "pred": [0, 0, 0, 0, 1, 1, 1, 1],
        "prob": [0.1, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.9],
    })

    episodes = extract_hypoxia_episodes(df_syn)
    assert len(episodes) == 1
    ep = episodes.iloc[0]
    assert ep["pond_id"] == "pond_1"
    assert ep["detected"] == 1
    assert ep["interval_count"] == 4
    # Actual crossing time is end_time (01:45) + 15 min = 02:00
    # First alert is at start_time (01:00)
    # Lead time = 60 minutes
    assert ep["lead_time_minutes"] == 60.0


def test_real_event_level_metrics(real_eval_df):
    """Verify event-level metrics on real test set."""
    episodes = extract_hypoxia_episodes(real_eval_df)
    metrics = compute_event_level_metrics(episodes)

    assert metrics["total_hypoxia_episodes"] == 136
    assert metrics["detected_episodes"] == 124
    assert metrics["missed_episodes"] == 12
    assert metrics["event_detection_rate"] == pytest.approx(0.9118, abs=1e-3)
    assert metrics["event_miss_rate"] == pytest.approx(0.0882, abs=1e-3)
    assert metrics["mean_lead_time_minutes"] == pytest.approx(101.7, abs=1.0)
    assert metrics["median_lead_time_minutes"] == pytest.approx(120.0, abs=1.0)


def test_false_alarms_per_pond_day(real_eval_df):
    """Verify false alarm rate on real test set."""
    res = compute_false_alarms_per_pond_day(real_eval_df)
    assert res["total_test_ponds"] == 17
    assert res["total_false_positive_intervals"] == 698
    assert res["mean_false_alarms_per_pond_day"] == pytest.approx(4.84, abs=0.1)


def test_alert_stability_metrics(real_eval_df):
    """Verify alert stability and chattering metrics."""
    res = compute_alert_stability_metrics(real_eval_df)
    assert res["total_alert_activations"] > 0
    assert 0.0 <= res["chattering_rate"] <= 1.0
    assert res["mean_alert_duration_minutes"] > 15.0


def test_hysteresis_filter(real_eval_df):
    """Verify that a 2-step hysteresis filter reduces false alarms while preserving sensitivity."""
    res = evaluate_hysteresis_filter(real_eval_df, consecutive_steps_required=2)
    assert res["fp"] < 698
    assert res["fp_reduction"] > 0
    assert res["fp_reduction_percent"] > 20.0
    assert res["specificity"] > 0.9046
    assert res["recall"] > 0.70
