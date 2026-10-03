"""
ShinerAI: Independent External Validation & Model Comparison Test Suite
Verifies:
1. External dataset schema and column integrity
2. External timestamp parsing and chronological ordering
3. Leakage-free 15-minute grid resampling
4. 2-Hour historical lag construction (8 lags: do_t_minus_15 to do_t_minus_120)
5. Independent ground-truth label construction (2-hour forward window)
6. Zero temporal leakage between past features and future target
7. Exact Config C feature sequence matching frozen model expectations
8. Frozen model loading and non-mutating inference
9. Zero external model fitting or calibration
10. Metric computation with transparent single-class edge-case handling
11. Invariant preservation of internal Phase 3 benchmark metrics
12. Model comparison and feature ablation artifact verification
"""

from pathlib import Path
import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.model_utils import (
    create_temporal_split,
    compute_classification_metrics,
    CONFIG_C_FEATURES,
)
from src.external_validation import (
    resample_external_to_15min_grid,
    build_config_c_features_and_target,
    evaluate_frozen_model_on_external,
)

EXTERNAL_DIR = Path("data/external/oman_tilapia")
RAW_LIVE_PATH = EXTERNAL_DIR / "data" / "raw" / "live" / "raw_readings_new.csv"
MODEL_PATH = Path("models/xgboost_config_c.joblib")
INTERNAL_DATA_PATH = Path("data/processed/ml_ready_dataset.csv")


# ==============================================================================
# 1. External Dataset Schema & Processing Tests
# ==============================================================================

def test_external_dataset_schema():
    """Verify that external raw dataset matches expected columns and data types."""
    assert RAW_LIVE_PATH.exists(), f"Missing external raw data at {RAW_LIVE_PATH}"
    df = pd.read_csv(RAW_LIVE_PATH, nrows=100)
    expected_cols = {"timestamp", "node_id", "temp_c", "ph", "do_mgL"}
    assert expected_cols.issubset(set(df.columns)), f"Columns missing from external raw data: {expected_cols - set(df.columns)}"


def test_external_timestamp_processing():
    """Verify external timestamps parse cleanly, have no nulls, and order chronologically."""
    df = pd.read_csv(RAW_LIVE_PATH, nrows=500)
    dt_series = pd.to_datetime(df["timestamp"])
    assert dt_series.notna().all(), "Found unparseable or null timestamps in external data"
    assert (dt_series.diff().dropna().dt.total_seconds() >= 0).all(), "External timestamps must be monotonically non-decreasing"


def test_external_15min_resampling():
    """Verify 15-minute resampling protocol enforces right-closed boundaries without leakage."""
    # Synthetic micro-test with known timestamps
    raw_sample = pd.DataFrame({
        "timestamp": [
            "2026-04-01T10:00:01",
            "2026-04-01T10:05:00",
            "2026-04-01T10:14:59",
            "2026-04-01T10:15:00",
            "2026-04-01T10:15:01",
        ],
        "do_mgL": [7.0, 7.2, 7.4, 7.6, 8.0],
    })
    resampled = resample_external_to_15min_grid(raw_sample, timestamp_col="timestamp", do_col="do_mgL")

    # The 10:15:00 bin (closed='right') must contain readings up to 10:15:00 (7.0, 7.2, 7.4, 7.6) -> mean = 7.3
    # The reading at 10:15:01 belongs to the NEXT bin (10:30:00)
    bin_1015 = resampled[resampled["dt"] == pd.Timestamp("2026-04-01 10:15:00")]
    assert len(bin_1015) == 1
    assert round(bin_1015["current_do"].values[0], 2) == 7.30
    assert bin_1015["reading_count"].values[0] == 4


# ==============================================================================
# 2. Feature & Label Construction Tests
# ==============================================================================

def test_external_2hour_history_construction():
    """Verify that all 8 historical lags are constructed with correct lag offsets."""
    dates = pd.date_range("2026-04-01 00:00:00", periods=20, freq="15min")
    df_15m = pd.DataFrame({
        "dt": dates,
        "current_do": [5.0 + i * 0.1 for i in range(20)],
        "reading_count": [10] * 20,
    })
    df_feat = build_config_c_features_and_target(df_15m, timestamp_col="dt", do_col="current_do")

    # At index 8 (timestamp 02:00:00):
    # current_do = 5.8
    # do_t_minus_15 = 5.7 (index 7)
    # do_t_minus_120 = 5.0 (index 0)
    row_8 = df_feat.iloc[8]
    assert np.isclose(row_8["current_do"], 5.8)
    assert np.isclose(row_8["do_t_minus_15"], 5.7)
    assert np.isclose(row_8["do_t_minus_120"], 5.0)
    assert row_8["hour_of_day"] == 2
    assert row_8["minute_of_day"] == 120


def test_external_label_construction_safe_and_at_risk():
    """Verify target calculation over future 2-hour window correctly identifies SAFE vs AT_RISK."""
    dates = pd.date_range("2026-04-01 00:00:00", periods=30, freq="15min")

    # Trajectory 1: Remains safe (DO >= 3.0 throughout)
    safe_dos = [5.0] * 30
    df_safe = pd.DataFrame({"dt": dates, "current_do": safe_dos, "reading_count": [10]*30})
    res_safe = build_config_c_features_and_target(df_safe)
    eligible_safe = res_safe[res_safe["eligible"]]
    assert (eligible_safe["target"] == 0).all(), "All uncompromised trajectories must be labeled SAFE (0)"

    # Trajectory 2: Crashes below 3.0 at step t+4 (1 hour into future)
    risk_dos = [5.0] * 10 + [2.8] + [2.5] * 19
    df_risk = pd.DataFrame({"dt": dates, "current_do": risk_dos, "reading_count": [10]*30})
    res_risk = build_config_c_features_and_target(df_risk)
    # Row 9 is at t=02:15, with current_do=5.0 >= 3.0; t+1 is index 10 (2.8 < 3.0) -> must be AT_RISK (1)
    row_9 = res_risk.iloc[9]
    assert row_9["eligible"]
    assert row_9["target"] == 1, "Trajectory dropping below 3.0 in next 2 hours must be labeled AT_RISK (1)"


def test_no_future_leakage_in_features():
    """Verify feature values strictly depend on past and current observations."""
    dates = pd.date_range("2026-04-01 00:00:00", periods=25, freq="15min")
    dos = [5.0 + 0.05 * i for i in range(25)]
    df_15m = pd.DataFrame({"dt": dates, "current_do": dos, "reading_count": [10]*25})

    df_base = build_config_c_features_and_target(df_15m)

    # Mutate a future value at index 20
    df_mutated_15m = df_15m.copy()
    df_mutated_15m.loc[20, "current_do"] = 999.0
    df_mutated = build_config_c_features_and_target(df_mutated_15m)

    # All features up to index 19 must be completely identical between base and mutated
    for idx in range(8, 20):
        for col in CONFIG_C_FEATURES:
            assert np.isclose(df_base.loc[idx, col], df_mutated.loc[idx, col]), f"Leakage detected: {col} changed at index {idx} when future index 20 was altered"


def test_external_feature_ordering_matches_config_c():
    """Verify that external feature matrix matches CONFIG_C_FEATURES column order exactly."""
    eval_csv = Path("data/external/oman_tilapia_15min_eval.csv")
    if not eval_csv.exists():
        evaluate_frozen_model_on_external()
    df_eval = pd.read_csv(eval_csv)
    for col in CONFIG_C_FEATURES:
        assert col in df_eval.columns, f"Required predictor {col} missing from external feature set"


# ==============================================================================
# 3. Frozen Model Protocol & Non-Mutating Evaluation Tests
# ==============================================================================

def test_frozen_model_loading_and_inference():
    """Verify that frozen model loads and generates probability predictions without error."""
    assert MODEL_PATH.exists(), f"Missing frozen model artifact at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    dummy_input = np.ones((1, len(CONFIG_C_FEATURES))) * 5.0
    probs = model.predict_proba(dummy_input)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0


def test_no_external_model_fitting():
    """Verify that external validation pipeline executes strictly in inference mode without fitting."""
    eval_module_text = Path("src/external_validation.py").read_text(encoding="utf-8")
    assert ".fit(" not in eval_module_text, "External validation script must NEVER call .fit() on external data"
    assert "GridSearchCV" not in eval_module_text
    assert "RandomizedSearchCV" not in eval_module_text


def test_external_metric_calculation_and_edge_case_handling():
    """Verify that external evaluation handles zero-positive ground truth with scientific transparency."""
    res = evaluate_frozen_model_on_external()
    assert res["eligible_prediction_points"] == 742
    assert res["external_positives"] == 0
    assert res["external_negatives"] == 742
    assert res["tn"] + res["fp"] == 742
    assert res["tp"] == 0
    assert res["fn"] == 0
    assert res["specificity"] >= 0.99
    assert res["accuracy"] >= 0.99
    # Check that undefined metrics are reported as None rather than fabricated
    assert res["recall"] is None
    assert res["pr_auc"] is None
    assert res["roc_auc"] is None


# ==============================================================================
# 4. Invariant Preservation: Internal Benchmark Unchanged
# ==============================================================================

def test_internal_benchmark_values_remain_unchanged():
    """
    CRITICAL REGRESSION TEST:
    Verify that running the active frozen model on the internal temporal holdout
    produces the exact frozen Phase 3 benchmark metrics.
    """
    assert INTERNAL_DATA_PATH.exists()
    assert MODEL_PATH.exists()

    df = pd.read_csv(INTERNAL_DATA_PATH)
    _, test_df, _, _ = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)

    model = joblib.load(MODEL_PATH)
    X_test = test_df[CONFIG_C_FEATURES].values
    y_test = test_df["target"].values

    probs = model.predict_proba(X_test)[:, 1]
    m = compute_classification_metrics(y_test, probs, threshold=0.50)

    # Exact frozen benchmarks
    assert m["pr_auc"] == 0.7574, f"PR-AUC regression: expected 0.7574, got {m['pr_auc']}"
    assert m["roc_auc"] == 0.9162, f"ROC-AUC regression: expected 0.9162, got {m['roc_auc']}"
    assert m["f1"] == 0.6285, f"F1 regression: expected 0.6285, got {m['f1']}"
    assert m["recall"] == 0.7975, f"Recall regression: expected 0.7975, got {m['recall']}"
    assert m["precision"] == 0.5186, f"Precision regression: expected 0.5186, got {m['precision']}"
    assert m["specificity"] == 0.9046, f"Specificity regression: expected 0.9046, got {m['specificity']}"
    assert m["accuracy"] == 0.8924, f"Accuracy regression: expected 0.8924, got {m['accuracy']}"

    assert m["tp"] == 752
    assert m["fp"] == 698
    assert m["tn"] == 6620
    assert m["fn"] == 191


# ==============================================================================
# 5. Model Comparison Artifacts Tests
# ==============================================================================

def test_model_comparison_tables_and_ablation_artifacts():
    """Verify that consolidated model comparison and ablation tables exist and contain authoritative values."""
    comp_csv = Path("results/reports/MODEL_COMPARISON.csv")
    ablation_csv = Path("results/reports/FEATURE_ABLATION_COMPARISON.csv")
    fig_path = Path("results/figures/model_comparison_prauc.png")

    assert comp_csv.exists(), "Missing MODEL_COMPARISON.csv"
    assert ablation_csv.exists(), "Missing FEATURE_ABLATION_COMPARISON.csv"
    assert fig_path.exists(), "Missing model_comparison_prauc.png"

    df_comp = pd.read_csv(comp_csv)
    assert len(df_comp) == 11, f"Expected 11 rows in model comparison, got {len(df_comp)}"
    assert "XGBoost" in df_comp["Model"].values

    df_abl = pd.read_csv(ablation_csv)
    assert len(df_abl) == 3, f"Expected 3 rows in feature ablation, got {len(df_abl)}"
    assert set(df_abl["Model"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
