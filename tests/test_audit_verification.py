"""
ShinerAI: Automated Scientific Audit Verification Test Suite
Phase 5 — Scientific Audit Certification

These tests verify:
1. Dataset preconditions (41,277 rows, 37 columns, no NaNs, min DO >= 3.0)
2. Leakage-safe temporal split accounting (32,908 train, 108 purged, 8,261 test)
3. Zero temporal leakage across all 17 ponds (purge gap >= 2.0 hours)
4. Feature configuration integrity and strict column exclusions
5. Deterministic active model metric reproduction (PR-AUC 0.7574, TP 752, FP 698)
6. Completeness of all 16 audit deliverables in results/audit/
7. Scientific domain contribution lift (+0.1425 PR-AUC)
"""

import os
import pytest
import joblib
import pandas as pd
import numpy as np

from src.model_utils import (
    verify_dataset_preconditions,
    create_temporal_split,
    compute_classification_metrics,
    FEATURE_CONFIGS,
    CONFIG_A_FEATURES,
    CONFIG_B_FEATURES,
    CONFIG_C_FEATURES,
    REDUNDANT_CURRENT_COLUMNS,
    NON_PREDICTOR_COLUMNS,
)

DATA_PATH = os.path.join("data", "processed", "ml_ready_dataset.csv")
MODEL_PATH = os.path.join("models", "xgboost_config_c.joblib")
AUDIT_DIR = os.path.join("results", "audit")


@pytest.fixture(scope="module")
def dataset():
    """Loads the processed ML-ready dataset."""
    assert os.path.exists(DATA_PATH), f"Missing dataset: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    return df


@pytest.fixture(scope="module")
def temporal_split(dataset):
    """Executes the temporal split with 2.0h purge gap."""
    train_df, test_df, purged_df, accounting_df = create_temporal_split(
        dataset, train_ratio=0.80, purge_hours=2.0
    )
    return train_df, test_df, purged_df, accounting_df


def test_dataset_preconditions(dataset):
    """Verifies that the dataset satisfies all integrity invariants."""
    res = verify_dataset_preconditions(dataset)
    assert res["status"] == "PASS"
    assert res["rows"] == 41277
    assert res["columns"] == 37
    assert res["pond_count"] == 17
    assert res["safe_count"] == 36101
    assert res["at_risk_count"] == 5176
    assert float(dataset["current_do"].min()) >= 3.0


def test_temporal_split_accounting(dataset, temporal_split):
    """Verifies row conservation and exact partition dimensions."""
    train_df, test_df, purged_df, accounting_df = temporal_split

    assert len(train_df) == 32908
    assert len(purged_df) == 108
    assert len(test_df) == 8261
    assert len(train_df) + len(purged_df) + len(test_df) == len(dataset)
    assert int(test_df["target"].sum()) == 943
    assert int((test_df["target"] == 0).sum()) == 7318


def test_temporal_purge_gap_strictly_enforced(temporal_split):
    """Verifies that train and test intervals have at least a 2.0-hour gap for every pond."""
    _, _, _, accounting_df = temporal_split

    for _, row in accounting_df.iterrows():
        t_purge_start = pd.to_datetime(row["t_purge_start"])
        t_cutoff = pd.to_datetime(row["t_cutoff"])
        delta_hours = (t_cutoff - t_purge_start).total_seconds() / 3600.0
        assert delta_hours >= 2.0, f"Purge gap violation in pond {row['pond_id']}: {delta_hours}h < 2.0h"


def test_feature_configurations_and_exclusions():
    """Verifies feature counts and strict isolation of excluded columns."""
    assert len(CONFIG_A_FEATURES) == 5
    assert len(CONFIG_B_FEATURES) == 29
    assert len(CONFIG_C_FEATURES) == 11

    # Verify Config C is a subset of Config B
    for f in CONFIG_C_FEATURES:
        assert f in CONFIG_B_FEATURES

    # Verify no excluded or future columns appear in predictors
    all_excluded = set(REDUNDANT_CURRENT_COLUMNS + NON_PREDICTOR_COLUMNS)
    for cfg_name, feat_list in FEATURE_CONFIGS.items():
        for feat in feat_list:
            assert feat not in all_excluded, f"Excluded feature {feat} found in {cfg_name}"
            assert not feat.startswith("do_t_plus"), f"Future feature {feat} found in {cfg_name}"
            assert not feat.startswith("ph_t_plus"), f"Future feature {feat} found in {cfg_name}"
            assert not feat.startswith("temp_t_plus"), f"Future feature {feat} found in {cfg_name}"


def test_active_production_model_metrics(temporal_split):
    """Verifies that the serialized production model reproduces documented metrics exactly."""
    assert os.path.exists(MODEL_PATH), f"Missing active model: {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)

    _, test_df, _, _ = temporal_split
    X_test = test_df[CONFIG_C_FEATURES].values
    y_test = test_df["target"].values

    probs = model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, probs, threshold=0.50)

    assert metrics["pr_auc"] == pytest.approx(0.7574, abs=1e-4)
    assert metrics["roc_auc"] == pytest.approx(0.9162, abs=1e-4)
    assert metrics["f1"] == pytest.approx(0.6285, abs=1e-4)
    assert metrics["recall"] == pytest.approx(0.7975, abs=1e-4)
    assert metrics["precision"] == pytest.approx(0.5186, abs=1e-4)
    assert metrics["specificity"] == pytest.approx(0.9046, abs=1e-4)
    assert metrics["accuracy"] == pytest.approx(0.8924, abs=1e-4)

    assert metrics["tp"] == 752
    assert metrics["fp"] == 698
    assert metrics["tn"] == 6620
    assert metrics["fn"] == 191


def test_domain_contribution_pr_auc_lift():
    """Verifies that recent DO history achieves a +0.1425 PR-AUC lift over static DO baseline."""
    master_csv = os.path.join("results", "reports", "MASTER_MODEL_EVALUATION.csv")
    assert os.path.exists(master_csv), f"Missing master evaluation: {master_csv}"
    df = pd.read_csv(master_csv)

    base_pr = float(df.loc[df["model"].str.startswith("Current-DO"), "pr_auc"].values[0])
    xgb_pr = float(
        df.loc[(df["model"] == "XGBoost") & (df["feature_set"].str.contains("Config C")), "pr_auc"].values[0]
    )

    lift = round(xgb_pr - base_pr, 4)
    assert lift == 0.1425, f"Expected lift +0.1425, got {lift}"


def test_audit_deliverables_complete():
    """Verifies that all 16 audit deliverables exist and contain content."""
    required_files = [
        "SCIENTIFIC_AUDIT_PLAN.md",
        "METRIC_RECONCILIATION.csv",
        "ACTIVE_MODEL_REPRODUCTION.md",
        "SPLIT_AUDIT.md",
        "FEATURE_LEAKAGE_AUDIT.md",
        "LABEL_AUDIT.md",
        "DATA_CLEANING_AUDIT.md",
        "MASTER_METRIC_RECALCULATION.csv",
        "DOMAIN_CONTRIBUTION_AUDIT.md",
        "SHAP_AUDIT.md",
        "ERROR_ANALYSIS_AUDIT.md",
        "WALL_TIME_AUDIT.md",
        "NOTEBOOK_REPRODUCIBILITY.md",
        "RESULT_TRACEABILITY.csv",
        "SCIENTIFIC_AUDIT_REPORT.md",
        "MENTOR_EVIDENCE_SUMMARY.md",
    ]

    for fname in required_files:
        fpath = os.path.join(AUDIT_DIR, fname)
        assert os.path.exists(fpath), f"Missing audit deliverable: {fpath}"
        assert os.path.getsize(fpath) > 100, f"Audit deliverable is empty: {fpath}"
