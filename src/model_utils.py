"""
ShinerAI: Model Utilities & Machine Learning Pipeline Helpers
Phase 3 — Machine Learning Training + Evaluation

This module provides reusable, reproducible utilities for:
1. Dataset precondition verification (41,277 rows, 37 columns, no NaNs, valid targets)
2. Explicit feature configuration definitions (Configs A, B, and C)
3. Leakage-safe temporal train/test splitting with a 2-hour purge gap
4. Unseen-pond GroupKFold splitting
5. Comprehensive classification metrics computation (PR-AUC, ROC-AUC, Recall, Precision, F1, Specificity)
6. Scikit-learn and XGBoost model factory
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==============================================================================
# 1. FEATURE CONFIGURATIONS
# ==============================================================================

# Time-of-day context
TIME_FEATURES: List[str] = ["hour_of_day", "minute_of_day"]

# Current sensor values at prediction time T
CURRENT_FEATURES: List[str] = ["current_do", "current_ph", "current_temperature"]

# Historical lag features (t-15m through t-120m, 8 lags each)
DO_LAGS: List[str] = [f"do_t_minus_{m}" for m in [15, 30, 45, 60, 75, 90, 105, 120]]
PH_LAGS: List[str] = [f"ph_t_minus_{m}" for m in [15, 30, 45, 60, 75, 90, 105, 120]]
TEMP_LAGS: List[str] = [f"temp_t_minus_{m}" for m in [15, 30, 45, 60, 75, 90, 105, 120]]

# Configuration A: CURRENT ONLY (5 features)
CONFIG_A_FEATURES: List[str] = CURRENT_FEATURES + TIME_FEATURES

# Configuration B: CURRENT + HISTORY (29 features)
CONFIG_B_FEATURES: List[str] = CURRENT_FEATURES + TIME_FEATURES + DO_LAGS + PH_LAGS + TEMP_LAGS

# Configuration C: DO-FOCUSED BASELINE (11 features)
CONFIG_C_FEATURES: List[str] = ["current_do"] + TIME_FEATURES + DO_LAGS

FEATURE_CONFIGS: Dict[str, List[str]] = {
    "config_a": CONFIG_A_FEATURES,
    "config_b": CONFIG_B_FEATURES,
    "config_c": CONFIG_C_FEATURES,
}

# Redundant duplicate columns to explicitly omit from model inputs
REDUNDANT_CURRENT_COLUMNS: List[str] = ["do_t", "ph_t", "temp_t"]

# Non-predictor metadata and quality columns
NON_PREDICTOR_COLUMNS: List[str] = [
    "pond_id",
    "prediction_timestamp",
    "target",
    "target_name",
    "data_quality_status",
]


# ==============================================================================
# 2. DATASET INTEGRITY PRECONDITION CHECKS
# ==============================================================================

def verify_dataset_preconditions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates that the input dataset meets all Phase 3 preconditions.
    Fails fast if any condition is violated.
    """
    errors = []

    # 1. Row count
    if len(df) != 41277:
        errors.append(f"Expected exactly 41,277 rows, found {len(df)}")

    # 2. Column count
    if len(df.columns) != 37:
        errors.append(f"Expected exactly 37 columns, found {len(df.columns)}")

    # 3. Target values
    unique_targets = set(df["target"].unique())
    if unique_targets != {0, 1}:
        errors.append(f"Target values must be exactly {{0, 1}}, found {unique_targets}")

    # 4. No NaNs
    nan_count = int(df.isna().sum().sum())
    if nan_count != 0:
        errors.append(f"Found {nan_count} NaN values in dataset")

    # 5. Current DO threshold
    min_current_do = float(df["current_do"].min())
    if min_current_do < 3.0:
        errors.append(f"All current_do values must be >= 3.0 mg/L, min is {min_current_do}")

    # 6. All 17 ponds represented
    unique_ponds = sorted(df["pond_id"].unique())
    if len(unique_ponds) != 17:
        errors.append(f"Expected 17 unique ponds, found {len(unique_ponds)}")

    # 7. Both classes represented
    target_counts = df["target"].value_counts().to_dict()
    if target_counts.get(0, 0) == 0 or target_counts.get(1, 0) == 0:
        errors.append(f"Both classes must be present, found {target_counts}")

    if errors:
        raise ValueError("Dataset verification failed:\n" + "\n".join(f"- {e}" for e in errors))

    return {
        "status": "PASS",
        "rows": len(df),
        "columns": len(df.columns),
        "safe_count": int(target_counts.get(0, 0)),
        "at_risk_count": int(target_counts.get(1, 0)),
        "pond_count": len(unique_ponds),
    }


# ==============================================================================
# 3. LEAKAGE-SAFE TEMPORAL SPLITTING WITH PURGE GAP
# ==============================================================================

def create_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    purge_hours: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a leak-safe temporal holdout split for each pond independently:
    1. Sorts chronologically by prediction_timestamp.
    2. Computes the train_ratio quantile timestamp (t_cutoff).
    3. Purges all training examples within [t_cutoff - purge_hours, t_cutoff)
       because their 2-hour future target window overlaps into the test period.
    4. Designates [t_cutoff, max_time] as the test period.

    Returns:
        train_df, test_df, purged_df, accounting_df
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["prediction_timestamp"]):
        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])

    train_dfs = []
    test_dfs = []
    purged_dfs = []
    accounting_rows = []

    for pond_id, group in df.groupby("pond_id"):
        group = group.sort_values("prediction_timestamp").reset_index(drop=True)
        total_p = len(group)

        # 80th percentile cutoff
        t_cutoff = group["prediction_timestamp"].quantile(train_ratio)
        t_purge_start = t_cutoff - pd.Timedelta(hours=purge_hours)

        test_mask = group["prediction_timestamp"] >= t_cutoff
        purge_mask = (group["prediction_timestamp"] >= t_purge_start) & (group["prediction_timestamp"] < t_cutoff)
        train_mask = group["prediction_timestamp"] < t_purge_start

        train_sub = group[train_mask]
        purge_sub = group[purge_mask]
        test_sub = group[test_mask]

        train_dfs.append(train_sub)
        purged_dfs.append(purge_sub)
        test_dfs.append(test_sub)

        accounting_rows.append({
            "pond_id": pond_id,
            "total_examples": total_p,
            "train_examples": len(train_sub),
            "purged_examples": len(purge_sub),
            "test_examples": len(test_sub),
            "t_min": group["prediction_timestamp"].min(),
            "t_purge_start": t_purge_start,
            "t_cutoff": t_cutoff,
            "t_max": group["prediction_timestamp"].max(),
            "train_pos": int(train_sub["target"].sum()),
            "train_pos_rate": float(train_sub["target"].mean()),
            "test_pos": int(test_sub["target"].sum()),
            "test_pos_rate": float(test_sub["target"].mean()),
        })

    train_df = pd.concat(train_dfs, ignore_index=True)
    purged_df = pd.concat(purged_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    accounting_df = pd.DataFrame(accounting_rows)

    # Verification identity
    assert len(train_df) + len(purged_df) + len(test_df) == len(df), "Row reconciliation discrepancy in temporal split"

    return train_df, test_df, purged_df, accounting_df


# ==============================================================================
# 4. CLASSIFICATION METRICS
# ==============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes comprehensive early-warning classification metrics.
    Primary metric: PR-AUC (Average Precision score).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # PR-AUC
    pr_auc = float(average_precision_score(y_true, y_prob))

    # ROC-AUC (handle edge case where only one class exists in small test slices)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "specificity": round(spec, 4),
        "accuracy": round(acc, 4),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "test_examples": len(y_true),
        "test_pos": int(y_true.sum()),
        "test_neg": int((y_true == 0).sum()),
    }


# ==============================================================================
# 5. MODEL PIPELINE FACTORY
# ==============================================================================

def get_model(
    model_name: str,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> Any:
    """
    Instantiates standardized, reproducible candidate models:
    - 'logistic_regression': StandardScaler + LogisticRegression(class_weight='balanced')
    - 'random_forest': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced')
    - 'xgboost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=...)
    """
    model_name_lower = model_name.lower().replace(" ", "_")

    if model_name_lower in ["logistic_regression", "lr"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
                solver="lbfgs",
            )),
        ])

    elif model_name_lower in ["random_forest", "rf"]:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name_lower in ["xgboost", "xgb"]:
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: 'logistic_regression', 'random_forest', 'xgboost'")
