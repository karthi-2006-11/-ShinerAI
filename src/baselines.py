"""
ShinerAI: Stronger Baselines Module
Implements domain-standard time-series baselines:
1. Strict Persistence Baseline (assumes current DO persists into future horizon)
2. 120-minute Linear Trend Forecasting Baseline (extrapolates recent DO slope into 2h horizon)

Evaluated on the exact same leak-free temporal holdout (N = 8,261) as candidate models.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

TREND_WINDOW_COLS = [
    "do_t_minus_120",
    "do_t_minus_105",
    "do_t_minus_90",
    "do_t_minus_75",
    "do_t_minus_60",
    "do_t_minus_45",
    "do_t_minus_30",
    "do_t_minus_15",
    "current_do",
]
TREND_TIME_POINTS = np.array([-120.0, -105.0, -90.0, -75.0, -60.0, -45.0, -30.0, -15.0, 0.0])


def evaluate_persistence_baseline(
    test_df: pd.DataFrame,
    target_col: str = "target",
    current_do_col: str = "current_do",
    threshold_boundary: float = 3.0,
) -> Dict[str, Any]:
    """
    Evaluates the strict persistence baseline.
    Under the ShinerAI operational invariant, all observations entering early warning have:
        current_do >= 3.0 mg/L at time T.
    Strict persistence forecasts that DO remains at current_do for all future steps:
        DO_pred(T + h) = current_do >= 3.0 mg/L.
    Therefore, strict persistence always forecasts that DO will NOT drop below 3.0 mg/L (pred = 0, SAFE).
    """
    y_true = test_df[target_col].values
    N = len(y_true)
    
    # Always predict 0 (SAFE)
    preds = np.zeros(N, dtype=int)
    
    # Continuous risk score: inverse of current DO (closer to 3.0 = higher risk)
    current_do = test_df[current_do_col].values
    risk_scores = -current_do
    
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    
    # Strict persistence binary prediction is all zeros (always SAFE)
    # Binary PR-AUC for a constant classifier equals positive class prevalence
    pr_auc = float(np.mean(y_true))
    roc_auc = 0.5000
    f1 = float(f1_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y_true, preds))
    
    return {
        "model": "Persistence Baseline",
        "feature_set": "Strict Persistence (DO_T persists)",
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "specificity": round(spec, 4),
        "accuracy": round(acc, 4),
        "threshold": 0.50,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "test_examples": N,
    }


def compute_linear_trend_slopes(
    df: pd.DataFrame,
    cols: list = TREND_WINDOW_COLS,
    time_points: np.ndarray = TREND_TIME_POINTS,
) -> np.ndarray:
    """
    Computes ordinary least squares linear slope (mg/L per minute)
    across the historical window for each observation.
    """
    t_mean = time_points.mean()
    t_diff = time_points - t_mean
    denom = np.sum(t_diff**2)
    
    Y = df[cols].values
    Y_mean = Y.mean(axis=1, keepdims=True)
    Y_diff = Y - Y_mean
    slopes = np.sum(Y_diff * t_diff, axis=1) / denom  # mg/L per minute
    return slopes


def evaluate_trend_baseline(
    test_df: pd.DataFrame,
    target_col: str = "target",
    current_do_col: str = "current_do",
    lookahead_minutes: float = 120.0,
    hypoxia_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Evaluates the 120-minute Linear Trend Forecasting Baseline.
    
    Methodology:
    1. Fits linear slope m (mg/L per min) over the 9 historical points [t-120m ... t].
    2. Extrapolates DO forward 120 minutes:
       If slope < 0 (declining): minimum predicted DO is current_do + slope * 120.0.
       If slope >= 0 (rising/flat): minimum predicted DO is current_do.
    3. Binary decision: If minimum predicted DO < 3.0 mg/L, predict 1 (AT_RISK); else 0 (SAFE).
    4. Continuous risk score: -minimum_predicted_do (for PR-AUC and ROC-AUC).
    """
    y_true = test_df[target_col].values
    N = len(y_true)
    
    slopes = compute_linear_trend_slopes(test_df)
    current_do = test_df[current_do_col].values
    
    min_extrapolated_do = np.where(
        slopes < 0,
        current_do + slopes * lookahead_minutes,
        current_do
    )
    
    preds = (min_extrapolated_do < hypoxia_threshold).astype(int)
    risk_scores = -min_extrapolated_do
    
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    
    pr_auc = float(average_precision_score(y_true, risk_scores))
    roc_auc = float(roc_auc_score(y_true, risk_scores))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y_true, preds))
    
    return {
        "model": "Linear Trend Baseline",
        "feature_set": "120-min Linear Trend Extrapolation (9 Historical DO Lags)",
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "specificity": round(spec, 4),
        "accuracy": round(acc, 4),
        "threshold": 3.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "test_examples": N,
    }
