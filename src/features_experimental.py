"""
ShinerAI: Experimental Feature Engineering
Leakage-safe row-wise historical temporal trend features.

All features are strictly derived from current and past DO observations
at time T (current_do, do_t_minus_15 through do_t_minus_120).
Zero future target or post-prediction information is ever used.
"""

from typing import List
import pandas as pd
import numpy as np

# Baseline Config C features (11 features)
CONFIG_C_FEATURES: List[str] = [
    "current_do",
    "hour_of_day",
    "minute_of_day",
    "do_t_minus_15",
    "do_t_minus_30",
    "do_t_minus_45",
    "do_t_minus_60",
    "do_t_minus_75",
    "do_t_minus_90",
    "do_t_minus_105",
    "do_t_minus_120",
]

# Rate-of-change delta features (past trend slopes)
RATE_OF_CHANGE_FEATURES: List[str] = [
    "do_change_15",
    "do_change_30",
    "do_change_60",
    "do_change_120",
]

# Second-order rate of change (acceleration / deceleration of DO drop)
ACCELERATION_FEATURES: List[str] = [
    "do_accel_15",
]

# Summary historical statistics over the past 2-hour window
HISTORICAL_STATS_FEATURES: List[str] = [
    "do_min_120",
    "do_max_120",
    "do_mean_120",
    "do_std_120",
    "do_range_120",
]

CONFIG_C_PLUS_TRENDS: List[str] = CONFIG_C_FEATURES + RATE_OF_CHANGE_FEATURES
CONFIG_C_PLUS_ACCEL: List[str] = CONFIG_C_FEATURES + RATE_OF_CHANGE_FEATURES + ACCELERATION_FEATURES
CONFIG_C_PLUS_ALL: List[str] = CONFIG_C_FEATURES + RATE_OF_CHANGE_FEATURES + ACCELERATION_FEATURES + HISTORICAL_STATS_FEATURES


def compute_experimental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes leakage-free row-wise temporal trend and rolling features.
    Operates strictly across existing column values within each individual row.
    """
    out = df.copy()

    # Rate of change: delta between current DO and past observations (mg/L)
    out["do_change_15"] = out["current_do"] - out["do_t_minus_15"]
    out["do_change_30"] = out["current_do"] - out["do_t_minus_30"]
    out["do_change_60"] = out["current_do"] - out["do_t_minus_60"]
    out["do_change_120"] = out["current_do"] - out["do_t_minus_120"]

    # Second derivative: acceleration of DO decline over the last 15-30 min
    out["do_accel_15"] = (out["current_do"] - out["do_t_minus_15"]) - (out["do_t_minus_15"] - out["do_t_minus_30"])

    # 2-hour historical window columns (t=0 down to t=-120 min)
    lag_cols = ["current_do"] + [f"do_t_minus_{m}" for m in [15, 30, 45, 60, 75, 90, 105, 120]]
    out["do_min_120"] = out[lag_cols].min(axis=1)
    out["do_max_120"] = out[lag_cols].max(axis=1)
    out["do_mean_120"] = out[lag_cols].mean(axis=1)
    out["do_std_120"] = out[lag_cols].std(axis=1)
    out["do_range_120"] = out["do_max_120"] - out["do_min_120"]

    return out
