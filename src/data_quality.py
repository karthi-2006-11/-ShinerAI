"""
data_quality.py
===============
Quality control, statistical audit, sampling interval analysis, and feasibility
check functions for the Fish Farm Water Quality Early Warning System.

This module implements Phase 1 dataset auditing:
1. Pond-by-pond and combined statistical audits
2. Sampling interval continuity and gap detection
3. Quality Control (QC) flag frequency and overlap analysis
4. Equipment artifact (exact zero-value) reporting
5. DO < 3 mg/L threshold feasibility evaluation for future ML modeling
"""

from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd


IMPORTANT_QC_FLAGS = [
    "time_gap_>20min",
    "DO_<0.1",
    "DO_jump_>2",
    "DO_>10_before_noon",
    "pH_out_of_range",
    "pH_jump_>1",
]


def audit_single_pond(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a complete descriptive and data quality audit for a single pond.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for a single pond, containing timestamp, sensor readings, and QC flags.

    Returns
    -------
    Dict[str, Any]
        Dictionary of calculated audit metrics.
    """
    if df.empty:
        raise ValueError("Cannot audit an empty pond DataFrame.")

    # Ensure sorted order by timestamp
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    ts = df_sorted["timestamp"]

    # File and identification
    source_file = df_sorted["source_file"].iloc[0] if "source_file" in df_sorted.columns else "unknown"
    pond_id = df_sorted["pond_id"].iloc[0] if "pond_id" in df_sorted.columns else "unknown"
    num_rows = len(df_sorted)

    # Timestamp metrics
    earliest_ts = ts.min()
    latest_ts = ts.max()
    duration = latest_ts - earliest_ts
    unique_ts = ts.nunique()
    duplicate_ts = int(ts.duplicated().sum())

    # Sampling intervals
    ts_diffs = ts.diff().dropna()
    expected_interval_min = 15.0
    median_interval_sec = ts_diffs.median().total_seconds() if not ts_diffs.empty else np.nan
    median_interval_str = str(ts_diffs.median()) if not ts_diffs.empty else "N/A"

    # Percentage of intervals close to 15 minutes (between 14 and 16 minutes inclusive)
    if not ts_diffs.empty:
        close_15_mask = (ts_diffs >= pd.Timedelta(minutes=14)) & (ts_diffs <= pd.Timedelta(minutes=16))
        pct_close_15 = float(close_15_mask.sum() / len(ts_diffs) * 100.0)
        gaps_gt_20 = int((ts_diffs > pd.Timedelta(minutes=20)).sum())
        max_gap = ts_diffs.max()
        max_gap_str = str(max_gap)
    else:
        pct_close_15 = np.nan
        gaps_gt_20 = 0
        max_gap = pd.Timedelta(0)
        max_gap_str = "0"

    # Sensor measurements (numeric columns)
    do_series = df_sorted["do_mg_l"] if "do_mg_l" in df_sorted.columns else df_sorted["DO (mg/L)"]
    ph_series = df_sorted["ph"] if "ph" in df_sorted.columns else df_sorted["pH"]
    temp_col_name = "temperature_c" if "temperature_c" in df_sorted.columns else [c for c in df_sorted.columns if "temp" in c.lower()][0]
    temp_series = df_sorted[temp_col_name]

    # Missing values (NaN / Null)
    missing_do = int(do_series.isna().sum())
    missing_ph = int(ph_series.isna().sum())
    missing_temp = int(temp_series.isna().sum())

    # Exact zero values (equipment artifacts)
    zero_do = int((do_series == 0.0).sum())
    zero_ph = int((ph_series == 0.0).sum())
    zero_temp = int((temp_series == 0.0).sum())

    # Summary statistics for DO
    valid_do = do_series.dropna()
    min_do = float(valid_do.min()) if not valid_do.empty else np.nan
    max_do = float(valid_do.max()) if not valid_do.empty else np.nan
    mean_do = float(valid_do.mean()) if not valid_do.empty else np.nan
    median_do = float(valid_do.median()) if not valid_do.empty else np.nan

    # Summary statistics for pH
    valid_ph = ph_series.dropna()
    min_ph = float(valid_ph.min()) if not valid_ph.empty else np.nan
    max_ph = float(valid_ph.max()) if not valid_ph.empty else np.nan
    mean_ph = float(valid_ph.mean()) if not valid_ph.empty else np.nan
    median_ph = float(valid_ph.median()) if not valid_ph.empty else np.nan

    # Summary statistics for Temperature
    valid_temp = temp_series.dropna()
    min_temp = float(valid_temp.min()) if not valid_temp.empty else np.nan
    max_temp = float(valid_temp.max()) if not valid_temp.empty else np.nan
    mean_temp = float(valid_temp.mean()) if not valid_temp.empty else np.nan
    median_temp = float(valid_temp.median()) if not valid_temp.empty else np.nan

    return {
        "filename": source_file,
        "pond_id": pond_id,
        "rows": num_rows,
        "earliest_timestamp": str(earliest_ts),
        "latest_timestamp": str(latest_ts),
        "duration_days": round(duration.total_seconds() / 86400.0, 2),
        "duration_str": str(duration),
        "unique_timestamps": unique_ts,
        "duplicate_timestamps": duplicate_ts,
        "expected_interval_min": expected_interval_min,
        "median_interval": median_interval_str,
        "pct_close_15min": round(pct_close_15, 2),
        "gaps_gt_20min": gaps_gt_20,
        "max_gap": max_gap_str,
        "missing_do": missing_do,
        "missing_ph": missing_ph,
        "missing_temp": missing_temp,
        "zero_do": zero_do,
        "zero_ph": zero_ph,
        "zero_temp": zero_temp,
        "min_do": round(min_do, 2),
        "max_do": round(max_do, 2),
        "mean_do": round(mean_do, 2),
        "median_do": round(median_do, 2),
        "min_ph": round(min_ph, 2),
        "max_ph": round(max_ph, 2),
        "mean_ph": round(mean_ph, 2),
        "median_ph": round(median_ph, 2),
        "min_temp": round(min_temp, 2),
        "max_temp": round(max_temp, 2),
        "mean_temp": round(mean_temp, 2),
        "median_temp": round(median_temp, 2),
    }


def audit_all_ponds(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Run the audit for each pond and combine the results into a DataFrame.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        List of DataFrames, one per pond.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row contains audit metrics for one pond.
    """
    records = []
    for df in dfs:
        metrics = audit_single_pond(df)
        records.append(metrics)

    summary_df = pd.DataFrame(records)
    return summary_df.sort_values("pond_id").reset_index(drop=True)


def audit_combined_dataset(combined_df: pd.DataFrame, individual_pond_audits: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Calculate dataset-level audit statistics across all ponds combined.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Unified DataFrame containing all pond records.
    individual_pond_audits : Optional[pd.DataFrame]
        Precomputed pond audits to aggregate pond-specific metrics like gaps.

    Returns
    -------
    Dict[str, Any]
        Dictionary of overall dataset audit statistics.
    """
    num_ponds = int(combined_df["pond_id"].nunique())
    total_rows = len(combined_df)
    earliest_ts = combined_df["timestamp"].min()
    latest_ts = combined_df["timestamp"].max()
    overall_duration = latest_ts - earliest_ts

    do_series = combined_df["do_mg_l"]
    ph_series = combined_df["ph"]
    temp_series = combined_df["temperature_c"]

    missing_do = int(do_series.isna().sum())
    missing_ph = int(ph_series.isna().sum())
    missing_temp = int(temp_series.isna().sum())

    zero_do = int((do_series == 0.0).sum())
    zero_ph = int((ph_series == 0.0).sum())
    zero_temp = int((temp_series == 0.0).sum())

    valid_do = do_series.dropna()
    valid_ph = ph_series.dropna()
    valid_temp = temp_series.dropna()

    # Sum gaps and duplicates across ponds
    if individual_pond_audits is not None:
        total_gaps_gt_20 = int(individual_pond_audits["gaps_gt_20min"].sum())
        total_dups = int(individual_pond_audits["duplicate_timestamps"].sum())
        mean_pct_close_15 = round(float(individual_pond_audits["pct_close_15min"].mean()), 2)
    else:
        total_gaps_gt_20 = 0
        total_dups = int(combined_df.groupby("pond_id")["timestamp"].apply(lambda s: s.duplicated().sum()).sum())
        mean_pct_close_15 = np.nan

    return {
        "num_ponds": num_ponds,
        "total_rows": total_rows,
        "earliest_timestamp": str(earliest_ts),
        "latest_timestamp": str(latest_ts),
        "overall_duration_days": round(overall_duration.total_seconds() / 86400.0, 2),
        "total_duplicate_timestamps": total_dups,
        "total_gaps_gt_20min": total_gaps_gt_20,
        "mean_pct_close_15min": mean_pct_close_15,
        "missing_do": missing_do,
        "missing_ph": missing_ph,
        "missing_temp": missing_temp,
        "zero_do": zero_do,
        "zero_ph": zero_ph,
        "zero_temp": zero_temp,
        "min_do": round(float(valid_do.min()), 2),
        "max_do": round(float(valid_do.max()), 2),
        "mean_do": round(float(valid_do.mean()), 2),
        "median_do": round(float(valid_do.median()), 2),
        "min_ph": round(float(valid_ph.min()), 2),
        "max_ph": round(float(valid_ph.max()), 2),
        "mean_ph": round(float(valid_ph.mean()), 2),
        "median_ph": round(float(valid_ph.median()), 2),
        "min_temp": round(float(valid_temp.min()), 2),
        "max_temp": round(float(valid_temp.max()), 2),
        "mean_temp": round(float(valid_temp.mean()), 2),
        "median_temp": round(float(valid_temp.median()), 2),
    }


def analyze_qc_flags(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze quality control flag frequencies and multi-flag co-occurrences.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for a single pond or all ponds combined.

    Returns
    -------
    Dict[str, Any]
        Dictionary with counts for each flag, rows with multiple flags,
        and total flagged rows.
    """
    dt_flags = df["qc_flag_datetime"].fillna("").astype(str) if "qc_flag_datetime" in df.columns else pd.Series([""] * len(df))
    do_flags = df["qc_flag_do"].fillna("").astype(str) if "qc_flag_do" in df.columns else pd.Series([""] * len(df))
    ph_flags = df["qc_flag_ph"].fillna("").astype(str) if "qc_flag_ph" in df.columns else pd.Series([""] * len(df))

    # Combine into a single text representation per row
    combined_flags = dt_flags + "; " + do_flags + "; " + ph_flags

    flag_counts = {flag: 0 for flag in IMPORTANT_QC_FLAGS}
    multi_flag_rows = 0
    total_flagged_rows = 0

    for row_text in combined_flags:
        active = [fl for fl in IMPORTANT_QC_FLAGS if fl in row_text]
        for fl in active:
            flag_counts[fl] += 1
        if len(active) > 1:
            multi_flag_rows += 1
        if len(active) > 0:
            total_flagged_rows += 1

    return {
        "flag_counts": flag_counts,
        "multi_flag_rows": multi_flag_rows,
        "total_flagged_rows": total_flagged_rows,
        "total_rows": len(df),
        "pct_flagged": round(total_flagged_rows / len(df) * 100.0, 2) if len(df) > 0 else 0.0,
    }


def summarize_qc_by_pond(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a per-pond breakdown of QC flags and artifact zeros.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined DataFrame of all pond data.

    Returns
    -------
    pd.DataFrame
        Table summarizing QC flag counts and zero readings for each pond.
    """
    records = []
    for pond_id, group in combined_df.groupby("pond_id"):
        qc_res = analyze_qc_flags(group)
        rec = {
            "pond_id": pond_id,
            "total_rows": len(group),
            "flagged_rows": qc_res["total_flagged_rows"],
            "pct_flagged": qc_res["pct_flagged"],
            "multi_flag_rows": qc_res["multi_flag_rows"],
            "zero_do_count": int((group["do_mg_l"] == 0.0).sum()),
            "zero_ph_count": int((group["ph"] == 0.0).sum()),
            "zero_temp_count": int((group["temperature_c"] == 0.0).sum()),
        }
        for flag_name, count in qc_res["flag_counts"].items():
            rec[f"flag_{flag_name}"] = count
        records.append(rec)

    summary_df = pd.DataFrame(records)
    # Sort descending by percentage of flagged rows
    return summary_df.sort_values(by="pct_flagged", ascending=False).reset_index(drop=True)


def check_do_feasibility(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
    """
    Perform a feasibility analysis for predicting dissolved oxygen falling below
    a provisional threshold (e.g. 3.0 mg/L).

    According to dataset documentation:
    Exact zero values (DO == 0.0) are sensor initialization/power artifacts,
    not valid biological dissolved oxygen levels. They are excluded from the
    valid DO calculations and documented.

    Parameters
    ----------
    df : pd.DataFrame
        Single pond DataFrame.
    threshold : float
        Provisional DO threshold in mg/L (default: 3.0).

    Returns
    -------
    Dict[str, Any]
        Feasibility statistics including low-DO observation counts, percentages,
        episodes, and longest continuous low-DO duration.
    """
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    pond_id = df_sorted["pond_id"].iloc[0] if "pond_id" in df_sorted.columns else "unknown"

    do_col = df_sorted["do_mg_l"]

    # Filter out equipment artifact zeros (documented decision)
    # Valid readings are positive non-null measurements
    valid_mask = (do_col > 0.0) & (do_col.notna())
    valid_df = df_sorted[valid_mask].copy()

    total_valid = len(valid_df)
    below_thresh_mask = valid_df["do_mg_l"] < threshold
    valid_below_count = int(below_thresh_mask.sum())
    pct_below = round(valid_below_count / total_valid * 100.0, 2) if total_valid > 0 else 0.0

    # Calculate episodes:
    # A continuous episode is a contiguous block of valid DO < threshold readings.
    # An episode ends if DO >= threshold OR if there is a gap > 30 minutes between consecutive readings.
    if valid_below_count > 0:
        time_diff = valid_df["timestamp"].diff()
        new_episode = below_thresh_mask & (
            (~below_thresh_mask.shift(1, fill_value=False)) | (time_diff > pd.Timedelta(minutes=30))
        )
        episode_ids = new_episode.cumsum()
        valid_df["episode_id"] = episode_ids

        low_subset = valid_df[below_thresh_mask]
        num_episodes = int(low_subset["episode_id"].nunique())

        # Longest continuous episode duration
        # For an episode with n points: (t_last - t_first) + 15 min (or 15 min if single reading)
        durations = low_subset.groupby("episode_id")["timestamp"].agg(
            lambda t: (t.max() - t.min()) if len(t) > 1 else pd.Timedelta(minutes=15)
        )
        max_duration = durations.max()
        max_duration_str = str(max_duration)
        max_duration_hours = round(max_duration.total_seconds() / 3600.0, 2)

        # Longest episode in terms of consecutive rows
        max_consecutive_readings = int(low_subset.groupby("episode_id").size().max())
    else:
        num_episodes = 0
        max_duration_str = "0 days 00:00:00"
        max_duration_hours = 0.0
        max_consecutive_readings = 0

    return {
        "pond_id": pond_id,
        "total_valid_do_obs": total_valid,
        "valid_below_3_count": valid_below_count,
        "pct_below_3": pct_below,
        "num_episodes": num_episodes,
        "longest_period_str": max_duration_str,
        "longest_period_hours": max_duration_hours,
        "max_consecutive_readings": max_consecutive_readings,
    }


def feasibility_summary_all_ponds(dfs: List[pd.DataFrame], threshold: float = 3.0) -> pd.DataFrame:
    """
    Run DO threshold feasibility evaluation for all ponds and compile a comparative table.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        List of pond DataFrames.
    threshold : float
        DO threshold in mg/L (default: 3.0).

    Returns
    -------
    pd.DataFrame
        Table of feasibility metrics per pond.
    """
    records = []
    for df in dfs:
        rec = check_do_feasibility(df, threshold=threshold)
        records.append(rec)

    summary_df = pd.DataFrame(records)
    return summary_df.sort_values(by="pct_below_3", ascending=False).reset_index(drop=True)
