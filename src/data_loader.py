"""
data_loader.py
==============
Reusable data loading functions for the Fish Farm Water Quality Early Warning System.

This module is responsible for:
1. Discovering raw CSV files in data/raw/csv/
2. Distinguishing pond continuous time-series files from metadata/comparison files
3. Loading individual pond CSVs and parsing timestamps
4. Adding identifying columns (pond_id, source_file)
5. Combining all pond datasets into a single structured Pandas DataFrame
6. Preserving all original sensor readings without modification
"""

import os
from pathlib import Path
from typing import List, Tuple, Union, Optional
import pandas as pd


# Known metadata and comparison files that should be excluded from pond time-series analysis
METADATA_FILES = {
    "continuous_monitor_vs_prodss_comparison.csv",
    "key_events.csv",
    "prodss_and_photometer.csv",
    "prodss_vs_continuous_monitors_comparison.csv",
    "qc_flags.csv",
}


def find_csv_files(data_dir: Union[str, Path] = "data/raw/csv") -> Tuple[List[Path], List[Path]]:
    """
    Scan the specified directory for all CSV files and separate pond time-series
    files from metadata/reference CSV files.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory where raw CSV files are stored. Defaults to 'data/raw/csv'.

    Returns
    -------
    Tuple[List[Path], List[Path]]
        A tuple of (pond_files, metadata_files) as sorted lists of Path objects.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at: {data_path.resolve()}")

    all_csvs = sorted(list(data_path.glob("*.csv")))
    pond_files: List[Path] = []
    metadata_files: List[Path] = []

    for file_path in all_csvs:
        filename_lower = file_path.name.lower()
        if filename_lower in METADATA_FILES:
            metadata_files.append(file_path)
        elif filename_lower.startswith("ara2_"):
            pond_files.append(file_path)
        else:
            # Any non-ara2 file is treated as supplementary/metadata
            metadata_files.append(file_path)

    return pond_files, metadata_files


def extract_pond_id(file_path: Union[str, Path]) -> str:
    """
    Extract the clean pond identifier from a file path.
    Example: 'data/raw/csv/ara2_0677080b.csv' -> 'ara2_0677080b'.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the pond CSV file.

    Returns
    -------
    str
        Extracted pond identifier.
    """
    path = Path(file_path)
    return path.stem


def load_pond_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single pond time-series CSV file, parse the timestamp,
    add the pond_id column, and preserve all original measurements.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the pond CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed timestamps, pond_id, original sensor
        readings, and QC flag columns.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Pond CSV not found: {path.resolve()}")

    # Read CSV using UTF-8 encoding (dataset contains degree symbols like °C)
    df = pd.read_csv(path, encoding="utf-8")

    # Detect the timestamp column (typically 'Date/Time (IST)')
    dt_col_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() and "flag" not in c.lower()]
    if not dt_col_candidates:
        raise ValueError(f"No date/time column found in {path.name}")
    raw_dt_col = dt_col_candidates[0]

    # Parse timestamps to datetime64[ns]
    parsed_timestamps = pd.to_datetime(df[raw_dt_col], errors="coerce")
    if parsed_timestamps.isna().any():
        num_invalid = parsed_timestamps.isna().sum()
        print(f"Warning: {num_invalid} timestamp(s) could not be parsed in {path.name}")

    # Add standard timestamp and Timestamp columns for seamless access
    df["timestamp"] = parsed_timestamps
    df["Timestamp"] = parsed_timestamps

    # Add metadata identifiers
    pond_id = extract_pond_id(path)
    df["pond_id"] = pond_id
    df["source_file"] = path.name

    # Standardize column references while preserving original column names
    # Detect DO column
    do_cols = [c for c in df.columns if c.startswith("DO") and "flag" not in c.lower()]
    if do_cols:
        df["do_mg_l"] = pd.to_numeric(df[do_cols[0]], errors="coerce")

    # Detect pH column
    ph_cols = [c for c in df.columns if c == "pH" or (c.lower() == "ph" and "flag" not in c.lower())]
    if ph_cols:
        df["ph"] = pd.to_numeric(df[ph_cols[0]], errors="coerce")

    # Detect Temperature column
    temp_cols = [c for c in df.columns if "temp" in c.lower() and "flag" not in c.lower()]
    if temp_cols:
        df["temperature_c"] = pd.to_numeric(df[temp_cols[0]], errors="coerce")

    # Standardize QC flag column names
    qc_dt_cols = [c for c in df.columns if "qc" in c.lower() and ("date" in c.lower() or "time" in c.lower())]
    if qc_dt_cols:
        df["qc_flag_datetime"] = df[qc_dt_cols[0]].fillna("").astype(str)

    qc_do_cols = [c for c in df.columns if "qc" in c.lower() and "do" in c.lower()]
    if qc_do_cols:
        df["qc_flag_do"] = df[qc_do_cols[0]].fillna("").astype(str)

    qc_ph_cols = [c for c in df.columns if "qc" in c.lower() and "ph" in c.lower()]
    if qc_ph_cols:
        df["qc_flag_ph"] = df[qc_ph_cols[0]].fillna("").astype(str)

    # Sort strictly by timestamp to maintain temporal sequence
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    return df


def load_all_ponds(data_dir: Union[str, Path] = "data/raw/csv") -> List[pd.DataFrame]:
    """
    Load all discovered pond CSV files into a list of DataFrames.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory where raw CSV files are stored. Defaults to 'data/raw/csv'.

    Returns
    -------
    List[pd.DataFrame]
        List of loaded DataFrames, one per pond.
    """
    pond_files, _ = find_csv_files(data_dir)

    if not pond_files:
        raise FileNotFoundError(
            f"No pond CSV files (expected 17) were found in '{data_dir}'. "
            "Please ensure the raw files are placed in data/raw/csv/."
        )

    dfs: List[pd.DataFrame] = []
    for file_path in pond_files:
        df = load_pond_csv(file_path)
        dfs.append(df)

    return dfs


def combine_pond_data(data_source: Union[str, Path, List[pd.DataFrame]] = "data/raw/csv") -> pd.DataFrame:
    """
    Combine all individual pond DataFrames into a single unified DataFrame.

    Parameters
    ----------
    data_source : Union[str, Path, List[pd.DataFrame]]
        Either a list of DataFrames or the path to the directory containing raw CSVs.

    Returns
    -------
    pd.DataFrame
        Unified DataFrame containing all ponds, sorted by pond_id and timestamp.
    """
    if isinstance(data_source, list):
        dfs = data_source
    else:
        dfs = load_all_ponds(data_source)

    if not dfs:
        raise ValueError("Cannot combine empty list of pond DataFrames.")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by pond_id and timestamp for systematic time-series ordering
    combined_df = combined_df.sort_values(by=["pond_id", "timestamp"]).reset_index(drop=True)

    return combined_df
