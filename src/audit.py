"""
audit.py
========
Executable audit pipeline for the Fish Farm Water Quality Early Warning System.

Usage:
    python src/audit.py

Actions performed:
1. Discovers and loads all 17 pond time-series CSV files
2. Validates timestamp parsing, schema, and row counts
3. Runs comprehensive data quality audits:
   - Sampling intervals and recording gaps
   - Quality control (QC) flag distribution and co-occurrences
   - Equipment artifact zero-value readings
   - DO < 3 mg/L threshold feasibility for future ML modeling
4. Saves tabular reports in results/reports/
5. Generates figures in results/figures/ (single figures, no subplots, default colors)
6. Prints a concise executive summary to the terminal
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path so 'src' imports work from any working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.data_loader as dl
import src.data_quality as dq


def generate_figures(dfs: List[pd.DataFrame], combined_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Generate all required Phase 1 figures.

    Constraints strictly honored:
    - Use separate figures, NOT subplots
    - Do NOT manually specify chart colors (let matplotlib default color cycle handle styling)
    - Save all plots to results/figures/
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # 1. DO over time for each pond (17 separate figures, one per pond)
    for df in dfs:
        pond_id = df["pond_id"].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["timestamp"], df["do_mg_l"])
        ax.set_title(f"Dissolved Oxygen (DO) Over Time - Pond: {pond_id}")
        ax.set_xlabel("Date/Time (IST)")
        ax.set_ylabel("DO (mg/L)")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        filepath = output_dir / f"do_time_series_{pond_id}.png"
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        generated_files.append(filepath)

    # 2. DO distribution for all ponds
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_do = combined_df["do_mg_l"].dropna()
    ax.hist(valid_do, bins=50)
    ax.set_title("DO Distribution - All Ponds Combined")
    ax.set_xlabel("DO (mg/L)")
    ax.set_ylabel("Frequency (Count)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    filepath = output_dir / "do_distribution.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    generated_files.append(filepath)

    # 3. pH distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_ph = combined_df["ph"].dropna()
    ax.hist(valid_ph, bins=50)
    ax.set_title("pH Distribution - All Ponds Combined")
    ax.set_xlabel("pH")
    ax.set_ylabel("Frequency (Count)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    filepath = output_dir / "ph_distribution.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    generated_files.append(filepath)

    # 4. Temperature distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_temp = combined_df["temperature_c"].dropna()
    ax.hist(valid_temp, bins=50)
    ax.set_title("Temperature Distribution - All Ponds Combined")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Frequency (Count)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    filepath = output_dir / "temperature_distribution.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    generated_files.append(filepath)

    # 5. Number of observations per pond
    obs_counts = combined_df.groupby("pond_id").size().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(obs_counts.index, obs_counts.values)
    ax.set_title("Number of Observations per Pond")
    ax.set_xlabel("Number of Rows")
    ax.set_ylabel("Pond ID")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    fig.tight_layout()
    filepath = output_dir / "observations_per_pond.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    generated_files.append(filepath)

    # 6. Sampling-gap distribution
    all_diffs_min: List[float] = []
    for df in dfs:
        diffs = df["timestamp"].diff().dropna()
        all_diffs_min.extend(diffs.dt.total_seconds() / 60.0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # Clip large gaps for meaningful visualization around normal 15 min interval
    clipped_diffs = [min(m, 60.0) for m in all_diffs_min]
    ax.hist(clipped_diffs, bins=40)
    ax.set_title("Sampling Interval Gap Distribution (Capped at 60 min for detail)")
    ax.set_xlabel("Interval Duration (Minutes)")
    ax.set_ylabel("Frequency (Count)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    filepath = output_dir / "sampling_gap_distribution.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    generated_files.append(filepath)

    # 7. DO values below 3 mg/L by pond
    # Calculate count of valid DO < 3.0 mg/L per pond
    low_do_counts = {}
    for df in dfs:
        pid = df["pond_id"].iloc[0]
        # Exclude exact zeros (equipment artifacts)
        valid_low = ((df["do_mg_l"] > 0.0) & (df["do_mg_l"] < 3.0)).sum()
        low_do_counts[pid] = valid_low

    low_do_series = pd.Series(low_do_counts).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(low_do_series.index, low_do_series.values)
    ax.set_title("Valid DO Observations Below 3 mg/L by Pond (Excluding 0 mg/L Artifacts)")
    ax.set_xlabel("Count of Readings < 3.0 mg/L")
    ax.set_ylabel("Pond ID")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    fig.tight_layout()
    filepath = output_dir / "do_below_3_by_pond.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    generated_files.append(filepath)

    return generated_files


def run_audit(
    raw_data_dir: str = "data/raw/csv",
    reports_dir: str = "results/reports",
    figures_dir: str = "results/figures",
) -> Dict[str, Any]:
    """
    Execute the entire Phase 1 audit pipeline.

    Returns
    -------
    Dict[str, Any]
        Dictionary of core audit statistics.
    """
    raw_path = Path(raw_data_dir)
    rep_path = Path(reports_dir)
    fig_path = Path(figures_dir)

    rep_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    # 1. Discover files
    pond_files, meta_files = dl.find_csv_files(raw_path)
    if not pond_files:
        raise FileNotFoundError(
            f"Expected pond CSV files were not found in {raw_path.resolve()}. "
            "Please check that raw CSV files are placed in data/raw/csv/."
        )

    # 2. Load pond data
    dfs = dl.load_all_ponds(raw_path)
    combined_df = dl.combine_pond_data(dfs)

    # 3. Perform statistical & quality audits
    pond_audit_df = dq.audit_all_ponds(dfs)
    combined_stats = dq.audit_combined_dataset(combined_df, pond_audit_df)
    qc_summary_df = dq.summarize_qc_by_pond(combined_df)
    feasibility_df = dq.feasibility_summary_all_ponds(dfs, threshold=3.0)

    # 4. Save reports to results/reports/
    pond_audit_df.to_csv(rep_path / "pond_summary.csv", index=False)
    qc_summary_df.to_csv(rep_path / "qc_summary.csv", index=False)
    feasibility_df.to_csv(rep_path / "feasibility_summary.csv", index=False)

    with open(rep_path / "combined_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined_stats, f, indent=2)

    # 5. Generate figures
    generated_figures = generate_figures(dfs, combined_df, fig_path)

    # 6. Overall stats
    total_low_do_valid = int(feasibility_df["valid_below_3_count"].sum())
    total_gaps_gt_20 = int(pond_audit_df["gaps_gt_20min"].sum())

    # Print terminal output matching requested concise summary
    print("==================================================")
    print("PHASE 1 DATASET AUDIT")
    print("==================================================")
    print(f"Pond files found: {len(pond_files)}")
    print(f"Metadata files detected: {len(meta_files)}")
    print(f"Total rows: {combined_stats['total_rows']:,}")
    print(f"Date range: {combined_stats['earliest_timestamp']} -> {combined_stats['latest_timestamp']}")
    print(f"DO < 3 mg/L observations: {total_low_do_valid:,} ({total_low_do_valid / combined_stats['total_rows'] * 100:.2f}%)")
    print(f"Gaps > 20 min: {total_gaps_gt_20:,}")
    print(f"Equipment artifact zeros (DO=0): {combined_stats['zero_do']:,}")
    print(f"Equipment artifact zeros (pH=0): {combined_stats['zero_ph']:,}")
    print(f"Equipment artifact zeros (Temp=0): {combined_stats['zero_temp']:,}")
    print(f"Figures generated: {len(generated_figures)} in {fig_path.as_posix()}")
    print(f"Reports saved in: {rep_path.as_posix()}")
    print("==================================================")

    return {
        "pond_files_count": len(pond_files),
        "total_rows": combined_stats["total_rows"],
        "earliest_timestamp": combined_stats["earliest_timestamp"],
        "latest_timestamp": combined_stats["latest_timestamp"],
        "total_low_do_valid": total_low_do_valid,
        "total_gaps_gt_20": total_gaps_gt_20,
        "combined_stats": combined_stats,
        "pond_audit_df": pond_audit_df,
        "qc_summary_df": qc_summary_df,
        "feasibility_df": feasibility_df,
    }


if __name__ == "__main__":
    run_audit()
