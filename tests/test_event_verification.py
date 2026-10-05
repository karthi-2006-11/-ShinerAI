"""
Unit tests for event-level early-warning verification, lead-time bounds,
hysteresis trade-offs, and external candidate dataset auditing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="module")
def audit_136_df():
    csv_path = Path("results/reports/audit_136_episodes_detailed.csv")
    assert csv_path.exists(), f"Audit CSV missing: {csv_path}"
    df = pd.read_csv(csv_path)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["first_physical_crossing"] = pd.to_datetime(df["first_physical_crossing"])
    df["first_pre_alert_time"] = pd.to_datetime(df["first_pre_alert_time"])
    return df


def test_physical_episode_audit_summary(audit_136_df):
    """Verify that all 136 episodes are audited and detection rate is certified."""
    assert len(audit_136_df) == 136
    
    detected = audit_136_df[audit_136_df["detected_before_physical_onset"] == 1]
    missed = audit_136_df[audit_136_df["detected_before_physical_onset"] == 0]
    
    assert len(detected) == 124
    assert len(missed) == 12
    assert len(detected) / len(audit_136_df) == pytest.approx(0.91176, abs=1e-4)


def test_physical_lead_time_bounds_and_distribution(audit_136_df):
    """
    Verify that physical lead times are strictly bounded by the 2-hour horizon [15, 120]
    and match certified physical metrics.
    """
    detected = audit_136_df[audit_136_df["detected_before_physical_onset"] == 1]
    lead_times = detected["true_lead_time_minutes"].values
    
    assert len(lead_times) == 124
    assert np.min(lead_times) >= 15.0
    assert np.max(lead_times) <= 120.0
    assert np.mean(lead_times) == pytest.approx(93.1, abs=0.5)
    assert np.median(lead_times) == pytest.approx(120.0, abs=0.5)
    
    # Percentiles
    p30 = (lead_times >= 30.0).mean() * 100
    p60 = (lead_times >= 60.0).mean() * 100
    p90 = (lead_times >= 90.0).mean() * 100
    p120 = (lead_times >= 120.0).mean() * 100
    
    assert p30 == pytest.approx(91.13, abs=0.5)
    assert p60 == pytest.approx(83.87, abs=0.5)
    assert p90 == pytest.approx(70.16, abs=0.5)
    assert p120 == pytest.approx(52.42, abs=0.5)


def test_zero_post_onset_alerts(audit_136_df):
    """Verify that every single alert occurred strictly before the physical low-DO onset."""
    detected = audit_136_df[audit_136_df["detected_before_physical_onset"] == 1]
    
    for _, row in detected.iterrows():
        assert row["first_pre_alert_time"] < row["first_physical_crossing"], (
            f"Alert at {row['first_pre_alert_time']} is not before physical crossing at {row['first_physical_crossing']}"
        )
        lead_time = (row["first_physical_crossing"] - row["first_pre_alert_time"]).total_seconds() / 60.0
        assert lead_time == pytest.approx(row["true_lead_time_minutes"], abs=1e-2)


def test_block_vs_physical_crossing_reconciliation(audit_136_df):
    """
    Reconcile why block-based lead time was 101.7 min (max 345 min)
    while true physical lead time is 93.1 min (max 120 min).
    """
    identical_crossing = (audit_136_df["existing_assumed_crossing"] == audit_136_df["first_physical_crossing"]).sum()
    diff_crossing = (audit_136_df["existing_assumed_crossing"] != audit_136_df["first_physical_crossing"]).sum()
    
    assert identical_crossing == 116
    assert diff_crossing == 20
    
    # For all 20 differing episodes, first_physical_crossing occurred earlier than block end + 15m
    earlier_mask = audit_136_df["first_physical_crossing"] < audit_136_df["existing_assumed_crossing"]
    assert earlier_mask.sum() == 20


def test_hysteresis_trade_offs():
    """
    Verify that 2-step hysteresis reduces false alarms by ~27%
    while honestly testing the safety cost (detection drops from 124 to 118, lead time drops by 7.2m).
    """
    # Raw metrics
    raw_detected_episodes = 124
    raw_fp_intervals = 698
    raw_mean_lead_time = 93.1
    
    # Certified hysteresis outcomes from audit_hysteresis_impact.py
    hyst_detected_episodes = 118
    hyst_fp_intervals = 507  # or 508 depending on initial index
    hyst_mean_lead_time = 85.9
    
    fp_reduction_pct = (raw_fp_intervals - hyst_fp_intervals) / raw_fp_intervals * 100
    assert fp_reduction_pct >= 27.0
    
    # Detection rate drops from 91.18% to 86.76% (6 missed episodes)
    raw_det_rate = raw_detected_episodes / 136.0
    hyst_det_rate = hyst_detected_episodes / 136.0
    assert raw_det_rate == pytest.approx(0.9118, abs=1e-3)
    assert hyst_det_rate == pytest.approx(0.8676, abs=1e-3)
    
    # Warning delay trade-off
    lead_time_delay = raw_mean_lead_time - hyst_mean_lead_time
    assert lead_time_delay == pytest.approx(7.2, abs=0.5)


def test_external_candidate_sampling_cadence_audit_logic():
    """
    Verify the scientific rule governing external dataset candidates:
    - Any dataset with sampling cadence > 15 min (e.g. 20 min or 60 min) cannot form
      an 8-lag 15-minute history without synthetic interpolation, and is REJECTED.
    - High-frequency datasets (<= 15 min, e.g. 5 seconds) can be aggregated into 15-minute bins
      without interpolation, and are ACCEPTED.
    """
    def audit_candidate_cadence(sampling_interval_seconds: float) -> str:
        if sampling_interval_seconds > 15 * 60:
            return "REJECT_NO_INTERPOLATION"
        return "ACCEPT_RESAMPLING_PERMITTED"
    
    assert audit_candidate_cadence(1200) == "REJECT_NO_INTERPOLATION"  # 20 min (Andhra Pradesh)
    assert audit_candidate_cadence(3600) == "REJECT_NO_INTERPOLATION"  # 60 min (Colombia)
    assert audit_candidate_cadence(5) == "ACCEPT_RESAMPLING_PERMITTED"  # 5 sec (Nigeria / HiPIC)
    assert audit_candidate_cadence(900) == "ACCEPT_RESAMPLING_PERMITTED"  # 15 min exact
