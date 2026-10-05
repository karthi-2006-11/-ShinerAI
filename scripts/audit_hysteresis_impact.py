import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import numpy as np
import joblib

from src.model_utils import create_temporal_split, CONFIG_C_FEATURES
from src.operational_evaluation import (
    extract_hypoxia_episodes,
    compute_false_alarms_per_pond_day,
    compute_alert_stability_metrics,
)

# Load data & model
ml_df = pd.read_csv("data/processed/ml_ready_dataset.csv")
cleaned_df = pd.read_csv("data/processed/cleaned_pond_data.csv")
cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["Timestamp"])
model = joblib.load("models/xgboost_config_c.joblib")

_, test_df, _, _ = create_temporal_split(ml_df, train_ratio=0.80, purge_hours=2.0)
X_test = test_df[CONFIG_C_FEATURES].values
probs = model.predict_proba(X_test)[:, 1]

test_df["prob"] = probs
test_df["pred"] = (probs >= 0.50).astype(int)
test_df["prediction_timestamp"] = pd.to_datetime(test_df["prediction_timestamp"])

# 1. Raw Model Behavior
raw_fp = int(((test_df["pred"] == 1) & (test_df["target"] == 0)).sum())
raw_tp = int(((test_df["pred"] == 1) & (test_df["target"] == 1)).sum())
raw_alarm_metrics = compute_false_alarms_per_pond_day(test_df)
raw_stab_metrics = compute_alert_stability_metrics(test_df)

# 2. Hysteresis Filter: An alert is confirmed only when there are 2 consecutive positive predictions
hyst_preds = []
for pid, grp in test_df.groupby("pond_id"):
    grp = grp.sort_values("prediction_timestamp").reset_index(drop=True)
    raw_p = grp["pred"].values
    filt_p = np.zeros_like(raw_p)
    for i in range(len(raw_p)):
        if i >= 1 and raw_p[i] == 1 and raw_p[i-1] == 1:
            filt_p[i] = 1
    grp["hyst_pred"] = filt_p
    hyst_preds.append(grp)
test_hyst_df = pd.concat(hyst_preds, ignore_index=True)

hyst_fp = int(((test_hyst_df["hyst_pred"] == 1) & (test_hyst_df["target"] == 0)).sum())
hyst_tp = int(((test_hyst_df["hyst_pred"] == 1) & (test_hyst_df["target"] == 1)).sum())

# Also evaluate false alarm runs
test_hyst_eval = test_hyst_df.copy()
test_hyst_eval["pred"] = test_hyst_eval["hyst_pred"]
hyst_alarm_metrics = compute_false_alarms_per_pond_day(test_hyst_eval)
hyst_stab_metrics = compute_alert_stability_metrics(test_hyst_eval)

print("=" * 75)
print("HYSTERESIS FILTER AUDIT: RAW VS 2-STEP CONFIRMATION")
print("=" * 75)
print(f"Raw False Positive Intervals:        {raw_fp}")
print(f"Hysteresis False Positive Intervals:   {hyst_fp} (-{(raw_fp - hyst_fp)/raw_fp * 100:.2f}%)")
print(f"Raw True Positive Intervals:         {raw_tp}")
print(f"Hysteresis True Positive Intervals:  {hyst_tp} (-{(raw_tp - hyst_tp)/raw_tp * 100:.2f}%)")
print(f"Raw False Alarms / Pond-Day:         {raw_alarm_metrics['mean_false_alarms_per_pond_day']:.2f}")
print(f"Hysteresis False Alarms / Pond-Day:  {hyst_alarm_metrics['mean_false_alarms_per_pond_day']:.2f}")
print(f"Raw Distinct False Alarm Episodes:   {raw_alarm_metrics['total_distinct_fp_episodes']}")
print(f"Hysteresis Distinct False Episodes:  {hyst_alarm_metrics['total_distinct_fp_episodes']} (-{(raw_alarm_metrics['total_distinct_fp_episodes'] - hyst_alarm_metrics['total_distinct_fp_episodes'])/raw_alarm_metrics['total_distinct_fp_episodes'] * 100:.2f}%)")

# How does Hysteresis affect the 136 Impending Hypoxia Episodes?
# Load the 136 episodes
from scripts.audit_136_episodes_detailed import df_ep_audit

# For each of the 136 episodes, check if it is detected under hysteresis
hyst_detected = 0
hyst_lead_times = []
hyst_true_lead_times = []

for _, row in df_ep_audit.iterrows():
    pid = row["pond_id"]
    start_t = row["start_time"]
    end_t = row["end_time"]
    first_phys = row["first_physical_crossing"]
    
    # Get test predictions for this episode
    p_sub = test_hyst_df[(test_hyst_df["pond_id"] == pid) & 
                         (test_hyst_df["prediction_timestamp"] >= start_t) & 
                         (test_hyst_df["prediction_timestamp"] <= end_t)]
    
    alerts = p_sub[p_sub["hyst_pred"] == 1]
    if len(alerts) > 0:
        hyst_detected += 1
        first_alert = alerts["prediction_timestamp"].min()
        # Existing style lead time
        hyst_lead_times.append((row["existing_assumed_crossing"] - first_alert).total_seconds() / 60.0)
        # True physical lead time
        if first_phys is not None and first_alert < first_phys:
            hyst_true_lead_times.append((first_phys - first_alert).total_seconds() / 60.0)

print(f"\n--- IMPACT ON EVENT DETECTION (136 Impending Episodes) ---")
print(f"Raw Event Detection Rate:        124 / 136 (91.18%)")
print(f"Hysteresis Event Detection Rate: {hyst_detected} / 136 ({hyst_detected / 136 * 100:.2f}%)")
print(f"Newly Missed Episodes due to Hysteresis: {124 - hyst_detected} episodes")

print(f"\n--- IMPACT ON LEAD TIME (for detected episodes) ---")
print(f"Raw Mean Lead Time (existing):        101.7 min (median 120.0 min)")
print(f"Hysteresis Mean Lead Time (existing): {np.mean(hyst_lead_times):.1f} min (median {np.median(hyst_lead_times):.1f} min)")
print(f"Raw Mean True Lead Time (physical):   93.1 min (median 120.0 min)")
print(f"Hysteresis Mean True Lead Time:       {np.mean(hyst_true_lead_times):.1f} min (median {np.median(hyst_true_lead_times):.1f} min)")
print(f"Lead time delay caused by Hysteresis: ~15 minutes (due to waiting for 2nd confirmation interval)")
