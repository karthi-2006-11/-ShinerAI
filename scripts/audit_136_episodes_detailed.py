import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import numpy as np
import joblib

from src.model_utils import create_temporal_split, CONFIG_C_FEATURES
from src.operational_evaluation import extract_hypoxia_episodes

# 1. Load data & model
ml_df = pd.read_csv("data/processed/ml_ready_dataset.csv")
cleaned_df = pd.read_csv("data/processed/cleaned_pond_data.csv")
cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["Timestamp"])
model = joblib.load("models/xgboost_config_c.joblib")

_, test_df, _, _ = create_temporal_split(ml_df, train_ratio=0.80, purge_hours=2.0)
X_test = test_df[CONFIG_C_FEATURES].values
probs = model.predict_proba(X_test)[:, 1]

test_eval = test_df.copy()
test_eval["prob"] = probs
test_eval["pred"] = (probs >= 0.50).astype(int)
test_eval["prediction_timestamp"] = pd.to_datetime(test_eval["prediction_timestamp"])

episodes = extract_hypoxia_episodes(test_eval)

print(f"Total impending episodes extracted: {len(episodes)}")
detected_existing = (episodes["detected"] == 1).sum()
print(f"Detected in existing code: {detected_existing} / {len(episodes)} ({detected_existing / len(episodes):.4%})")

# Now inspect physical DO readings for each of the 136 episodes
episode_audit_rows = []

for idx, ep in episodes.iterrows():
    pid = ep["pond_id"]
    ep_id = ep["episode_id"]
    start_t = ep["start_time"]
    end_t = ep["end_time"]
    
    # Get all cleaned telemetry for this pond in [start_t, end_t + 2h]
    p_clean = cleaned_df[(cleaned_df["pond_id"] == pid) & 
                         (cleaned_df["timestamp"] >= start_t) & 
                         (cleaned_df["timestamp"] <= end_t + pd.Timedelta(hours=2))].sort_values("timestamp")
    
    # When is the FIRST physical DO < 3.0 mg/L in this entire window?
    low_do_rows = p_clean[p_clean["do_mg_l"] < 3.0]
    
    if len(low_do_rows) > 0:
        first_physical_crossing = low_do_rows.iloc[0]["timestamp"]
        first_physical_do = float(low_do_rows.iloc[0]["do_mg_l"])
    else:
        first_physical_crossing = None
        first_physical_do = None
        
    # Check predictions that occurred strictly BEFORE first_physical_crossing
    # and within 120 minutes of first_physical_crossing
    if first_physical_crossing is not None:
        pre_preds = test_eval[(test_eval["pond_id"] == pid) & 
                              (test_eval["prediction_timestamp"] < first_physical_crossing) & 
                              (test_eval["prediction_timestamp"] >= first_physical_crossing - pd.Timedelta(minutes=120))]
        pre_alerts = pre_preds[pre_preds["pred"] == 1]
        
        if len(pre_alerts) > 0:
            first_pre_alert = pre_alerts["prediction_timestamp"].min()
            true_lead_time = (first_physical_crossing - first_pre_alert).total_seconds() / 60.0
            detected_before_onset = True
        else:
            first_pre_alert = None
            true_lead_time = 0.0
            detected_before_onset = False
    else:
        first_pre_alert = None
        true_lead_time = 0.0
        detected_before_onset = False
        
    episode_audit_rows.append({
        "episode_id": ep_id,
        "pond_id": pid,
        "start_time": start_t,
        "end_time": end_t,
        "interval_count": ep["interval_count"],
        "existing_assumed_crossing": ep["actual_crossing_time"],
        "first_physical_crossing": first_physical_crossing,
        "first_physical_do": first_physical_do,
        "existing_detected": ep["detected"],
        "existing_lead_time": ep["lead_time_minutes"],
        "detected_before_physical_onset": int(detected_before_onset),
        "true_lead_time_minutes": true_lead_time,
        "first_pre_alert_time": first_pre_alert,
    })

df_ep_audit = pd.DataFrame(episode_audit_rows)
print("\n--- COMPARISON ON THE 136 EPISODES ---")
print("Existing assumed crossing vs. First physical crossing:")
same_crossing = (df_ep_audit["existing_assumed_crossing"] == df_ep_audit["first_physical_crossing"]).sum()
diff_crossing = (df_ep_audit["existing_assumed_crossing"] != df_ep_audit["first_physical_crossing"]).sum()
print(f"Exact same timestamp: {same_crossing} / 136")
print(f"Different timestamp:   {diff_crossing} / 136")

print("\nWhere they differ:")
diff_df = df_ep_audit[df_ep_audit["existing_assumed_crossing"] != df_ep_audit["first_physical_crossing"]]
print(f"Number of episodes where first physical crossing was EARLIER than existing assumed crossing: {(diff_df['first_physical_crossing'] < diff_df['existing_assumed_crossing']).sum()}")
for _, r in diff_df.head(10).iterrows():
    print(f"{r['episode_id']} ({r['pond_id']}): intervals={r['interval_count']}, start={r['start_time']}, end={r['end_time']}, existing_cross={r['existing_assumed_crossing']}, first_phys={r['first_physical_crossing']}")

print("\nDetection rate comparison on these 136 episodes:")
print(f"Existing detected:                 {df_ep_audit['existing_detected'].sum()} / 136 ({df_ep_audit['existing_detected'].mean():.2%})")
print(f"Detected BEFORE physical onset:    {df_ep_audit['detected_before_physical_onset'].sum()} / 136 ({df_ep_audit['detected_before_physical_onset'].mean():.2%})")

det_true = df_ep_audit[df_ep_audit["detected_before_physical_onset"] == 1]
print(f"\nLead time comparison for detected episodes:")
print(f"Existing lead time:  mean = {episodes[episodes['detected']==1]['lead_time_minutes'].mean():.1f} min, median = {episodes[episodes['detected']==1]['lead_time_minutes'].median():.1f} min, max = {episodes[episodes['detected']==1]['lead_time_minutes'].max():.1f} min")
print(f"True physical lead:  mean = {det_true['true_lead_time_minutes'].mean():.1f} min, median = {det_true['true_lead_time_minutes'].median():.1f} min, max = {det_true['true_lead_time_minutes'].max():.1f} min")
