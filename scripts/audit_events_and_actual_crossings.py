"""
Comprehensive script to audit event definitions:
1. Existing methodology (blocks of target=1 in test_df)
2. True physical DO < 3.0 mg/L crossing events from continuous cleaned telemetry (cleaned_pond_data.csv)
3. Matching pre-event alerts within the 2-hour forecasting contract
4. Lead time calculations (strictly pre-onset)
5. Hysteresis filter impact on false alarms, event detection, and lead time
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.model_utils import create_temporal_split, CONFIG_C_FEATURES
from src.operational_evaluation import extract_hypoxia_episodes, compute_event_level_metrics

def main():
    print("=" * 80)
    print("AUDITING EVENT-LEVEL EARLY WARNING AND PHYSICAL LOW-DO CROSSINGS")
    print("=" * 80)

    # 1. Load ML Dataset and Model
    ml_path = Path("data/processed/ml_ready_dataset.csv")
    cleaned_path = Path("data/processed/cleaned_pond_data.csv")
    model_path = Path("models/xgboost_config_c.joblib")

    ml_df = pd.read_csv(ml_path)
    train_df, test_df, purged_df, accounting_df = create_temporal_split(ml_df, train_ratio=0.80, purge_hours=2.0)
    model = joblib.load(model_path)

    # Predictions
    X_test = test_df[CONFIG_C_FEATURES].values
    probs = model.predict_proba(X_test)[:, 1]
    test_eval_df = test_df.copy()
    test_eval_df["prob"] = probs
    test_eval_df["pred"] = (probs >= 0.50).astype(int)
    test_eval_df["prediction_timestamp"] = pd.to_datetime(test_eval_df["prediction_timestamp"])

    # 2. Existing Methodology: Contiguous target=1 blocks in test_df
    print("\n--- 1. EXISTING METHODOLOGY AUDIT (target=1 blocks in test_df) ---")
    df_episodes_existing = extract_hypoxia_episodes(test_eval_df)
    existing_metrics = compute_event_level_metrics(df_episodes_existing)
    print(f"Total impending episodes: {existing_metrics['total_hypoxia_episodes']}")
    print(f"Detected: {existing_metrics['detected_episodes']} ({existing_metrics['event_detection_rate']:.2%})")
    print(f"Missed: {existing_metrics['missed_episodes']}")
    print(f"Mean lead time: {existing_metrics['mean_lead_time_minutes']} min")
    print(f"Median lead time: {existing_metrics['median_lead_time_minutes']} min")
    print(f"Min / Max lead time: {existing_metrics['min_lead_time_minutes']} / {existing_metrics['max_lead_time_minutes']} min")

    # 3. Load continuous cleaned pond data to find actual physical crossings
    print("\n--- 2. PHYSICAL LOW-DO CROSSING AUDIT (cleaned_pond_data.csv) ---")
    cleaned_df = pd.read_csv(cleaned_path)
    cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["Timestamp"])
    cleaned_df = cleaned_df[cleaned_df["is_usable"] == True].sort_values(["pond_id", "timestamp"]).reset_index(drop=True)

    # For each pond, get the test start timestamp from accounting_df or test_df
    # In temporal split, test starts at t_cutoff
    test_time_bounds = {}
    for pid, grp in test_eval_df.groupby("pond_id"):
        test_time_bounds[pid] = (grp["prediction_timestamp"].min(), grp["prediction_timestamp"].max())

    print(f"Loaded {len(cleaned_df)} usable observations from cleaned_pond_data.csv across {cleaned_df['pond_id'].nunique()} ponds.")

    # Identify physical low-DO episodes across all ponds in the test partition
    # An episode is a contiguous sequence where DO < 3.0 mg/L (with gaps <= 30 min)
    # preceded by DO >= 3.0 mg/L
    physical_episodes = []
    
    for pid, grp in cleaned_df.groupby("pond_id"):
        if pid not in test_time_bounds:
            continue
        test_start, test_end = test_time_bounds[pid]
        # Restrict to observations on or after test_start - 2 hours (so we can see events starting during test period)
        # Up to test_end + 2 hours
        grp = grp[(grp["timestamp"] >= test_start - pd.Timedelta(hours=2)) & 
                  (grp["timestamp"] <= test_end + pd.Timedelta(hours=2))].sort_values("timestamp").reset_index(drop=True)
        
        # Identify episodes of DO < 3.0
        grp["is_low_do"] = grp["do_mg_l"] < 3.0
        time_diffs = grp["timestamp"].diff()
        gap_mask = time_diffs > pd.Timedelta(minutes=30)
        state_diff = grp["is_low_do"].diff().ne(0)
        blocks = (gap_mask | state_diff).cumsum()

        for b_id, b_grp in grp.groupby(blocks):
            if b_grp["is_low_do"].iloc[0]: # This is a low-DO episode
                ev_start = b_grp["timestamp"].min()
                ev_end = b_grp["timestamp"].max()
                
                # Check if this episode is within the test evaluation window:
                # An episode is relevant to the test set if it occurs when test predictions could forecast it:
                # i.e., ev_start >= test_start and ev_start <= test_end + 2h
                if ev_start >= test_start and ev_start <= test_end + pd.Timedelta(hours=2):
                    min_do = float(b_grp["do_mg_l"].min())
                    duration_minutes = (ev_end - ev_start).total_seconds() / 60.0 + 15.0 # duration including interval
                    
                    physical_episodes.append({
                        "pond_id": pid,
                        "actual_event_start": ev_start,
                        "actual_event_end": ev_end,
                        "duration_minutes": duration_minutes,
                        "minimum_DO": min_do,
                        "num_points": len(b_grp)
                    })

    df_phys = pd.DataFrame(physical_episodes)
    print(f"\nFound {len(df_phys)} actual physical low-DO episodes across test monitoring windows.")
    
    # 4. Now evaluate TRUE ADVANCE WARNING for each physical episode:
    # A prediction at T is a valid pre-event prediction if:
    # 1. T is in test_eval_df for that pond
    # 2. T < actual_event_start
    # 3. T >= actual_event_start - 120 minutes (within 2-hour contract)
    # An alert at T counts as TRUE ADVANCE WARNING if pred == 1
    
    event_results = []
    for idx, ep in df_phys.iterrows():
        pid = ep["pond_id"]
        ev_start = ep["actual_event_start"]
        ev_end = ep["actual_event_end"]
        
        # Find test predictions in [ev_start - 120m, ev_start)
        pre_preds = test_eval_df[(test_eval_df["pond_id"] == pid) & 
                                 (test_eval_df["prediction_timestamp"] < ev_start) & 
                                 (test_eval_df["prediction_timestamp"] >= ev_start - pd.Timedelta(minutes=120))]
        
        pre_alerts = pre_preds[pre_preds["pred"] == 1]
        
        if len(pre_alerts) > 0:
            first_alert_time = pre_alerts["prediction_timestamp"].min()
            lead_time = (ev_start - first_alert_time).total_seconds() / 60.0
            detected = True
            missed = False
        else:
            first_alert_time = None
            lead_time = 0.0
            detected = False
            missed = True
            
        event_results.append({
            "event_id": f"{pid}_phys_{idx+1:03d}",
            "pond_id": pid,
            "actual_event_start": ev_start,
            "actual_event_end": ev_end,
            "first_pre_event_alert": first_alert_time,
            "lead_time_minutes": lead_time,
            "detected_before_event": detected,
            "missed": missed,
            "minimum_DO": ep["minimum_DO"],
            "event_duration": ep["duration_minutes"],
            "pre_event_prediction_points": len(pre_preds),
        })

    df_res = pd.DataFrame(event_results)
    
    # Analyze results
    print("\n--- 3. TRUE PHYSICAL ADVANCE WARNING METRICS ---")
    tot_events = len(df_res)
    detected_count = df_res["detected_before_event"].sum()
    missed_count = df_res["missed"].sum()
    det_rate = detected_count / tot_events if tot_events > 0 else 0
    miss_rate = missed_count / tot_events if tot_events > 0 else 0
    
    detected_events = df_res[df_res["detected_before_event"] == True]
    lead_times = detected_events["lead_time_minutes"].values
    
    print(f"Total physical low-DO events: {tot_events}")
    print(f"Detected before event onset:  {detected_count} ({det_rate:.2%})")
    print(f"Missed:                       {missed_count} ({miss_rate:.2%})")
    
    if len(lead_times) > 0:
        mean_lt = np.mean(lead_times)
        med_lt = np.median(lead_times)
        q25_lt = np.percentile(lead_times, 25)
        q75_lt = np.percentile(lead_times, 75)
        min_lt = np.min(lead_times)
        max_lt = np.max(lead_times)
        
        p30 = (lead_times >= 30).mean() * 100
        p60 = (lead_times >= 60).mean() * 100
        p90 = (lead_times >= 90).mean() * 100
        p120 = (lead_times >= 120).mean() * 100
        
        print(f"Mean lead time:   {mean_lt:.1f} minutes")
        print(f"Median lead time: {med_lt:.1f} minutes")
        print(f"25th percentile:  {q25_lt:.1f} minutes")
        print(f"75th percentile:  {q75_lt:.1f} minutes")
        print(f"Min / Max:        {min_lt:.1f} / {max_lt:.1f} minutes")
        print(f"Lead time >= 30 min:  {p30:.1f}%")
        print(f"Lead time >= 60 min:  {p60:.1f}%")
        print(f"Lead time >= 90 min:  {p90:.1f}%")
        print(f"Lead time >= 120 min: {p120:.1f}%")

    # Save detailed table
    df_res.to_csv("results/reports/physical_hypoxia_events_audit.csv", index=False)
    print("\nSaved results/reports/physical_hypoxia_events_audit.csv")

    # 5. Hysteresis Audit on Physical Events
    print("\n--- 4. HYSTERESIS FILTER AUDIT ---")
    # Apply 2-step hysteresis: an alert requires 2 consecutive pred=1
    hyst_preds = []
    for pid, grp in test_eval_df.groupby("pond_id"):
        grp = grp.sort_values("prediction_timestamp").reset_index(drop=True)
        raw_p = grp["pred"].values
        filt_p = np.zeros_like(raw_p)
        for i in range(len(raw_p)):
            if i >= 1 and raw_p[i] == 1 and raw_p[i-1] == 1:
                filt_p[i] = 1
        grp["hyst_pred"] = filt_p
        hyst_preds.append(grp)
    test_hyst_df = pd.concat(hyst_preds, ignore_index=True)
    
    # Hysteresis on physical events
    hyst_det_count = 0
    hyst_lead_times = []
    for idx, ep in df_phys.iterrows():
        pid = ep["pond_id"]
        ev_start = ep["actual_event_start"]
        pre_preds = test_hyst_df[(test_hyst_df["pond_id"] == pid) & 
                                 (test_hyst_df["prediction_timestamp"] < ev_start) & 
                                 (test_hyst_df["prediction_timestamp"] >= ev_start - pd.Timedelta(minutes=120))]
        pre_alerts = pre_preds[pre_preds["hyst_pred"] == 1]
        if len(pre_alerts) > 0:
            hyst_det_count += 1
            first_alert_time = pre_alerts["prediction_timestamp"].min()
            hyst_lead_times.append((ev_start - first_alert_time).total_seconds() / 60.0)
            
    print(f"Raw detection rate:        {detected_count} / {tot_events} ({det_rate:.2%})")
    print(f"Hysteresis detection rate: {hyst_det_count} / {tot_events} ({hyst_det_count/tot_events:.2%})")
    if len(hyst_lead_times) > 0:
        print(f"Raw mean lead time:        {mean_lt:.1f} min (median {med_lt:.1f} min)")
        print(f"Hysteresis mean lead time: {np.mean(hyst_lead_times):.1f} min (median {np.median(hyst_lead_times):.1f} min)")

    # False positive reduction
    raw_fp = ((test_eval_df["pred"] == 1) & (test_eval_df["target"] == 0)).sum()
    hyst_fp = ((test_hyst_df["hyst_pred"] == 1) & (test_hyst_df["target"] == 0)).sum()
    fp_red_pct = (raw_fp - hyst_fp) / raw_fp * 100.0
    print(f"Raw FP: {raw_fp} -> Hysteresis FP: {hyst_fp} (-{fp_red_pct:.1f}%)")

if __name__ == '__main__':
    main()
