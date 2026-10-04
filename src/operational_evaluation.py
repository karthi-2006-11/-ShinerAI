"""
ShinerAI: Operational & Event-Level Evaluation Module
Evaluates real-world farm decision support metrics on test set predictions:
1. Event-Level Hypoxia Grouping (contiguous low-DO episodes)
2. Warning Lead Time Distribution (minutes of advance notice before DO < 3.0 mg/L)
3. Missed Event Rate vs. Event Detection Rate
4. False Alarms Per Pond Per Day (labor and alert fatigue metric)
5. Alert Stability & Dynamics (0->1 turn-ons, single-interval chattering pulses, alert duration)
6. Operational Hysteresis Filter Analysis (2-consecutive alert requirement)
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def extract_hypoxia_episodes(
    test_df: pd.DataFrame,
    max_inter_event_gap_minutes: float = 30.0,
) -> pd.DataFrame:
    """
    Groups contiguous rows with target == 1 into distinct impending hypoxia episodes.
    An episode represents an operational crisis window leading to hypoxia.
    """
    df = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["prediction_timestamp"]):
        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
    
    episodes = []
    
    for pid, grp in df.groupby("pond_id"):
        grp = grp.sort_values("prediction_timestamp").reset_index(drop=True)
        time_diffs = grp["prediction_timestamp"].diff()
        gap_mask = time_diffs > pd.Timedelta(minutes=max_inter_event_gap_minutes)
        target_diff = grp["target"].diff().ne(0)
        blocks = (gap_mask | target_diff).cumsum()
        
        for b_id, b_grp in grp.groupby(blocks):
            if b_grp["target"].iloc[0] == 1:
                has_alert = (b_grp["pred"] == 1).any()
                alert_rows = b_grp[b_grp["pred"] == 1]
                first_alert_time = alert_rows["prediction_timestamp"].min() if has_alert else None
                
                # In our 2-hour lookahead dataset, each observation at T was labeled target=1
                # because DO drops < 3.0 mg/L within [T+15m, T+120m].
                # The end of the impending episode is the last valid prediction point before DO drops < 3.0.
                # Therefore, the actual hypoxia crossing occurs at episode end_time + 15 min.
                actual_hypoxia_crossing = b_grp["prediction_timestamp"].max() + pd.Timedelta(minutes=15)
                
                lead_time_minutes = (
                    (actual_hypoxia_crossing - first_alert_time).total_seconds() / 60.0
                    if has_alert else 0.0
                )
                
                episodes.append({
                    "pond_id": pid,
                    "episode_id": f"{pid}_ep_{len(episodes)+1:03d}",
                    "start_time": b_grp["prediction_timestamp"].min(),
                    "end_time": b_grp["prediction_timestamp"].max(),
                    "actual_crossing_time": actual_hypoxia_crossing,
                    "interval_count": len(b_grp),
                    "min_current_do": float(b_grp["current_do"].min()),
                    "mean_risk_prob": float(b_grp["prob"].mean()) if "prob" in b_grp else 0.0,
                    "total_alerts": int((b_grp["pred"] == 1).sum()),
                    "detected": int(has_alert),
                    "first_alert_time": first_alert_time,
                    "lead_time_minutes": float(lead_time_minutes),
                })
                
    return pd.DataFrame(episodes)


def compute_event_level_metrics(df_episodes: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary event-level detection metrics across all episodes.
    """
    total_episodes = len(df_episodes)
    if total_episodes == 0:
        return {}
    
    detected = df_episodes[df_episodes["detected"] == 1]
    detected_count = len(detected)
    missed_count = total_episodes - detected_count
    
    detection_rate = detected_count / total_episodes
    miss_rate = missed_count / total_episodes
    
    lead_times = detected["lead_time_minutes"].values
    
    return {
        "total_hypoxia_episodes": total_episodes,
        "detected_episodes": detected_count,
        "missed_episodes": missed_count,
        "event_detection_rate": round(float(detection_rate), 4),
        "event_miss_rate": round(float(miss_rate), 4),
        "mean_lead_time_minutes": round(float(np.mean(lead_times)), 1) if len(lead_times) > 0 else 0.0,
        "median_lead_time_minutes": round(float(np.median(lead_times)), 1) if len(lead_times) > 0 else 0.0,
        "min_lead_time_minutes": round(float(np.min(lead_times)), 1) if len(lead_times) > 0 else 0.0,
        "max_lead_time_minutes": round(float(np.max(lead_times)), 1) if len(lead_times) > 0 else 0.0,
    }


def compute_false_alarms_per_pond_day(test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates false alarms per pond per day based on actual test monitoring exposure time.
    """
    df = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["prediction_timestamp"]):
        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
        
    records = []
    total_fp = 0
    total_pond_days = 0.0
    
    for pid, grp in df.groupby("pond_id"):
        span_seconds = (grp["prediction_timestamp"].max() - grp["prediction_timestamp"].min()).total_seconds()
        span_days = span_seconds / 86400.0
        fp = int(((grp["pred"] == 1) & (grp["target"] == 0)).sum())
        total_fp += fp
        total_pond_days += span_days
        
        # Count distinct false alarm contiguous episodes (clusters of false alarms)
        grp = grp.sort_values("prediction_timestamp").reset_index(drop=True)
        fp_mask = (grp["pred"] == 1) & (grp["target"] == 0)
        fp_diff = fp_mask.astype(int).diff().ne(0)
        fp_blocks = fp_diff.cumsum()
        distinct_fp_episodes = sum(1 for _, b in grp.groupby(fp_blocks) if ((b["pred"] == 1) & (b["target"] == 0)).all())
        
        records.append({
            "pond_id": pid,
            "span_days": round(span_days, 2),
            "false_positives": fp,
            "fp_per_day": round(fp / span_days, 2) if span_days > 0 else 0.0,
            "distinct_fp_episodes": distinct_fp_episodes,
            "fp_episodes_per_day": round(distinct_fp_episodes / span_days, 2) if span_days > 0 else 0.0,
        })
        
    df_pond = pd.DataFrame(records)
    
    return {
        "total_test_ponds": len(df_pond),
        "total_monitoring_pond_days": round(total_pond_days, 2),
        "total_false_positive_intervals": total_fp,
        "mean_false_alarms_per_pond_day": round(total_fp / total_pond_days, 2) if total_pond_days > 0 else 0.0,
        "median_false_alarms_per_pond_day": round(float(df_pond["fp_per_day"].median()), 2),
        "min_false_alarms_per_pond_day": round(float(df_pond["fp_per_day"].min()), 2),
        "max_false_alarms_per_pond_day": round(float(df_pond["fp_per_day"].max()), 2),
        "total_distinct_fp_episodes": int(df_pond["distinct_fp_episodes"].sum()),
        "mean_fp_episodes_per_pond_day": round(float(df_pond["distinct_fp_episodes"].sum() / total_pond_days), 2) if total_pond_days > 0 else 0.0,
        "per_pond_summary": df_pond.to_dict(orient="records"),
    }


def compute_alert_stability_metrics(test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Measures alert chattering, activation frequency, and alert run duration.
    """
    df = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["prediction_timestamp"]):
        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
        
    total_activations = 0 # 0 -> 1 transitions
    total_deactivations = 0 # 1 -> 0 transitions
    single_step_pulses = 0 # 0 -> 1 -> 0 pulses (chattering)
    alert_run_lengths = []
    
    for _, grp in df.groupby("pond_id"):
        grp = grp.sort_values("prediction_timestamp").reset_index(drop=True)
        preds = grp["pred"].values
        diffs = np.diff(preds)
        
        total_activations += int(np.sum(diffs == 1))
        total_deactivations += int(np.sum(diffs == -1))
        
        for i in range(1, len(preds) - 1):
            if preds[i-1] == 0 and preds[i] == 1 and preds[i+1] == 0:
                single_step_pulses += 1
                
        current_run = 0
        for p in preds:
            if p == 1:
                current_run += 1
            elif current_run > 0:
                alert_run_lengths.append(current_run)
                current_run = 0
        if current_run > 0:
            alert_run_lengths.append(current_run)
            
    chattering_rate = (single_step_pulses / total_activations) if total_activations > 0 else 0.0
    mean_duration_steps = np.mean(alert_run_lengths) if alert_run_lengths else 0.0
    mean_duration_minutes = mean_duration_steps * 15.0
    
    return {
        "total_test_intervals": len(df),
        "total_alert_activations": total_activations,
        "total_alert_deactivations": total_deactivations,
        "single_step_pulses": single_step_pulses,
        "chattering_rate": round(float(chattering_rate), 4),
        "total_contiguous_alert_runs": len(alert_run_lengths),
        "mean_alert_duration_intervals": round(float(mean_duration_steps), 2),
        "mean_alert_duration_minutes": round(float(mean_duration_minutes), 1),
        "median_alert_duration_minutes": round(float(np.median(alert_run_lengths) * 15.0), 1) if alert_run_lengths else 0.0,
    }


def evaluate_hysteresis_filter(
    test_df: pd.DataFrame,
    consecutive_steps_required: int = 2,
) -> Dict[str, Any]:
    """
    Evaluates an operational post-processing hysteresis filter:
    An alert is confirmed only if predicted AT_RISK for >= consecutive_steps_required intervals.
    NOTE: Evaluated as a operational post-processing analysis; does NOT modify the frozen ML model.
    """
    df = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["prediction_timestamp"]):
        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
        
    filtered_preds = []
    for _, grp in df.groupby("pond_id"):
        grp = grp.sort_values("prediction_timestamp").reset_index(drop=True)
        raw_p = grp["pred"].values
        filt_p = np.zeros_like(raw_p)
        
        for i in range(len(raw_p)):
            if i >= consecutive_steps_required - 1:
                # Active if last K steps are all 1
                if all(raw_p[i - k] == 1 for k in range(consecutive_steps_required)):
                    filt_p[i] = 1
            else:
                filt_p[i] = raw_p[i]
        filtered_preds.extend(filt_p)
        
    df["filtered_pred"] = filtered_preds
    y_true = df["target"].values
    y_pred = df["filtered_pred"].values
    
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "consecutive_steps_required": consecutive_steps_required,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "specificity": round(spec, 4),
        "fp_reduction": int(698 - fp), # vs raw 698
        "fp_reduction_percent": round((698 - fp) / 698.0 * 100.0, 1),
    }
