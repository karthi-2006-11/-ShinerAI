import sys
sys.path.append(".")
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.external_validation import resample_external_to_15min_grid, build_config_c_features_and_target, CONFIG_C_FEATURES

model = joblib.load("models/xgboost_config_c.joblib")
if isinstance(model, dict) and "model" in model:
    model = model["model"]

raw_path = Path("data/external/oman_tilapia/data/raw/live/raw_readings_new.csv")
df_raw = pd.read_csv(raw_path)

print(f"Total raw rows: {len(df_raw)}")
zero_count = (df_raw["do_mgL"] <= 0.0).sum()
print(f"Zero / negative DO readings in raw data: {zero_count}")
zero_rows = df_raw[df_raw["do_mgL"] <= 0.0]
print(zero_rows[["timestamp", "do_mgL", "temp_c", "ph"]])

# RUN 1: WITH QC ZERO FILTERING (df.loc[df['do_mgL'] <= 0] = np.nan)
df_15m_filtered = resample_external_to_15min_grid(df_raw, "timestamp", "do_mgL")
df_eval_filtered = build_config_c_features_and_target(df_15m_filtered, "dt", "current_do")
df_valid_filtered = df_eval_filtered[df_eval_filtered["eligible"]].copy()
X_filt = df_valid_filtered[CONFIG_C_FEATURES].values
probs_filt = model.predict_proba(X_filt)[:, 1]
preds_filt = (probs_filt >= 0.50).astype(int)

print("\n--- RUN 1 (WITH QC ZERO FILTERING) ---")
print(f"Eligible samples: {len(df_valid_filtered)}")
print(f"Predictions >= 0.50 (AT_RISK): {preds_filt.sum()}")
print(f"Predictions < 0.50 (SAFE): {(preds_filt == 0).sum()}")
print(f"TN: {(preds_filt == 0).sum()}, FP: {preds_filt.sum()}")

# RUN 2: UNFILTERED (NO QC ZERO FILTERING)
df_unfilt = df_raw.copy()
df_unfilt["dt"] = pd.to_datetime(df_unfilt["timestamp"])
df_unfilt = df_unfilt.sort_values("dt").drop_duplicates(subset=["dt"])
resampled_unfilt = df_unfilt.set_index("dt").resample("15min", closed="right", label="right").agg(
    {"do_mgL": ["count", "mean"]}
)
resampled_unfilt.columns = ["reading_count", "current_do"]
resampled_unfilt.loc[resampled_unfilt["reading_count"] == 0, "current_do"] = np.nan
df_15m_unfilt = resampled_unfilt.reset_index()

df_eval_unfilt = build_config_c_features_and_target(df_15m_unfilt, "dt", "current_do")
df_valid_unfilt = df_eval_unfilt[df_eval_unfilt["eligible"]].copy()
X_unfilt = df_valid_unfilt[CONFIG_C_FEATURES].values
probs_unfilt = model.predict_proba(X_unfilt)[:, 1]
preds_unfilt = (probs_unfilt >= 0.50).astype(int)

print("\n--- RUN 2 (WITHOUT QC ZERO FILTERING / UNFILTERED) ---")
print(f"Eligible samples: {len(df_valid_unfilt)}")
print(f"Predictions >= 0.50 (AT_RISK): {preds_unfilt.sum()}")
print(f"Predictions < 0.50 (SAFE): {(preds_unfilt == 0).sum()}")
print(f"TN: {(preds_unfilt == 0).sum()}, FP: {preds_unfilt.sum()}")

if preds_unfilt.sum() > 0:
    print("\nFalse alarm timestamps in unfiltered run:")
    fp_indices = np.where(preds_unfilt == 1)[0]
    for idx in fp_indices:
        row = df_valid_unfilt.iloc[idx]
        print(f"  Time: {row['dt']}, current_do: {row['current_do']:.2f}, prob: {probs_unfilt[idx]:.4f}")
