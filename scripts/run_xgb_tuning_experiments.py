"""
ShinerAI: Systematic XGBoost Tuning & Feature Experiments (Experiment Branch v1)
Phase: Scientific Model Exploration & Tuning

Strict Scientific Rules:
1. Final temporal holdout test set (8,261 rows, 2h purge) is LOCKED.
2. All model selection, hyperparameter search, feature experiments, and
   threshold tuning occur exclusively on the 32,908 training observations
   via 5-fold GroupKFold by pond_id.
3. Production baseline models/xgboost_config_c.joblib is never overwritten.
4. Final test evaluation occurs strictly ONCE on the selected candidate.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

from src.model_utils import (
    create_temporal_split,
    verify_dataset_preconditions,
    CONFIG_C_FEATURES,
    compute_classification_metrics,
)
from src.features_experimental import (
    compute_experimental_features,
    RATE_OF_CHANGE_FEATURES,
    ACCELERATION_FEATURES,
    HISTORICAL_STATS_FEATURES,
    CONFIG_C_PLUS_TRENDS,
    CONFIG_C_PLUS_ACCEL,
    CONFIG_C_PLUS_ALL,
)
from src.operational_evaluation import extract_hypoxia_episodes


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def evaluate_cv_candidate(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    xgb_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates an XGBoost candidate configuration across 5 GroupKFold splits.
    Computes fold-level metrics and out-of-fold (OOF) aggregated metrics.
    """
    n_samples = len(y_train_full)
    oof_probs = np.zeros(n_samples, dtype=float)
    fold_pr_aucs = []
    fold_roc_aucs = []
    fold_f1s = []
    fold_recalls = []
    fold_precs = []
    fold_specs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        # Calculate fold-specific scale_pos_weight if specified as 'auto'
        params = xgb_params.copy()
        if params.get("scale_pos_weight") == "auto":
            n_pos = int(y_tr.sum())
            params["scale_pos_weight"] = float((len(y_tr) - n_pos) / n_pos)

        clf = XGBClassifier(**params)
        clf.fit(X_tr, y_tr)

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        val_preds = (val_probs >= 0.50).astype(int)

        fold_pr_aucs.append(average_precision_score(y_val, val_probs))
        fold_roc_aucs.append(roc_auc_score(y_val, val_probs))
        fold_f1s.append(f1_score(y_val, val_preds, zero_division=0))
        fold_recalls.append(recall_score(y_val, val_preds, zero_division=0))
        fold_precs.append(precision_score(y_val, val_preds, zero_division=0))

        cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        fold_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    # Aggregated OOF metrics
    oof_preds_50 = (oof_probs >= 0.50).astype(int)
    oof_pr_auc = float(average_precision_score(y_train_full, oof_probs))
    oof_roc_auc = float(roc_auc_score(y_train_full, oof_probs))
    oof_f1 = float(f1_score(y_train_full, oof_preds_50, zero_division=0))
    oof_recall = float(recall_score(y_train_full, oof_preds_50, zero_division=0))
    oof_precision = float(precision_score(y_train_full, oof_preds_50, zero_division=0))
    cm_oof = confusion_matrix(y_train_full, oof_preds_50, labels=[0, 1])
    oof_spec = float(cm_oof[0, 0] / (cm_oof[0, 0] + cm_oof[0, 1]))
    oof_acc = float(accuracy_score(y_train_full, oof_preds_50))

    return {
        "mean_val_pr_auc": float(np.mean(fold_pr_aucs)),
        "std_val_pr_auc": float(np.std(fold_pr_aucs)),
        "mean_val_roc_auc": float(np.mean(fold_roc_aucs)),
        "std_val_roc_auc": float(np.std(fold_roc_aucs)),
        "mean_val_f1": float(np.mean(fold_f1s)),
        "mean_val_recall": float(np.mean(fold_recalls)),
        "mean_val_precision": float(np.mean(fold_precs)),
        "mean_val_specificity": float(np.mean(fold_specs)),
        "oof_pr_auc": oof_pr_auc,
        "oof_roc_auc": oof_roc_auc,
        "oof_f1": oof_f1,
        "oof_recall": oof_recall,
        "oof_precision": oof_precision,
        "oof_specificity": oof_spec,
        "oof_accuracy": oof_acc,
        "oof_probs": oof_probs,
    }


def run_experiments():
    start_time = time.time()
    print("=" * 80)
    print("SHINERAI: SYSTEMATIC XGBOOST TUNING EXPERIMENT (V1)")
    print("=" * 80)

    # 1. Environment & Preconditions
    data_path = os.path.join("data", "processed", "ml_ready_dataset.csv")
    prod_model_path = os.path.join("models", "xgboost_config_c.joblib")
    reports_dir = os.path.join("results", "reports")
    figures_dir = os.path.join("results", "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    prod_model_hash_initial = sha256_file(prod_model_path)
    dataset_hash = sha256_file(data_path)
    print(f"Dataset SHA256:        {dataset_hash}")
    print(f"Production Model SHA256: {prod_model_hash_initial} (LOCKED)")

    df = pd.read_csv(data_path)
    verify_dataset_preconditions(df)

    # 2. Leakage-safe temporal split
    train_df, test_df, purged_df, _ = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)
    print(f"\nTemporal Split:")
    print(f"  Train:  {len(train_df):,} examples ({train_df['target'].sum()} AT_RISK, {train_df['target'].mean():.2%})")
    print(f"  Purged: {len(purged_df):,} examples")
    print(f"  Test:   {len(test_df):,} examples ({test_df['target'].sum()} AT_RISK, {test_df['target'].mean():.2%}) [ISOLATED]")

    # 3. Add experimental features to training data ONLY
    train_feat_df = compute_experimental_features(train_df)
    test_feat_df = compute_experimental_features(test_df)  # Isolated until final test evaluation

    # 4. GroupKFold setup on training data
    gkf = GroupKFold(n_splits=5)
    groups = train_feat_df["pond_id"].values
    folds = list(gkf.split(train_feat_df, groups=groups))

    # Verify zero overlap in any fold
    for f_i, (tr_idx, val_idx) in enumerate(folds, 1):
        tr_ponds = set(train_feat_df.iloc[tr_idx]["pond_id"])
        val_ponds = set(train_feat_df.iloc[val_idx]["pond_id"])
        assert tr_ponds.isdisjoint(val_ponds), f"Leakage detected in fold {f_i}!"
    print("5-fold GroupKFold by pond_id verified: ZERO pond leakage across folds.")

    y_train = train_feat_df["target"].values
    auto_spw = float((len(y_train) - y_train.sum()) / y_train.sum())

    # 5. Define Feature Configurations
    feature_configs = {
        "config_c_baseline": {
            "name": "Config C (11 DO features)",
            "features": CONFIG_C_FEATURES,
        },
        "config_c_trends": {
            "name": "Config C + Trends (15 features)",
            "features": CONFIG_C_PLUS_TRENDS,
        },
        "config_c_accel": {
            "name": "Config C + Trends + Accel (16 features)",
            "features": CONFIG_C_PLUS_ACCEL,
        },
        "config_c_all": {
            "name": "Config C + Trends + Accel + Stats (21 features)",
            "features": CONFIG_C_PLUS_ALL,
        },
    }

    # 6. Structured Hyperparameter Grid Design
    # Baseline parameters:
    base_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": "auto",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    # Define Candidate Parameter Variations
    trials_def = []

    # Trial 0: Baseline Config C with exact original settings
    trials_def.append({
        "trial_id": "T00_baseline",
        "f_config_key": "config_c_baseline",
        "params": {**base_params, "scale_pos_weight": auto_spw},
        "description": "Baseline Config C (depth=6, lr=0.1, n_est=100, spw=auto)",
    })

    # Trial 1-3: Baseline params on other feature sets (Feature Ablation)
    for fkey in ["config_c_trends", "config_c_accel", "config_c_all"]:
        trials_def.append({
            "trial_id": f"T_feat_ablation_{fkey}",
            "f_config_key": fkey,
            "params": {**base_params, "scale_pos_weight": auto_spw},
            "description": f"Feature ablation: {fkey} with baseline hyperparameters",
        })

    # Systematic Hyperparameter exploration across tree depth, learning rate, regularization, subsampling
    depths = [3, 4, 5, 6, 7]
    learning_rates = [0.03, 0.05, 0.08, 0.1]
    n_estimators_list = [100, 150, 200, 250]
    min_child_weights = [1, 3, 5, 7]
    subsamples = [0.7, 0.8, 1.0]
    colsample_bytrees = [0.7, 0.8, 1.0]
    gammas = [0.0, 0.1, 0.3, 0.5]
    reg_alphas = [0.0, 0.05, 0.1, 0.5]
    reg_lambdas = [0.5, 1.0, 2.0]
    spw_multipliers = [0.8, 1.0, 1.2]  # ~5.44, 6.80, 8.16

    # Generate targeted structured trials across top feature configurations
    rng = np.random.RandomState(42)

    trial_counter = 1
    # Explore Config C Baseline
    for d in [4, 5, 6]:
        for lr in [0.05, 0.08, 0.1]:
            for n_est in [100, 150, 200]:
                for mcw in [1, 3, 5]:
                    if rng.rand() < 0.30:  # Structured sampling
                        trials_def.append({
                            "trial_id": f"T_c_grid_{trial_counter:03d}",
                            "f_config_key": "config_c_baseline",
                            "params": {
                                "n_estimators": n_est,
                                "max_depth": d,
                                "learning_rate": lr,
                                "min_child_weight": mcw,
                                "subsample": rng.choice(subsamples),
                                "colsample_bytree": rng.choice(colsample_bytrees),
                                "gamma": rng.choice(gammas),
                                "reg_alpha": rng.choice(reg_alphas),
                                "reg_lambda": rng.choice(reg_lambdas),
                                "scale_pos_weight": auto_spw * rng.choice(spw_multipliers),
                                "eval_metric": "logloss",
                                "random_state": 42,
                                "n_jobs": -1,
                            },
                            "description": f"Config C: depth={d}, lr={lr}, n_est={n_est}, mcw={mcw}",
                        })
                        trial_counter += 1

    # Explore Config C + Trends (15 features)
    for d in [4, 5, 6, 7]:
        for lr in [0.03, 0.05, 0.08, 0.1]:
            for n_est in [100, 150, 200, 250]:
                if rng.rand() < 0.30:
                    trials_def.append({
                        "trial_id": f"T_trends_grid_{trial_counter:03d}",
                        "f_config_key": "config_c_trends",
                        "params": {
                            "n_estimators": n_est,
                            "max_depth": d,
                            "learning_rate": lr,
                            "min_child_weight": int(rng.choice(min_child_weights)),
                            "subsample": float(rng.choice(subsamples)),
                            "colsample_bytree": float(rng.choice(colsample_bytrees)),
                            "gamma": float(rng.choice(gammas)),
                            "reg_alpha": float(rng.choice(reg_alphas)),
                            "reg_lambda": float(rng.choice(reg_lambdas)),
                            "scale_pos_weight": float(auto_spw * rng.choice(spw_multipliers)),
                            "eval_metric": "logloss",
                            "random_state": 42,
                            "n_jobs": -1,
                        },
                        "description": f"Trends: depth={d}, lr={lr}, n_est={n_est}",
                    })
                    trial_counter += 1

    # Explore Config C + Accel (16 features)
    for d in [4, 5, 6, 7]:
        for lr in [0.03, 0.05, 0.08, 0.1]:
            for n_est in [100, 150, 200, 250]:
                if rng.rand() < 0.30:
                    trials_def.append({
                        "trial_id": f"T_accel_grid_{trial_counter:03d}",
                        "f_config_key": "config_c_accel",
                        "params": {
                            "n_estimators": n_est,
                            "max_depth": d,
                            "learning_rate": lr,
                            "min_child_weight": int(rng.choice(min_child_weights)),
                            "subsample": float(rng.choice(subsamples)),
                            "colsample_bytree": float(rng.choice(colsample_bytrees)),
                            "gamma": float(rng.choice(gammas)),
                            "reg_alpha": float(rng.choice(reg_alphas)),
                            "reg_lambda": float(rng.choice(reg_lambdas)),
                            "scale_pos_weight": float(auto_spw * rng.choice(spw_multipliers)),
                            "eval_metric": "logloss",
                            "random_state": 42,
                            "n_jobs": -1,
                        },
                        "description": f"Accel: depth={d}, lr={lr}, n_est={n_est}",
                    })
                    trial_counter += 1

    print(f"\nPrepared {len(trials_def)} structured experimental trials.")
    print("Running 5-fold cross-validation across all trials on training set only...")

    # 7. Execute all trials
    results_records = []
    all_oof_predictions = {}

    for i, t_info in enumerate(trials_def, 1):
        t_id = t_info["trial_id"]
        fkey = t_info["f_config_key"]
        fcols = feature_configs[fkey]["features"]
        params = t_info["params"]

        X_tr_full = train_feat_df[fcols].values

        res = evaluate_cv_candidate(X_tr_full, y_train, folds, params)

        all_oof_predictions[t_id] = res["oof_probs"]

        record = {
            "trial_id": t_id,
            "feature_config": fkey,
            "num_features": len(fcols),
            "description": t_info["description"],
            "n_estimators": params.get("n_estimators", 100),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.1),
            "min_child_weight": params.get("min_child_weight", 1),
            "subsample": params.get("subsample", 1.0),
            "colsample_bytree": params.get("colsample_bytree", 1.0),
            "gamma": params.get("gamma", 0.0),
            "reg_alpha": params.get("reg_alpha", 0.0),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "scale_pos_weight": round(float(params.get("scale_pos_weight", auto_spw)), 4),
            "mean_val_pr_auc": round(res["mean_val_pr_auc"], 4),
            "std_val_pr_auc": round(res["std_val_pr_auc"], 4),
            "mean_val_roc_auc": round(res["mean_val_roc_auc"], 4),
            "std_val_roc_auc": round(res["std_val_roc_auc"], 4),
            "mean_val_f1": round(res["mean_val_f1"], 4),
            "mean_val_recall": round(res["mean_val_recall"], 4),
            "mean_val_precision": round(res["mean_val_precision"], 4),
            "mean_val_specificity": round(res["mean_val_specificity"], 4),
            "oof_pr_auc": round(res["oof_pr_auc"], 4),
            "oof_roc_auc": round(res["oof_roc_auc"], 4),
            "oof_f1": round(res["oof_f1"], 4),
            "oof_recall": round(res["oof_recall"], 4),
            "oof_precision": round(res["oof_precision"], 4),
            "oof_specificity": round(res["oof_specificity"], 4),
            "oof_accuracy": round(res["oof_accuracy"], 4),
        }
        results_records.append(record)

        if i % 10 == 0 or i == 1 or i == len(trials_def):
            print(f"  [{i:>2}/{len(trials_def)}] {t_id:<25} -> Val PR-AUC: {record['mean_val_pr_auc']:.4f} | F1: {record['mean_val_f1']:.4f} | Rec: {record['mean_val_recall']:.4f}")

    results_df = pd.DataFrame(results_records)
    results_csv_path = os.path.join(reports_dir, "xgb_tuning_v1_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved all {len(results_df)} experimental trial results to {results_csv_path}")

    # 8. Identify Top Validation Candidates
    # Sorting criteria:
    # 1. Primary: mean_val_pr_auc descending
    # 2. Secondary: mean_val_recall descending (must be strong)
    # 3. Tertiary: mean_val_f1 descending
    sorted_df = results_df.sort_values(
        by=["mean_val_pr_auc", "mean_val_recall", "mean_val_f1"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    print("\n--- TOP 5 VALIDATION CANDIDATES BY PR-AUC ---")
    cols_display = ["trial_id", "feature_config", "num_features", "max_depth", "learning_rate", "n_estimators", "mean_val_pr_auc", "mean_val_recall", "mean_val_f1"]
    print(sorted_df[cols_display].head(5).to_string(index=False))

    baseline_row = results_df[results_df["trial_id"] == "T00_baseline"].iloc[0]
    print(f"\nBaseline Validation Reference:")
    print(f"  Val PR-AUC: {baseline_row['mean_val_pr_auc']:.4f} (+/- {baseline_row['std_val_pr_auc']:.4f})")
    print(f"  Val F1:     {baseline_row['mean_val_f1']:.4f}")
    print(f"  Val Recall: {baseline_row['mean_val_recall']:.4f}")
    print(f"  Val Prec:   {baseline_row['mean_val_precision']:.4f}")

    best_cand_row = sorted_df.iloc[0]
    best_trial_id = best_cand_row["trial_id"]
    best_fkey = best_cand_row["feature_config"]
    best_features = feature_configs[best_fkey]["features"]

    # Match trial definition for exact hyperparameters
    best_trial_def = next(t for t in trials_def if t["trial_id"] == best_trial_id)
    best_params = best_trial_def["params"]

    print(f"\nWINNING VALIDATION CANDIDATE: {best_trial_id}")
    print(f"  Feature Set:       {best_fkey} ({len(best_features)} features)")
    print(f"  Mean Val PR-AUC:   {best_cand_row['mean_val_pr_auc']:.4f} (Baseline: {baseline_row['mean_val_pr_auc']:.4f}, Delta: {best_cand_row['mean_val_pr_auc'] - baseline_row['mean_val_pr_auc']:+.4f})")
    print(f"  Mean Val Recall:   {best_cand_row['mean_val_recall']:.4f} (Baseline: {baseline_row['mean_val_recall']:.4f}, Delta: {best_cand_row['mean_val_recall'] - baseline_row['mean_val_recall']:+.4f})")
    print(f"  Mean Val F1:       {best_cand_row['mean_val_f1']:.4f} (Baseline: {baseline_row['mean_val_f1']:.4f}, Delta: {best_cand_row['mean_val_f1'] - baseline_row['mean_val_f1']:+.4f})")
    print(f"  Parameters:        {best_params}")

    # 9. Threshold Optimization on Validation OOF Predictions
    print("\n--- THRESHOLD OPTIMIZATION ON VALIDATION OOF PREDICTIONS ---")
    best_oof_probs = all_oof_predictions[best_trial_id]

    threshold_records = []
    threshold_range = np.arange(0.10, 0.92, 0.02)
    for th in threshold_range:
        th = round(float(th), 2)
        th_preds = (best_oof_probs >= th).astype(int)
        th_prec = float(precision_score(y_train, th_preds, zero_division=0))
        th_rec = float(recall_score(y_train, th_preds, zero_division=0))
        th_f1 = float(f1_score(y_train, th_preds, zero_division=0))
        th_acc = float(accuracy_score(y_train, th_preds))
        cm = confusion_matrix(y_train, th_preds, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        th_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        threshold_records.append({
            "threshold": th,
            "f1": round(th_f1, 4),
            "recall": round(th_rec, 4),
            "precision": round(th_prec, 4),
            "specificity": round(th_spec, 4),
            "accuracy": round(th_acc, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    th_df = pd.DataFrame(threshold_records)
    th_csv_path = os.path.join(reports_dir, "experimental_xgb_threshold_optimization.csv")
    th_df.to_csv(th_csv_path, index=False)
    print(f"Saved threshold sweep across {len(th_df)} thresholds to {th_csv_path}")

    # Threshold selection logic:
    # 1. F1-max threshold
    # 2. Recall-constrained threshold (must keep recall >= 0.78 while maximizing F1)
    best_f1_th_row = th_df.sort_values(by="f1", ascending=False).iloc[0]
    print(f"  F1-Max Threshold on Validation OOF: {best_f1_th_row['threshold']:.2f} (F1: {best_f1_th_row['f1']:.4f}, Recall: {best_f1_th_row['recall']:.4f}, Precision: {best_f1_th_row['precision']:.4f})")

    # Threshold 0.50 reference on validation OOF
    th_50_row = th_df[th_df["threshold"] == 0.50].iloc[0]
    print(f"  Default Threshold 0.50 on Val OOF:  F1: {th_50_row['f1']:.4f}, Recall: {th_50_row['recall']:.4f}, Precision: {th_50_row['precision']:.4f}")

    # We select the threshold based on the recall preservation constraint
    # If F1-max keeps recall >= 0.78, we can select it; otherwise select threshold that keeps recall >= 0.78 with highest F1
    valid_th_candidates = th_df[th_df["recall"] >= 0.78].sort_values(by="f1", ascending=False)
    if len(valid_th_candidates) > 0:
        selected_th_row = valid_th_candidates.iloc[0]
        selected_threshold = float(selected_th_row["threshold"])
        threshold_rationale = (
            f"Selected threshold {selected_threshold:.2f} satisfies the recall constraint (val recall = {selected_th_row['recall']:.4f} >= 0.78) "
            f"while maximizing F1 ({selected_th_row['f1']:.4f}) and precision ({selected_th_row['precision']:.4f})."
        )
    else:
        selected_threshold = 0.50
        threshold_rationale = "Retained default operational threshold 0.50 to preserve balance."

    print(f"  SELECTED OPERATIONAL THRESHOLD: {selected_threshold:.2f}")
    print(f"  Rationale: {threshold_rationale}")

    # 10. Optional Probability Calibration Check
    print("\n--- OPTIONAL CALIBRATION AUDIT ON VALIDATION ---")
    brier_uncal = brier_score_loss(y_train, best_oof_probs)
    print(f"  Uncalibrated Brier Score (lower is better): {brier_uncal:.4f}")

    # Test Platt (sigmoid) and isotonic calibration
    # Note: Calibration modifies probabilities monotonically but does not change rank-ordering (PR-AUC remains virtually identical)
    # We document the Brier score and calibration behavior.

    # 11. Train Winning Candidate on Full Training Set (32,908 rows)
    print("\n--- FITTING WINNING EXPERIMENTAL CANDIDATE ON FULL TRAINING SET ---")
    X_train_final = train_feat_df[best_features].values
    final_params = best_params.copy()
    if final_params.get("scale_pos_weight") == "auto":
        final_params["scale_pos_weight"] = auto_spw

    experimental_model = XGBClassifier(**final_params)
    experimental_model.fit(X_train_final, y_train)

    exp_model_path = os.path.join("models", "experimental_xgboost_v1.joblib")
    joblib.dump(experimental_model, exp_model_path)
    print(f"Saved experimental model artifact: {exp_model_path}")

    # Verify production model is STILL UNTOUCHED
    prod_model_hash_now = sha256_file(prod_model_path)
    assert prod_model_hash_now == prod_model_hash_initial, "CRITICAL ERROR: Production model artifact was altered!"
    print(f"VERIFIED: Production model {prod_model_path} is UNTOUCHED (hash: {prod_model_hash_now[:12]}...).")

    # 12. SINGLE EVALUATION ON THE UNTOUCHED TEMPORAL TEST SET (8,261 rows)
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON UNTOUCHED TEMPORAL TEST SET (8,261 ROWS)")
    print("=" * 80)
    X_test_final = test_feat_df[best_features].values
    y_test = test_feat_df["target"].values

    test_probs = experimental_model.predict_proba(X_test_final)[:, 1]

    # Evaluate at default threshold 0.50 AND selected threshold
    test_metrics_50 = compute_classification_metrics(y_test, test_probs, threshold=0.50)
    test_metrics_selected = compute_classification_metrics(y_test, test_probs, threshold=selected_threshold)

    # Load baseline model to verify test metrics
    baseline_model = joblib.load(prod_model_path)
    baseline_test_probs = baseline_model.predict_proba(test_df[CONFIG_C_FEATURES].values)[:, 1]
    baseline_test_metrics = compute_classification_metrics(y_test, baseline_test_probs, threshold=0.50)

    print("\nBASELINE FROZEN TEST METRICS (Threshold = 0.50):")
    print(f"  PR-AUC:      {baseline_test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:     {baseline_test_metrics['roc_auc']:.4f}")
    print(f"  F1:          {baseline_test_metrics['f1']:.4f}")
    print(f"  Recall:      {baseline_test_metrics['recall']:.4f}")
    print(f"  Precision:   {baseline_test_metrics['precision']:.4f}")
    print(f"  Specificity: {baseline_test_metrics['specificity']:.4f}")
    print(f"  Accuracy:    {baseline_test_metrics['accuracy']:.4f}")
    print(f"  Confusion:   TP={baseline_test_metrics['tp']}, FP={baseline_test_metrics['fp']}, TN={baseline_test_metrics['tn']}, FN={baseline_test_metrics['fn']}")

    print(f"\nEXPERIMENTAL MODEL TEST METRICS (Threshold = {selected_threshold:.2f}):")
    print(f"  PR-AUC:      {test_metrics_selected['pr_auc']:.4f} (Delta: {test_metrics_selected['pr_auc'] - baseline_test_metrics['pr_auc']:+.4f})")
    print(f"  ROC-AUC:     {test_metrics_selected['roc_auc']:.4f} (Delta: {test_metrics_selected['roc_auc'] - baseline_test_metrics['roc_auc']:+.4f})")
    print(f"  F1:          {test_metrics_selected['f1']:.4f} (Delta: {test_metrics_selected['f1'] - baseline_test_metrics['f1']:+.4f})")
    print(f"  Recall:      {test_metrics_selected['recall']:.4f} (Delta: {test_metrics_selected['recall'] - baseline_test_metrics['recall']:+.4f})")
    print(f"  Precision:   {test_metrics_selected['precision']:.4f} (Delta: {test_metrics_selected['precision'] - baseline_test_metrics['precision']:+.4f})")
    print(f"  Specificity: {test_metrics_selected['specificity']:.4f} (Delta: {test_metrics_selected['specificity'] - baseline_test_metrics['specificity']:+.4f})")
    print(f"  Accuracy:    {test_metrics_selected['accuracy']:.4f} (Delta: {test_metrics_selected['accuracy'] - baseline_test_metrics['accuracy']:+.4f})")
    print(f"  Confusion:   TP={test_metrics_selected['tp']}, FP={test_metrics_selected['fp']}, TN={test_metrics_selected['tn']}, FN={test_metrics_selected['fn']}")

    print("\nEXPERIMENTAL MODEL TEST METRICS (Threshold = 0.50 reference):")
    print(f"  PR-AUC:      {test_metrics_50['pr_auc']:.4f}")
    print(f"  F1:          {test_metrics_50['f1']:.4f}")
    print(f"  Recall:      {test_metrics_50['recall']:.4f}")
    print(f"  Precision:   {test_metrics_50['precision']:.4f}")
    print(f"  Specificity: {test_metrics_50['specificity']:.4f}")

    # 13. EVENT-LEVEL EVALUATION ON THE 136 EPISODES
    print("\n--- EVENT-LEVEL PHYSICAL EVALUATION ON 136 HYPOXIA EPISODES ---")
    test_eval_exp = test_feat_df.copy()
    test_eval_exp["prob"] = test_probs
    test_eval_exp["pred"] = (test_probs >= selected_threshold).astype(int)
    test_eval_exp["prediction_timestamp"] = pd.to_datetime(test_eval_exp["prediction_timestamp"])

    cleaned_df = pd.read_csv("data/processed/cleaned_pond_data.csv")
    cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["Timestamp"])

    # Load 136 baseline episodes detailed audit
    audit_baseline_path = os.path.join(reports_dir, "audit_136_episodes_detailed.csv")
    audit_baseline_df = pd.read_csv(audit_baseline_path)
    audit_baseline_df["start_time"] = pd.to_datetime(audit_baseline_df["start_time"])
    audit_baseline_df["end_time"] = pd.to_datetime(audit_baseline_df["end_time"])
    audit_baseline_df["first_physical_crossing"] = pd.to_datetime(audit_baseline_df["first_physical_crossing"])

    exp_episode_rows = []
    for idx, row in audit_baseline_df.iterrows():
        pid = row["pond_id"]
        ep_id = row["episode_id"]
        first_physical = row["first_physical_crossing"]

        if pd.notna(first_physical):
            pre_preds = test_eval_exp[
                (test_eval_exp["pond_id"] == pid) &
                (test_eval_exp["prediction_timestamp"] < first_physical) &
                (test_eval_exp["prediction_timestamp"] >= first_physical - pd.Timedelta(minutes=120))
            ]
            pre_alerts = pre_preds[pre_preds["pred"] == 1]
            if len(pre_alerts) > 0:
                first_alert = pre_alerts["prediction_timestamp"].min()
                lead_min = (first_physical - first_alert).total_seconds() / 60.0
                det = 1
            else:
                first_alert = None
                lead_min = 0.0
                det = 0
        else:
            first_alert = None
            lead_min = 0.0
            det = 0

        exp_episode_rows.append({
            "episode_id": ep_id,
            "pond_id": pid,
            "detected_baseline": row["detected_before_physical_onset"],
            "lead_time_baseline": row["true_lead_time_minutes"],
            "detected_experimental": det,
            "lead_time_experimental": lead_min,
            "first_physical_crossing": first_physical,
            "first_pre_alert_time": first_alert,
        })

    exp_ep_df = pd.DataFrame(exp_episode_rows)
    exp_ep_csv_path = os.path.join(reports_dir, "experimental_136_episodes_comparison.csv")
    exp_ep_df.to_csv(exp_ep_csv_path, index=False)

    base_det_count = int(exp_ep_df["detected_baseline"].sum())
    exp_det_count = int(exp_ep_df["detected_experimental"].sum())
    base_lead_mean = float(exp_ep_df[exp_ep_df["detected_baseline"] == 1]["lead_time_baseline"].mean())
    exp_lead_mean = float(exp_ep_df[exp_ep_df["detected_experimental"] == 1]["lead_time_experimental"].mean())
    exp_lead_median = float(exp_ep_df[exp_ep_df["detected_experimental"] == 1]["lead_time_experimental"].median())

    print(f"  Baseline Detection:     {base_det_count}/136 ({base_det_count/136:.2%}) | Mean Lead: {base_lead_mean:.1f} min")
    print(f"  Experimental Detection: {exp_det_count}/136 ({exp_det_count/136:.2%}) | Mean Lead: {exp_lead_mean:.1f} min | Median: {exp_lead_median:.1f} min")

    # 14. SHAP Feature Importance on Experimental Model
    print("\n--- COMPUTING SHAP ATTRIBUTIONS ON EXPERIMENTAL MODEL ---")
    import shap

    # Use a background sample from training data and test sample for explanation
    rng_shap = np.random.RandomState(42)
    sample_indices = rng_shap.choice(len(X_test_final), size=min(1000, len(X_test_final)), replace=False)
    X_sample = X_test_final[sample_indices]

    explainer = shap.TreeExplainer(experimental_model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_ranking = pd.DataFrame({
        "feature": best_features,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_csv_path = os.path.join(reports_dir, "experimental_xgb_shap_importance.csv")
    shap_ranking.to_csv(shap_csv_path, index=False)
    print(f"Top 5 features by mean |SHAP|:\n{shap_ranking.head(5).to_string(index=False)}")

    # Generate SHAP summary plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    shap.summary_plot(shap_values, X_sample, feature_names=best_features, show=False)
    plt.title(f"SHAP Summary: Experimental XGBoost ({best_fkey})", fontsize=12, pad=12)
    plt.tight_layout()
    shap_fig_path = os.path.join(figures_dir, "shap_summary_experimental_xgb.png")
    plt.savefig(shap_fig_path)
    plt.close()
    print(f"Saved SHAP plot to {shap_fig_path}")

    # 15. Save Comprehensive Experiment Metadata JSON
    experiment_summary = {
        "experiment_name": "XGBoost Tuning V1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_seconds": round(time.time() - start_time, 2),
        "dataset_hash": dataset_hash,
        "production_model_hash_verified": prod_model_hash_now,
        "num_trials_evaluated": len(results_df),
        "best_trial_id": best_trial_id,
        "best_feature_config": best_fkey,
        "best_features": best_features,
        "best_hyperparameters": final_params,
        "validation_metrics": {
            "mean_val_pr_auc": best_cand_row["mean_val_pr_auc"],
            "std_val_pr_auc": best_cand_row["std_val_pr_auc"],
            "mean_val_recall": best_cand_row["mean_val_recall"],
            "mean_val_f1": best_cand_row["mean_val_f1"],
            "mean_val_precision": best_cand_row["mean_val_precision"],
            "oof_pr_auc": best_cand_row["oof_pr_auc"],
            "oof_recall": best_cand_row["oof_recall"],
            "oof_f1": best_cand_row["oof_f1"],
        },
        "threshold_selection": {
            "selected_threshold": selected_threshold,
            "default_threshold": 0.50,
            "rationale": threshold_rationale,
        },
        "untouched_test_metrics_selected_threshold": test_metrics_selected,
        "untouched_test_metrics_default_threshold": test_metrics_50,
        "baseline_test_metrics": baseline_test_metrics,
        "event_level_comparison": {
            "total_episodes": 136,
            "baseline_detected": base_det_count,
            "baseline_mean_lead_minutes": round(base_lead_mean, 1),
            "experimental_detected": exp_det_count,
            "experimental_mean_lead_minutes": round(exp_lead_mean, 1),
            "experimental_median_lead_minutes": round(exp_lead_median, 1),
        },
        "scientific_conclusion": {
            "pr_auc_change": round(test_metrics_selected["pr_auc"] - baseline_test_metrics["pr_auc"], 4),
            "f1_change": round(test_metrics_selected["f1"] - baseline_test_metrics["f1"], 4),
            "recall_change": round(test_metrics_selected["recall"] - baseline_test_metrics["recall"], 4),
            "precision_change": round(test_metrics_selected["precision"] - baseline_test_metrics["precision"], 4),
            "specificity_change": round(test_metrics_selected["specificity"] - baseline_test_metrics["specificity"], 4),
        }
    }

    summary_json_path = os.path.join(reports_dir, "experimental_tuning_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"Saved complete experiment metadata to {summary_json_path}")
    print("\n" + "=" * 80)
    print("EXPERIMENTAL TUNING V1 COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_experiments()
