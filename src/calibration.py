"""
ShinerAI: Probability Calibration & Cost-Sensitive Threshold Module
Evaluates:
1. Probability calibration quality (Brier score, reliability diagrams)
2. Post-hoc calibration (Platt scaling, Isotonic regression fitted strictly on training data)
3. Cost-sensitive decision threshold optimization (evaluating FN vs FP economic trade-offs)
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Brier score, reliability curve bins, and Expected Calibration Error (ECE).
    """
    brier = float(brier_score_loss(y_true, probs))
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    
    # Compute Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_records = []
    
    for i in range(n_bins):
        in_bin = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if i == n_bins - 1:
            in_bin = (probs >= bin_edges[i]) & (probs <= bin_edges[i+1])
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / len(y_true)) * bin_error
            bin_records.append({
                "bin_idx": i + 1,
                "bin_range": f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})",
                "sample_count": bin_count,
                "predicted_prob": round(bin_conf, 4),
                "empirical_freq": round(bin_acc, 4),
                "calibration_gap": round(bin_acc - bin_conf, 4),
            })
            
    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(float(ece), 4),
        "bin_diagnostics": bin_records,
    }


def fit_and_evaluate_calibrators(
    y_train: np.ndarray,
    train_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
) -> Dict[str, Any]:
    """
    Fits Platt scaling and Isotonic regression strictly on training set probabilities,
    and evaluates on held-out test probabilities.
    """
    eps = 1e-7
    train_logits = np.log((train_probs + eps) / (1 - train_probs + eps)).reshape(-1, 1)
    test_logits = np.log((test_probs + eps) / (1 - test_probs + eps)).reshape(-1, 1)
    
    # 1. Platt Scaling
    platt = LogisticRegression()
    platt.fit(train_logits, y_train)
    test_probs_platt = platt.predict_proba(test_logits)[:, 1]
    
    # 2. Isotonic Regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_probs, y_train)
    test_probs_iso = iso.predict(test_probs)
    
    uncal_diag = compute_calibration_diagnostics(y_test, test_probs)
    platt_diag = compute_calibration_diagnostics(y_test, test_probs_platt)
    iso_diag = compute_calibration_diagnostics(y_test, test_probs_iso)
    
    return {
        "uncalibrated": {
            "brier_score": uncal_diag["brier_score"],
            "ece": uncal_diag["expected_calibration_error"],
            "probs": test_probs,
            "bins": uncal_diag["bin_diagnostics"],
        },
        "platt_scaling": {
            "brier_score": platt_diag["brier_score"],
            "ece": platt_diag["expected_calibration_error"],
            "probs": test_probs_platt,
            "bins": platt_diag["bin_diagnostics"],
        },
        "isotonic_regression": {
            "brier_score": iso_diag["brier_score"],
            "ece": iso_diag["expected_calibration_error"],
            "probs": test_probs_iso,
            "bins": iso_diag["bin_diagnostics"],
        },
    }


def analyze_cost_sensitive_thresholds(
    y_train: np.ndarray,
    train_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    cost_ratios: List[Tuple[float, float]] = [(1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (10.0, 1.0)],
) -> Dict[str, Any]:
    """
    Performs cost-sensitive threshold search on training set,
    then evaluates optimal thresholds on held-out test set.
    """
    threshold_grid = np.arange(0.10, 0.95, 0.05)
    
    train_table = []
    for th in threshold_grid:
        preds = (train_probs >= th).astype(int)
        tp = int(((preds == 1) & (y_train == 1)).sum())
        fp = int(((preds == 1) & (y_train == 0)).sum())
        tn = int(((preds == 0) & (y_train == 0)).sum())
        fn = int(((preds == 0) & (y_train == 1)).sum())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / len(y_train)
        
        row = {
            "threshold": round(float(th), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "specificity": round(float(spec), 4),
            "accuracy": round(float(acc), 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        for (c_fn, c_fp) in cost_ratios:
            norm_cost = (c_fn * fn + c_fp * fp) / len(y_train)
            row[f"cost_{int(c_fn)}_{int(c_fp)}"] = round(float(norm_cost), 4)
        train_table.append(row)
        
    df_train = pd.DataFrame(train_table)
    
    # Identify optimal thresholds on train set
    optimal_thresholds = {}
    for (c_fn, c_fp) in cost_ratios:
        col = f"cost_{int(c_fn)}_{int(c_fp)}"
        best_th = float(df_train.loc[df_train[col].idxmin(), "threshold"])
        optimal_thresholds[f"{int(c_fn)}:1"] = best_th
        
    optimal_thresholds["best_f1"] = float(df_train.loc[df_train["f1"].idxmax(), "threshold"])
    
    # Evaluate chosen thresholds on untouched test set
    test_eval_table = []
    eval_thresholds = sorted(list(set(list(optimal_thresholds.values()) + [0.50])))
    for th in eval_thresholds:
        preds = (test_probs >= th).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / len(y_test)
        
        row = {
            "threshold": round(float(th), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "specificity": round(float(spec), 4),
            "accuracy": round(float(acc), 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        for (c_fn, c_fp) in cost_ratios:
            norm_cost = (c_fn * fn + c_fp * fp) / len(y_test)
            row[f"cost_{int(c_fn)}_{int(c_fp)}"] = round(float(norm_cost), 4)
        test_eval_table.append(row)
        
    return {
        "train_threshold_table": train_table,
        "optimal_thresholds_on_train": optimal_thresholds,
        "test_evaluation_table": test_eval_table,
    }
