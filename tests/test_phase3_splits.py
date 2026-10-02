"""
Unit tests for Phase 3 temporal holdout and GroupKFold splitting.
Verifies temporal ordering, 2-hour purge gap enforcement, and zero cross-pond leakage.
"""

import os
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.model_utils import (
    verify_dataset_preconditions,
    create_temporal_split,
)


def test_dataset_preconditions():
    """Verify that ml_ready_dataset.csv passes all Phase 3 integrity checks."""
    data_path = os.path.join("data", "processed", "ml_ready_dataset.csv")
    assert os.path.exists(data_path), f"File {data_path} not found"
    df = pd.read_csv(data_path)
    res = verify_dataset_preconditions(df)
    assert res["status"] == "PASS"
    assert res["rows"] == 41277
    assert res["columns"] == 37
    assert res["pond_count"] == 17


def test_temporal_split_chronological_ordering_and_purge_gap():
    """Verify that train < test chronologically and the 2h purge gap is strictly respected."""
    data_path = os.path.join("data", "processed", "ml_ready_dataset.csv")
    df = pd.read_csv(data_path)
    train_df, test_df, purged_df, accounting_df = create_temporal_split(df, train_ratio=0.80, purge_hours=2.0)

    # 1. Row reconciliation identity
    assert len(train_df) + len(purged_df) + len(test_df) == 41277
    assert len(accounting_df) == 17

    # 2. Check each pond's temporal sequence
    for pond_id, group in df.groupby("pond_id"):
        tr_pond = train_df[train_df["pond_id"] == pond_id]
        te_pond = test_df[test_df["pond_id"] == pond_id]
        pu_pond = purged_df[purged_df["pond_id"] == pond_id]

        assert len(tr_pond) > 0, f"Pond {pond_id} has empty train set"
        assert len(te_pond) > 0, f"Pond {pond_id} has empty test set"

        max_train_t = pd.to_datetime(tr_pond["prediction_timestamp"]).max()
        min_test_t = pd.to_datetime(te_pond["prediction_timestamp"]).min()

        # Strict temporal ordering: train is strictly before test
        assert max_train_t < min_test_t, f"Pond {pond_id} train overlaps or is after test!"

        # Purge gap verification: delta must be >= 2 hours (120 minutes)
        time_gap = min_test_t - max_train_t
        assert time_gap >= pd.Timedelta(hours=2), (
            f"Pond {pond_id} purge gap is only {time_gap}, expected >= 2 hours"
        )


def test_group_kfold_no_cross_pond_leakage():
    """Verify that in GroupKFold, no test pond ever appears in training for any fold."""
    data_path = os.path.join("data", "processed", "ml_ready_dataset.csv")
    df = pd.read_csv(data_path)

    gkf = GroupKFold(n_splits=5)
    groups = df["pond_id"]

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_ponds = set(df.iloc[train_idx]["pond_id"].unique())
        test_ponds = set(df.iloc[test_idx]["pond_id"].unique())

        # Sets must be strictly disjoint
        intersection = train_ponds.intersection(test_ponds)
        assert len(intersection) == 0, f"Fold {fold} has leaking ponds: {intersection}"
        assert len(train_ponds) + len(test_ponds) == 17
