"""Validation split helpers. Select a strategy that matches the data-generating process."""
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split


def make_splitter(strategy="kfold", n_splits=5, random_state=42):
    strategy = strategy.lower()
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unknown validation strategy: {strategy}")


def split_indices(X, y=None, strategy="kfold", n_splits=5, random_state=42, groups=None, test_size=.2):
    if strategy.lower() == "holdout":
        indices = list(range(len(X)))
        stratify = None
        if y is not None:
            y_series = pd.Series(y)
            if not pd.api.types.is_numeric_dtype(y_series) or y_series.nunique() <= 20:
                stratify = y
        train_idx, valid_idx = train_test_split(indices, test_size=test_size, random_state=random_state,
                                                stratify=stratify)
        return [(train_idx, valid_idx)]
    splitter = make_splitter(strategy, n_splits, random_state)
    if strategy.lower() == "group_kfold":
        if groups is None:
            raise ValueError("groups are required for group_kfold")
        return list(splitter.split(X, y, groups))
    return list(splitter.split(X, y))
