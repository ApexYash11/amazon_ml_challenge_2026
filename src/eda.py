"""Lightweight EDA summaries and plots; nothing runs automatically at import."""
import matplotlib.pyplot as plt
import pandas as pd

from .data import compare_train_test, detect_column_types, find_missing, summarize_dataframe


def dataframe_overview(df):
    return {**summarize_dataframe(df), "columns_by_type": detect_column_types(df),
            "missing": find_missing(df)}


def target_distribution(target, max_categories=30):
    series = pd.Series(target).dropna()
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > max_categories:
        return series.describe(percentiles=[.01, .1, .25, .5, .75, .9, .99])
    return series.value_counts(dropna=False).head(max_categories)


def plot_numeric_distributions(df, columns=None, max_columns=12):
    columns = columns or detect_column_types(df)["numeric"]
    columns = list(columns)[:max_columns]
    if not columns:
        return None
    axes = df[columns].hist(figsize=(12, max(3, 2.5 * ((len(columns) + 2) // 3))), bins=30)
    return axes


def categorical_cardinality(df, columns=None):
    columns = columns or detect_column_types(df)["categorical"]
    return pd.Series({col: df[col].nunique(dropna=False) for col in columns}, name="unique_count").sort_values(ascending=False)


def missingness(df):
    return find_missing(df)


def train_test_distribution(train, test, columns=None, max_categories=20):
    shared = [c for c in train.columns if c in test.columns]
    columns = columns or shared
    result = {}
    for col in columns:
        if col not in train or col not in test:
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            result[col] = {"train": train[col].describe(), "test": test[col].describe()}
        else:
            result[col] = {"train": train[col].value_counts(normalize=True).head(max_categories),
                           "test": test[col].value_counts(normalize=True).head(max_categories)}
    return result


def suspicious_leakage(train, target, id_column=None, unique_threshold=0.98):
    """Flag review candidates; this is heuristic and does not prove leakage."""
    flags = []
    n_rows = max(len(train), 1)
    for col in train.columns:
        if col == target:
            continue
        unique_ratio = train[col].nunique(dropna=False) / n_rows
        reasons = []
        if id_column and col == id_column:
            reasons.append("configured identifier")
        if unique_ratio >= unique_threshold:
            reasons.append(f"near-unique ({unique_ratio:.3f})")
        if train[col].equals(train[target]):
            reasons.append("identical to target")
        if reasons:
            flags.append({"column": col, "reasons": reasons})
    return flags
