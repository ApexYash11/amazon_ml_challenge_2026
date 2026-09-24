"""Small, schema-agnostic data inspection helpers."""
from pathlib import Path

import pandas as pd


def load_data(path):
    """Load CSV or Parquet data based on its file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data format: {path.suffix}. Use CSV or Parquet.")


def summarize_dataframe(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 3),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def find_missing(df):
    counts = df.isna().sum()
    result = pd.DataFrame({"missing_count": counts, "missing_fraction": counts / max(len(df), 1)})
    return result[result["missing_count"] > 0].sort_values("missing_count", ascending=False)


def find_duplicates(df, subset=None):
    return int(df.duplicated(subset=subset).sum())


def compare_train_test(train, test):
    train_columns, test_columns = set(train.columns), set(test.columns)
    return {
        "train_only": sorted(train_columns - test_columns),
        "test_only": sorted(test_columns - train_columns),
        "shared": sorted(train_columns & test_columns),
        "dtype_mismatches": {
            col: (str(train[col].dtype), str(test[col].dtype))
            for col in train_columns & test_columns
            if train[col].dtype != test[col].dtype
        },
    }


def detect_column_types(df, text_min_length=30):
    """Return conservative numeric, categorical, and likely-text column lists."""
    numeric, categorical, text = [], [], []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric.append(col)
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            non_null = series.dropna().astype(str)
            if not non_null.empty and non_null.str.len().median() >= text_min_length:
                text.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical, "text": text}


def summarize_target(target):
    series = pd.Series(target).dropna()
    result = {"count": int(series.size), "missing": int(pd.Series(target).isna().sum()),
              "unique": int(series.nunique())}
    if pd.api.types.is_numeric_dtype(series):
        result["describe"] = series.describe().to_dict()
    else:
        result["top_values"] = series.value_counts().head(10).to_dict()
    return result
