"""Small generic feature helpers; task-specific features belong in explicit hooks."""
import numpy as np
import pandas as pd


def add_missing_indicators(df, columns=None):
    result = df.copy()
    columns = columns or list(result.columns)
    for col in columns:
        if result[col].isna().any():
            result[f"{col}__missing"] = result[col].isna().astype("int8")
    return result


def add_log_transforms(df, columns):
    result = df.copy()
    for col in columns:
        values = pd.to_numeric(result[col], errors="coerce")
        if (values.dropna() < 0).any():
            raise ValueError(f"Cannot log-transform negative values in {col!r}; use a signed transform explicitly.")
        result[f"{col}__log1p"] = np.log1p(values)
    return result


def add_ratios(df, ratios):
    """Add only configured ratios; ratios is a mapping of new name to (numerator, denominator)."""
    result = df.copy()
    for name, (numerator, denominator) in ratios.items():
        divisor = pd.to_numeric(result[denominator], errors="coerce").replace(0, np.nan)
        result[name] = pd.to_numeric(result[numerator], errors="coerce") / divisor
    return result


def text_features(series, prefix=None):
    text = series.fillna("").astype(str)
    prefix = prefix or series.name or "text"
    words = text.str.findall(r"\b\w+\b")
    return pd.DataFrame({
        f"{prefix}__char_count": text.str.len(),
        f"{prefix}__word_count": words.str.len(),
        f"{prefix}__digit_count": text.str.count(r"\d"),
        f"{prefix}__punctuation_count": text.str.count(r"[^\w\s]"),
        f"{prefix}__unique_token_count": words.map(lambda tokens: len(set(t.lower() for t in tokens))),
    }, index=series.index)


def frequency_encode(train, test, columns):
    """Return copies with train-derived normalized frequency encodings."""
    train_out, test_out = train.copy(), test.copy()
    for col in columns:
        frequencies = train[col].value_counts(normalize=True, dropna=False)
        train_out[f"{col}__frequency"] = train[col].map(frequencies).fillna(0)
        test_out[f"{col}__frequency"] = test[col].map(frequencies).fillna(0)
    return train_out, test_out


def add_task_features(df):
    """Add explicitly designed, task-specific features after EDA."""
    return df.copy()
