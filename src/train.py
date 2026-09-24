"""Configurable LightGBM baseline for ordinary tabular regression/classification."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import detect_column_types, load_data
from .metrics import score
from .validation import split_indices


def build_preprocessor(X):
    kinds = detect_column_types(X)
    numeric = kinds["numeric"]
    categorical = kinds["categorical"] + kinds["text"]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
    if categorical:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", cat_pipeline, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop"), kinds


def main(config_path):
    started = time.perf_counter()
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_cfg, val_cfg = config["data"], config["validation"]
    target = data_cfg.get("target")
    if not target:
        raise ValueError("Set data.target in the YAML configuration after identifying the official target.")
    train = load_data(data_cfg["train_path"])
    if target not in train:
        raise ValueError(f"Target column {target!r} is not in the training data.")
    excluded = [target]
    id_column = data_cfg.get("id_column")
    if id_column and id_column in train:
        excluded.append(id_column)
    group_column = val_cfg.get("group_column")
    if group_column and group_column in train:
        excluded.append(group_column)
    X = train.drop(columns=excluded)
    y = train[target]
    if y.isna().any():
        raise ValueError("Target contains missing values; resolve these according to the task before training.")
    task = config["model"].get("task", "regression").lower()
    if task not in {"regression", "classification"}:
        raise ValueError("model.task must be regression or classification for this starter trainer.")
    if task == "classification" and val_cfg["strategy"] == "kfold":
        print("Note: for classification, consider stratified_kfold after inspecting the target distribution.")
    preprocessor, kinds = build_preprocessor(X)

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError("Install dependencies with `pip install -r requirements.txt` to use LightGBM.") from exc
    model_cls = LGBMRegressor if task == "regression" else LGBMClassifier
    estimator = model_cls(random_state=int(config.get("seed", 42)), verbosity=-1,
                          **config["model"].get("params", {}))
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    groups = train[group_column] if group_column and group_column in train else None
    folds = split_indices(X, y, strategy=val_cfg.get("strategy", "kfold"),
                          n_splits=int(val_cfg.get("n_splits", 5)),
                          random_state=int(val_cfg.get("random_state", config.get("seed", 42))),
                          groups=groups, test_size=float(val_cfg.get("test_size", .2)))
    metric_name = config.get("metric", {}).get("name", "rmse").lower()
    fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction = pipeline.predict(X.iloc[valid_idx])
        fold_score = score(metric_name, y.iloc[valid_idx], prediction)
        fold_scores.append(fold_score)
        print(f"Fold {fold}: {metric_name}={fold_score:.6f}")
    print(f"CV {metric_name}: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")

    pipeline.fit(X, y)
    model_path = Path(config["artifacts"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": list(X.columns), "task": task,
                 "target": target, "classes": getattr(estimator, "classes_", None)}, model_path)
    feature_info_path = Path(config["artifacts"]["feature_info_path"])
    feature_info_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_info_path.open("w", encoding="utf-8") as file:
        json.dump({"raw_features": list(X.columns), "detected_types": kinds,
                   "target": target, "task": task}, file, indent=2)
    print(f"Saved model: {model_path}")
    print(f"Saved feature information: {feature_info_path}")
    print(f"Runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
