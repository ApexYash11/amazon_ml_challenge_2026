"""Generate a submission using the saved training pipeline and YAML schema."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import load_data


def main(config_path):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    bundle = joblib.load(config["artifacts"]["model_path"])
    test = load_data(config["data"]["test_path"])
    submission_cfg = config["submission"]
    id_column = submission_cfg.get("id_column") or config["data"].get("id_column")
    if id_column and id_column not in test:
        raise ValueError(f"Configured ID column {id_column!r} is missing from the test data.")
    ids = test[id_column].copy() if id_column else None
    features = test.drop(columns=[id_column]) if id_column else test.copy()
    expected = bundle["feature_columns"]
    missing = sorted(set(expected) - set(features.columns))
    extra = sorted(set(features.columns) - set(expected))
    if missing:
        raise ValueError(f"Test data is missing model features: {missing}")
    if extra:
        print(f"Ignoring extra test columns: {extra}")
    features = features[expected]
    predictions = bundle["pipeline"].predict(features)
    if len(predictions) != len(test):
        raise ValueError("Prediction count does not match test row count.")
    prediction_array = np.asarray(predictions)
    if pd.isna(prediction_array).any():
        raise ValueError("Predictions contain NaN values.")
    if np.issubdtype(prediction_array.dtype, np.number) and not np.isfinite(prediction_array).all():
        raise ValueError("Predictions contain infinite values.")
    output = {}
    if id_column:
        output[id_column] = ids.to_numpy()
        if len(output[id_column]) != len(test):
            raise ValueError("ID count does not match test row count.")
    output[submission_cfg["prediction_column"]] = predictions
    submission = pd.DataFrame(output)
    required = ([id_column] if id_column else []) + [submission_cfg["prediction_column"]]
    if submission.columns.tolist() != required:
        raise ValueError("Submission column names or ordering are invalid.")
    if submission.isna().any().any():
        raise ValueError("Submission contains unexpected missing values.")
    if len(submission) != len(test):
        raise ValueError("Submission row count does not match test row count.")
    output_path = Path(submission_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
