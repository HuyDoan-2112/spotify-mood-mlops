"""
Evaluate a saved model on a processed dataset.
Run with: `python -m src.evaluate --model models/binary_model.joblib --data processed.csv --target target`
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.base import is_classifier, is_regressor

from . import config
from .preprocessing import split_features_target
from .validation import drop_missing_targets


def evaluate(model_path: Path, data_file: str, target: str) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(config.DATA_PROCESSED / data_file)
    df = drop_missing_targets(df, target)
    X, y = split_features_target(df, target)

    y_pred = model.predict(X)

    if is_classifier(model):
        print(metrics.classification_report(y, y_pred))
    elif is_regressor(model):
        print(f"MAE: {metrics.mean_absolute_error(y, y_pred):.3f}")
        print(f"RMSE: {(metrics.mean_squared_error(y, y_pred, squared=False)):.3f}")
        print(f"R^2: {metrics.r2_score(y, y_pred):.3f}")
    else:
        raise ValueError("Loaded model is neither classifier nor regressor.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a joblib model.")
    parser.add_argument("--data", required=True, help="Processed CSV filename in data/processed/")
    parser.add_argument("--target", required=True, help="Target column name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.data, args.target)
