"""
Regression training script.
Run with: `python -m src.train_regression --data regression.csv --target target_regression`
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocess_pipeline, infer_column_types
from .preprocessing import basic_clean, split_features_target
from .validation import drop_missing_targets


def load_dataset(filename: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED / filename
    return pd.read_csv(path)


def train(
    data_file: str,
    target: str,
    model_out: Path | None = None,
    test_size: float | None = None,
) -> Path:
    df = load_dataset(data_file)
    df = basic_clean(df)
    df = drop_missing_targets(df, target)

    X, y = split_features_target(df, target)
    numeric_cols, categorical_cols = infer_column_types(df, target)

    preprocess = build_preprocess_pipeline(numeric_cols, categorical_cols)
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=config.DEFAULT_REGRESSION_CONFIG.random_state,
        n_jobs=-1,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size or config.DEFAULT_REGRESSION_CONFIG.test_size,
        random_state=config.DEFAULT_REGRESSION_CONFIG.random_state,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print(f"MAE: {mean_absolute_error(y_val, y_pred):.3f}")
    print(f"RMSE: {(mean_squared_error(y_val, y_pred, squared=False)):.3f}")
    print(f"R^2: {r2_score(y_val, y_pred):.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_out or config.MODELS_DIR / "regression_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a regression model.")
    parser.add_argument("--data", default="regression.csv", help="Processed CSV filename in data/processed/")
    parser.add_argument("--target", default=config.DEFAULT_REGRESSION_CONFIG.target, help="Target column name.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the model.")
    parser.add_argument("--test-size", type=float, default=None, help="Validation split size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.target, args.output, args.test_size)
