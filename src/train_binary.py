"""
Binary classification training script.
Run with: `python -m src.train_binary --data binary.csv --target target_binary`
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size or config.DEFAULT_BINARY_CONFIG.test_size,
        random_state=config.DEFAULT_BINARY_CONFIG.random_state, stratify=y,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_out or config.MODELS_DIR / "binary_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary classifier.")
    parser.add_argument("--data", default="binary.csv", help="Processed CSV filename in data/processed/")
    parser.add_argument("--target", default=config.DEFAULT_BINARY_CONFIG.target, help="Target column name.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the model.")
    parser.add_argument("--test-size", type=float, default=None, help="Validation split size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.target, args.output, args.test_size)
