"""
Feature engineering helpers and sklearn pipelines.
"""
from typing import Iterable, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocess_pipeline(
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
    remainder: str = "drop",
    handle_unknown: str = "ignore",
    scaler: Optional[StandardScaler] = StandardScaler(),
) -> ColumnTransformer:
    """Create a ColumnTransformer that scales numeric and one-hot encodes categoricals."""
    numeric_transformer = Pipeline(steps=[("scaler", scaler)]) if numeric_cols else "drop"
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))]
    ) if categorical_cols else "drop"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ],
        remainder=remainder,
    )


def infer_column_types(df: pd.DataFrame, target: str):
    """Infer numeric and categorical feature columns from a dataframe."""
    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols
