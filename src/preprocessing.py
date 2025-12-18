"""
Data loading and preprocessing utilities.
"""
from pathlib import Path
from typing import Optional
import pandas as pd

from . import config
from .validation import assert_required_columns


def load_raw_csv(filename: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a CSV from the raw data directory."""
    data_dir = base_dir or config.DATA_RAW
    path = data_dir / filename
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame, drop_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Drop duplicates, optional columns, and standardize column names."""
    cleaned = df.copy()
    cleaned.columns = [c.strip() for c in cleaned.columns]
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols, errors="ignore")
    cleaned = cleaned.drop_duplicates()
    return cleaned


def split_features_target(df: pd.DataFrame, target: str):
    """Return X, y split for model training."""
    assert_required_columns(df, [target])
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
