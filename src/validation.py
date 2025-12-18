"""
Lightweight validation helpers for incoming datasets.
"""
from typing import Iterable
import pandas as pd


def assert_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise a ValueError if any required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def drop_missing_targets(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a copy with rows missing the target removed."""
    if target not in df:
        raise ValueError(f"Target column '{target}' not found.")
    return df.dropna(subset=[target]).copy()
