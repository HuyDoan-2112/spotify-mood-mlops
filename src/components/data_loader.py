from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path)
    return df


def validate_schema(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Found NaNs in required columns")

    if len(df) == 0:
        raise ValueError("Empty dataframe")


def dataset_hash(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> str:
    subset = df[feature_cols + [target_col]]
    return str(pd.util.hash_pandas_object(subset, index=True).sum())


def make_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    if len(X) != len(y):
        raise ValueError("X and y lengths do not match")
    return X, y


def split_train_val_test(X, y, train_size, val_size, test_size, random_state):
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train/val/test sizes must sum to 1.0")
    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split temp into train/val
    val_size_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_rel, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_split(cfg):
    df = load_raw_csv(cfg.RAW_DATA_PATH)
    validate_schema(df, cfg.FEATURE_COLS, cfg.TARGET_COL)
    X, y = make_xy(df, cfg.FEATURE_COLS, cfg.TARGET_COL)
    return split_train_val_test(
        X, y, cfg.TRAIN_SIZE, cfg.VAL_SIZE, cfg.TEST_SIZE, cfg.RANDOM_STATE
    )
