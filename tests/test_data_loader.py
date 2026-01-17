"""T2, T3: Unit tests for data loading and validation (src/components/data_loader.py)."""
import pandas as pd
import numpy as np
import pytest

from src.components.data_loader import (
    validate_schema,
    make_xy,
    split_train_val_test,
    dataset_hash,
)
from src.config import FEATURE_COLS, TARGET_COL


class TestValidateSchema:
    """T2: Test suite for validate_schema() function."""

    def test_raises_on_missing_feature_column(self):
        """T2: Verify ValueError raised when feature column is missing."""
        # Create DataFrame missing 'danceability'
        df = pd.DataFrame({
            "duration (ms)": [210000],
            # "danceability" intentionally missing
            "energy": [0.8],
            "loudness": [-5.0],
            "speechiness": [0.05],
            "acousticness": [0.2],
            "instrumentalness": [0.1],
            "liveness": [0.15],
            "valence": [0.6],
            "tempo": [120.0],
            "labels": [0],
        })

        with pytest.raises(ValueError) as excinfo:
            validate_schema(df, FEATURE_COLS, TARGET_COL)

        assert "Missing columns" in str(excinfo.value)
        assert "danceability" in str(excinfo.value)

    def test_raises_on_missing_target_column(self):
        """Verify ValueError raised when target column is missing."""
        df = pd.DataFrame({
            "duration (ms)": [210000],
            "danceability": [0.7],
            "energy": [0.8],
            "loudness": [-5.0],
            "speechiness": [0.05],
            "acousticness": [0.2],
            "instrumentalness": [0.1],
            "liveness": [0.15],
            "valence": [0.6],
            "tempo": [120.0],
            # "labels" intentionally missing
        })

        with pytest.raises(ValueError) as excinfo:
            validate_schema(df, FEATURE_COLS, TARGET_COL)

        assert "Missing columns" in str(excinfo.value)
        assert TARGET_COL in str(excinfo.value)

    def test_raises_on_nan_in_required_columns(self):
        """Verify ValueError raised when NaN exists in required columns."""
        df = pd.DataFrame({
            "duration (ms)": [210000],
            "danceability": [np.nan],  # NaN value
            "energy": [0.8],
            "loudness": [-5.0],
            "speechiness": [0.05],
            "acousticness": [0.2],
            "instrumentalness": [0.1],
            "liveness": [0.15],
            "valence": [0.6],
            "tempo": [120.0],
            "labels": [0],
        })

        with pytest.raises(ValueError) as excinfo:
            validate_schema(df, FEATURE_COLS, TARGET_COL)

        assert "NaN" in str(excinfo.value)

    def test_raises_on_empty_dataframe(self):
        """Verify ValueError raised for empty DataFrame."""
        df = pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL])

        with pytest.raises(ValueError) as excinfo:
            validate_schema(df, FEATURE_COLS, TARGET_COL)

        assert "Empty" in str(excinfo.value)

    def test_passes_with_valid_data(self):
        """Verify no exception raised for valid data."""
        df = pd.DataFrame({
            "duration (ms)": [210000, 180000],
            "danceability": [0.7, 0.5],
            "energy": [0.8, 0.6],
            "loudness": [-5.0, -8.0],
            "speechiness": [0.05, 0.10],
            "acousticness": [0.2, 0.3],
            "instrumentalness": [0.1, 0.2],
            "liveness": [0.15, 0.20],
            "valence": [0.6, 0.4],
            "tempo": [120.0, 100.0],
            "labels": [0, 1],
        })

        # Should not raise
        validate_schema(df, FEATURE_COLS, TARGET_COL)

    def test_raises_on_multiple_missing_columns(self):
        """Verify all missing columns reported in error message."""
        df = pd.DataFrame({
            "duration (ms)": [210000],
            # Multiple columns missing
            "labels": [0],
        })

        with pytest.raises(ValueError) as excinfo:
            validate_schema(df, FEATURE_COLS, TARGET_COL)

        error_msg = str(excinfo.value)
        assert "Missing columns" in error_msg
        # Should list multiple missing columns
        assert "danceability" in error_msg
        assert "energy" in error_msg


class TestMakeXY:
    """Test suite for make_xy() function."""

    def test_extracts_correct_columns(self):
        """Verify X contains only feature columns and y contains target."""
        df = pd.DataFrame({
            "duration (ms)": [210000, 180000],
            "danceability": [0.7, 0.5],
            "energy": [0.8, 0.6],
            "loudness": [-5.0, -8.0],
            "speechiness": [0.05, 0.10],
            "acousticness": [0.2, 0.3],
            "instrumentalness": [0.1, 0.2],
            "liveness": [0.15, 0.20],
            "valence": [0.6, 0.4],
            "tempo": [120.0, 100.0],
            "labels": [0, 1],
            "extra_column": ["a", "b"],  # Should be excluded from X
        })

        X, y = make_xy(df, FEATURE_COLS, TARGET_COL)

        assert list(X.columns) == FEATURE_COLS
        assert "extra_column" not in X.columns
        assert y.name == TARGET_COL
        assert len(X) == len(y) == 2

    def test_preserves_data_integrity(self):
        """Verify data values are preserved correctly."""
        df = pd.DataFrame({
            "duration (ms)": [210000],
            "danceability": [0.7],
            "energy": [0.8],
            "loudness": [-5.0],
            "speechiness": [0.05],
            "acousticness": [0.2],
            "instrumentalness": [0.1],
            "liveness": [0.15],
            "valence": [0.6],
            "tempo": [120.0],
            "labels": [2],
        })

        X, y = make_xy(df, FEATURE_COLS, TARGET_COL)

        assert X["danceability"].iloc[0] == 0.7
        assert y.iloc[0] == 2


class TestSplitTrainValTest:
    """T3: Test suite for split_train_val_test() function."""

    def test_preserves_class_distribution(self, classification_dataset):
        """T3: Verify stratified split preserves class distribution within 5%."""
        X, y = classification_dataset

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
        )

        # Original distribution: 80% class 0, 20% class 1
        original_ratio = (y == 0).mean()

        train_ratio = (y_train == 0).mean()
        val_ratio = (y_val == 0).mean()
        test_ratio = (y_test == 0).mean()

        # All splits should be within 15% of original (allow some variance for small samples)
        assert abs(train_ratio - original_ratio) < 0.15, f"Train ratio {train_ratio} deviates from {original_ratio}"
        assert abs(val_ratio - original_ratio) < 0.15, f"Val ratio {val_ratio} deviates from {original_ratio}"
        assert abs(test_ratio - original_ratio) < 0.15, f"Test ratio {test_ratio} deviates from {original_ratio}"

    def test_correct_split_sizes(self, classification_dataset):
        """Verify split sizes match requested proportions."""
        X, y = classification_dataset
        n = len(X)

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
        )

        # Allow ±2 samples tolerance for rounding
        assert abs(len(X_train) - int(n * 0.8)) <= 2
        assert abs(len(X_val) - int(n * 0.1)) <= 2
        assert abs(len(X_test) - int(n * 0.1)) <= 2

        # Total should equal original
        assert len(X_train) + len(X_val) + len(X_test) == n

    def test_raises_on_invalid_sizes(self, classification_dataset):
        """Verify ValueError raised when sizes don't sum to 1.0."""
        X, y = classification_dataset

        with pytest.raises(ValueError) as excinfo:
            split_train_val_test(
                X, y, train_size=0.7, val_size=0.1, test_size=0.1, random_state=42
            )

        assert "sum to 1.0" in str(excinfo.value)

    def test_reproducibility_with_same_seed(self, classification_dataset):
        """Verify same seed produces identical splits."""
        X, y = classification_dataset

        split1 = split_train_val_test(X, y, 0.8, 0.1, 0.1, random_state=42)
        split2 = split_train_val_test(X, y, 0.8, 0.1, 0.1, random_state=42)

        X_train1, _, _, y_train1, _, _ = split1
        X_train2, _, _, y_train2, _, _ = split2

        pd.testing.assert_frame_equal(X_train1.reset_index(drop=True), X_train2.reset_index(drop=True))
        pd.testing.assert_series_equal(y_train1.reset_index(drop=True), y_train2.reset_index(drop=True))

    def test_different_seeds_produce_different_splits(self, classification_dataset):
        """Verify different seeds produce different splits."""
        X, y = classification_dataset

        split1 = split_train_val_test(X, y, 0.8, 0.1, 0.1, random_state=42)
        split2 = split_train_val_test(X, y, 0.8, 0.1, 0.1, random_state=123)

        X_train1, _, _, _, _, _ = split1
        X_train2, _, _, _, _, _ = split2

        # Index should differ
        assert not X_train1.index.equals(X_train2.index)

    def test_no_data_leakage_between_splits(self, classification_dataset):
        """Verify no overlap between train/val/test sets."""
        X, y = classification_dataset

        X_train, X_val, X_test, _, _, _ = split_train_val_test(
            X, y, 0.8, 0.1, 0.1, random_state=42
        )

        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        assert train_idx.isdisjoint(val_idx), "Train and val sets overlap"
        assert train_idx.isdisjoint(test_idx), "Train and test sets overlap"
        assert val_idx.isdisjoint(test_idx), "Val and test sets overlap"


class TestDatasetHash:
    """Test suite for dataset_hash() function."""

    def test_deterministic_hash(self):
        """Verify same data produces same hash."""
        df = pd.DataFrame({
            "duration (ms)": [210000],
            "danceability": [0.7],
            "energy": [0.8],
            "loudness": [-5.0],
            "speechiness": [0.05],
            "acousticness": [0.2],
            "instrumentalness": [0.1],
            "liveness": [0.15],
            "valence": [0.6],
            "tempo": [120.0],
            "labels": [0],
        })

        hash1 = dataset_hash(df, FEATURE_COLS, TARGET_COL)
        hash2 = dataset_hash(df, FEATURE_COLS, TARGET_COL)

        assert hash1 == hash2

    def test_different_data_different_hash(self):
        """Verify different data produces different hash."""
        df1 = pd.DataFrame({
            "duration (ms)": [210000],
            "danceability": [0.7],
            "energy": [0.8],
            "loudness": [-5.0],
            "speechiness": [0.05],
            "acousticness": [0.2],
            "instrumentalness": [0.1],
            "liveness": [0.15],
            "valence": [0.6],
            "tempo": [120.0],
            "labels": [0],
        })

        df2 = df1.copy()
        df2["labels"] = [1]  # Change label

        hash1 = dataset_hash(df1, FEATURE_COLS, TARGET_COL)
        hash2 = dataset_hash(df2, FEATURE_COLS, TARGET_COL)

        assert hash1 != hash2
