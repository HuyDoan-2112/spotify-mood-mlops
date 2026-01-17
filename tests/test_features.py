"""T1: Unit tests for feature engineering (src/components/features.py)."""
import pandas as pd
import numpy as np
import pytest

from src.components.features import add_features
from src.config import FEATURE_COLS


class TestAddFeatures:
    """Test suite for add_features() function."""

    def test_intensity_calculation(self, sample_features_df):
        """T1: Verify intensity = energy * (-loudness)."""
        result = add_features(sample_features_df, FEATURE_COLS)

        # energy=0.8, loudness=-5.0 => intensity = 0.8 * 5.0 = 4.0
        expected_intensity = 0.8 * (-(-5.0))
        assert "intensity" in result.columns
        assert np.isclose(result["intensity"].iloc[0], expected_intensity)

    def test_rhythm_drive_calculation(self, sample_features_df):
        """T1: Verify rhythm_drive = danceability * tempo."""
        result = add_features(sample_features_df, FEATURE_COLS)

        # danceability=0.7, tempo=120.0 => rhythm_drive = 84.0
        expected_rhythm_drive = 0.7 * 120.0
        assert "rhythm_drive" in result.columns
        assert np.isclose(result["rhythm_drive"].iloc[0], expected_rhythm_drive)

    def test_calm_score_calculation(self, sample_features_df):
        """T1: Verify calm_score = acousticness + instrumentalness."""
        result = add_features(sample_features_df, FEATURE_COLS)

        # acousticness=0.2, instrumentalness=0.1 => calm_score = 0.3
        expected_calm_score = 0.2 + 0.1
        assert "calm_score" in result.columns
        assert np.isclose(result["calm_score"].iloc[0], expected_calm_score)

    def test_all_engineered_features_present(self, sample_features_df):
        """Verify all three engineered features are added."""
        result = add_features(sample_features_df, FEATURE_COLS)

        expected_features = ["intensity", "rhythm_drive", "calm_score"]
        for feat in expected_features:
            assert feat in result.columns, f"Missing engineered feature: {feat}"

    def test_original_features_preserved(self, sample_features_df):
        """Verify original FEATURE_COLS are preserved in output."""
        result = add_features(sample_features_df, FEATURE_COLS)

        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing original feature: {col}"

    def test_handles_numpy_array_input(self, sample_features_df):
        """Verify function handles numpy array input (sklearn pipeline compatibility)."""
        X_array = sample_features_df[FEATURE_COLS].values
        result = add_features(X_array, FEATURE_COLS)

        assert isinstance(result, pd.DataFrame)
        assert "intensity" in result.columns
        assert len(result) == 1

    def test_batch_processing(self, multi_row_features_df):
        """Verify function handles multiple rows correctly."""
        result = add_features(multi_row_features_df, FEATURE_COLS)

        assert len(result) == len(multi_row_features_df)
        assert "intensity" in result.columns
        assert "rhythm_drive" in result.columns
        assert "calm_score" in result.columns

    def test_edge_case_zero_energy(self):
        """Verify intensity is zero when energy is zero."""
        df = pd.DataFrame([{
            "duration (ms)": 210000,
            "danceability": 0.5,
            "energy": 0.0,  # Zero energy
            "loudness": -10.0,
            "speechiness": 0.05,
            "acousticness": 0.2,
            "instrumentalness": 0.1,
            "liveness": 0.15,
            "valence": 0.6,
            "tempo": 120.0,
        }])
        result = add_features(df, FEATURE_COLS)
        assert result["intensity"].iloc[0] == 0.0

    def test_edge_case_zero_loudness(self):
        """Verify intensity is zero when loudness is zero."""
        df = pd.DataFrame([{
            "duration (ms)": 210000,
            "danceability": 0.5,
            "energy": 0.8,
            "loudness": 0.0,  # Zero loudness (unusual but possible)
            "speechiness": 0.05,
            "acousticness": 0.2,
            "instrumentalness": 0.1,
            "liveness": 0.15,
            "valence": 0.6,
            "tempo": 120.0,
        }])
        result = add_features(df, FEATURE_COLS)
        assert result["intensity"].iloc[0] == 0.0
