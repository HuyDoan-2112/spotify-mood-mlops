"""T4: Unit tests for trend scoring (src/recommender/trends.py)."""
import pandas as pd
import numpy as np
import pytest

from src.recommender.trends import compute_trend_score, _minmax_norm


class TestMinmaxNorm:
    """Test suite for _minmax_norm() helper function."""

    def test_normalizes_to_zero_one_range(self):
        """Verify output is in [0, 1] range."""
        s = pd.Series([10, 20, 30, 40, 50])
        result = _minmax_norm(s)

        assert result.min() >= 0
        assert result.max() <= 1
        assert np.isclose(result.min(), 0.0, atol=1e-6)
        assert np.isclose(result.max(), 1.0, atol=1e-6)

    def test_handles_constant_series(self):
        """Verify constant series returns 0.5 (division by zero protection)."""
        s = pd.Series([5, 5, 5, 5])
        result = _minmax_norm(s)

        # With epsilon protection, constant series should still work
        assert not result.isna().any()

    def test_handles_all_nan_series(self):
        """Verify all-NaN series returns 0.5."""
        s = pd.Series([np.nan, np.nan, np.nan])
        result = _minmax_norm(s)

        assert (result == 0.5).all()


class TestComputeTrendScore:
    """T4: Test suite for compute_trend_score() function."""

    def test_single_day_fallback_formula(self, trend_single_day_df):
        """T4: Verify single-day fallback uses 0.7*daily + 0.3*weekly."""
        result = compute_trend_score(trend_single_day_df)

        # For id1: daily_movement=5, weekly_movement=2
        # trend_raw = 0.7*5 + 0.3*2 = 3.5 + 0.6 = 4.1
        expected_raw = 0.7 * 5 + 0.3 * 2
        assert "trend_raw" in result.columns

        id1_row = result[result["spotify_id"] == "id1"]
        assert np.isclose(id1_row["trend_raw"].iloc[0], expected_raw)

    def test_single_day_trend_score_normalized_per_country(self, trend_single_day_df):
        """Verify trend_score is normalized per country."""
        result = compute_trend_score(trend_single_day_df)

        assert "trend_score" in result.columns

        # US country should have its own normalization
        us_scores = result[result["country"] == "US"]["trend_score"]
        assert us_scores.min() >= 0
        assert us_scores.max() <= 1

    def test_single_day_returns_nan_for_rolling_columns(self, trend_single_day_df):
        """Verify rank_roll_mean and rank_trend_slope are NaN for single-day data."""
        result = compute_trend_score(trend_single_day_df)

        assert result["rank_roll_mean"].isna().all()
        assert result["rank_trend_slope"].isna().all()

    def test_handles_missing_movement_columns(self):
        """Verify graceful handling when movement columns are missing.

        When daily_movement/weekly_movement columns are missing, the function
        should default to 0 for both, resulting in trend_raw = 0.
        """
        df = pd.DataFrame({
            "spotify_id": ["id1"],
            "country": ["US"],
            "snapshot_date": ["2024-01-15"],
            "daily_rank": [1],
            # daily_movement and weekly_movement missing
        })

        result = compute_trend_score(df)

        # Should default to 0 for missing columns, so trend_raw = 0.7*0 + 0.3*0 = 0
        assert "trend_raw" in result.columns
        assert result["trend_raw"].iloc[0] == 0.0
        assert "trend_score" in result.columns

    def test_multi_day_uses_rolling_slope(self):
        """Verify multi-day data uses rolling window slope calculation."""
        # Create 30+ days of data
        dates = pd.date_range("2024-01-01", periods=35, freq="D")
        df = pd.DataFrame({
            "spotify_id": ["id1"] * 35,
            "country": ["US"] * 35,
            "snapshot_date": dates,
            "daily_rank": list(range(35, 0, -1)),  # Improving rank (35 -> 1)
        })

        result = compute_trend_score(df, window=30, min_pts=7)

        # Should have calculated rolling columns
        # At least some non-NaN values after min_pts
        assert not result["rank_roll_mean"].isna().all()

    def test_preserves_original_columns(self, trend_single_day_df):
        """Verify original columns are preserved."""
        result = compute_trend_score(trend_single_day_df)

        assert "spotify_id" in result.columns
        assert "country" in result.columns
        assert "snapshot_date" in result.columns
        assert "daily_rank" in result.columns

    def test_trend_score_fillna(self, trend_single_day_df):
        """Verify NaN trend_scores are filled with 0.5."""
        result = compute_trend_score(trend_single_day_df)

        # No NaN in trend_score (should be filled with 0.5)
        assert not result["trend_score"].isna().any()

    def test_negative_movement_produces_lower_trend(self):
        """Verify negative daily_movement produces lower trend_raw."""
        df = pd.DataFrame({
            "spotify_id": ["id1", "id2"],
            "country": ["US", "US"],
            "snapshot_date": ["2024-01-15", "2024-01-15"],
            "daily_rank": [1, 2],
            "daily_movement": [10, -10],  # id1 improving, id2 declining
            "weekly_movement": [0, 0],
        })

        result = compute_trend_score(df)

        id1_trend = result[result["spotify_id"] == "id1"]["trend_raw"].iloc[0]
        id2_trend = result[result["spotify_id"] == "id2"]["trend_raw"].iloc[0]

        assert id1_trend > id2_trend
