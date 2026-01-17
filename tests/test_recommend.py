"""T5: Unit tests for recommendation engine (src/recommender/recommend.py)."""
import pandas as pd
import numpy as np
import pytest

from src.recommender.recommend import recommend_df, MODE_WEIGHTS


class TestRecommendDF:
    """T5: Test suite for recommend_df() function."""

    def test_filters_by_country(self, recommendation_df):
        """T5: Verify country filter returns only matching country."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country="US",
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
        )

        assert len(result) > 0
        assert (result["country"] == "US").all()
        assert "VN" not in result["country"].values

    def test_filters_explicit_content(self, recommendation_df):
        """T5: Verify explicit filter removes explicit songs."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country="US",
            allow_explicit=False,  # No explicit
            min_conf=0.0,
            diversify_artist=False,
        )

        # id1 is US+explicit, id2 is US+non-explicit, id3 is US+non-explicit
        # Should return id2 and id3 only
        assert "id1" not in result["spotify_id"].values
        assert "id2" in result["spotify_id"].values or "id3" in result["spotify_id"].values

    def test_combined_country_and_explicit_filter(self, recommendation_df):
        """T5: Verify combined filters work correctly."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country="US",
            allow_explicit=False,
            min_conf=0.0,
            diversify_artist=False,
        )

        # Should only have US + non-explicit songs
        assert len(result) > 0
        assert (result["country"] == "US").all()

        # Verify no explicit songs in result
        # Need to check original df for is_explicit since it may not be in output
        for spotify_id in result["spotify_id"]:
            original_row = recommendation_df[recommendation_df["spotify_id"] == spotify_id]
            assert original_row["is_explicit"].iloc[0] == False

    def test_min_confidence_filter(self, recommendation_df):
        """Verify min_conf filter removes low-confidence predictions."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country=None,
            allow_explicit=True,
            min_conf=0.7,  # Only conf >= 0.7
            diversify_artist=False,
        )

        # Only id1 (0.9) and id2 (0.8) have mood_conf >= 0.7
        assert all(recommendation_df.loc[
            recommendation_df["spotify_id"].isin(result["spotify_id"]),
            "mood_conf"
        ] >= 0.7)

    def test_respects_k_limit(self, recommendation_df):
        """Verify returns at most k results."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=2,
            country=None,
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
        )

        assert len(result) <= 2

    def test_artist_diversification(self, recommendation_df):
        """Verify diversify_artist limits songs per artist."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country=None,
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=True,
            max_per_artist=1,
        )

        # Each artist should appear at most once
        artist_counts = result["artists"].value_counts()
        assert artist_counts.max() <= 1

    def test_diversification_max_per_artist_2(self, recommendation_df):
        """Verify max_per_artist=2 allows 2 songs per artist."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country=None,
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=True,
            max_per_artist=2,
        )

        artist_counts = result["artists"].value_counts()
        assert artist_counts.max() <= 2

    def test_score_calculation_popular_mode(self, recommendation_df):
        """Verify score uses correct weights for 'popular' mode."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country=None,
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
            mode="popular",
        )

        assert "score" in result.columns

        # Verify score is calculated (non-zero for valid data)
        assert result["score"].notna().all()

        # popular mode: (0.55, 0.20, 0.25) for mood, trend, pop
        w_mood, w_trend, w_pop = MODE_WEIGHTS["popular"]
        assert w_mood == 0.55
        assert w_trend == 0.20
        assert w_pop == 0.25

    def test_score_calculation_discovery_mode(self, recommendation_df):
        """Verify discovery mode emphasizes mood over popularity."""
        # discovery mode: (0.70, 0.20, 0.10)
        w_mood, w_trend, w_pop = MODE_WEIGHTS["discovery"]
        assert w_mood == 0.70
        assert w_pop == 0.10

    def test_results_sorted_by_score_descending(self, recommendation_df):
        """Verify results are sorted by score in descending order."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country=None,
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
        )

        scores = result["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_output_columns(self, recommendation_df):
        """Verify output contains expected columns."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country=None,
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
        )

        expected_cols = [
            "spotify_id", "name", "artists", "country", "score",
            "popularity", "daily_rank", "mood_conf"
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_raises_on_missing_mood_probability(self):
        """Verify KeyError raised when P_{mood_idx} column is missing."""
        df = pd.DataFrame({
            "spotify_id": ["id1"],
            "name": ["Song"],
            "artists": ["Artist"],
            "country": ["US"],
            "is_explicit": [False],
            "mood_conf": [0.9],
            # Missing P_0, P_1, etc.
            "trend_score": [0.5],
            "pop_norm": [0.5],
        })

        with pytest.raises(KeyError) as excinfo:
            recommend_df(df, mood_idx=0, k=10)

        assert "P_0" in str(excinfo.value)

    def test_empty_result_after_filtering(self, recommendation_df):
        """Verify empty DataFrame returned when all songs filtered out."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country="XX",  # Non-existent country
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
        )

        assert len(result) == 0

    def test_country_case_insensitive(self, recommendation_df):
        """Verify country filter is case-insensitive (uppercased)."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country="us",  # lowercase
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
        )

        # Should match "US" in data
        assert len(result) > 0
        assert (result["country"] == "US").all()

    def test_no_country_filter_returns_all_countries(self, recommendation_df):
        """Verify country=None returns songs from all countries."""
        result = recommend_df(
            recommendation_df,
            mood_idx=0,
            k=10,
            country=None,
            allow_explicit=True,
            min_conf=0.0,
            diversify_artist=False,
        )

        # Should have both US and VN songs
        countries = result["country"].unique()
        assert len(countries) > 1 or len(result) == len(recommendation_df)
