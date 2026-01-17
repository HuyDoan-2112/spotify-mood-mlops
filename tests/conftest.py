"""Pytest configuration and shared fixtures."""
import pandas as pd
import numpy as np
import pytest

from src.config import FEATURE_COLS


@pytest.fixture
def sample_feature_row():
    """Single row with all FEATURE_COLS for testing feature engineering."""
    return {
        "duration (ms)": 210000,
        "danceability": 0.7,
        "energy": 0.8,
        "loudness": -5.0,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.1,
        "liveness": 0.15,
        "valence": 0.6,
        "tempo": 120.0,
    }


@pytest.fixture
def sample_features_df(sample_feature_row):
    """DataFrame with a single row of audio features."""
    return pd.DataFrame([sample_feature_row])


@pytest.fixture
def multi_row_features_df():
    """DataFrame with multiple rows for batch testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "duration (ms)": np.random.randint(60000, 300000, n),
        "danceability": np.random.uniform(0, 1, n),
        "energy": np.random.uniform(0, 1, n),
        "loudness": np.random.uniform(-60, 0, n),
        "speechiness": np.random.uniform(0, 1, n),
        "acousticness": np.random.uniform(0, 1, n),
        "instrumentalness": np.random.uniform(0, 1, n),
        "liveness": np.random.uniform(0, 1, n),
        "valence": np.random.uniform(0, 1, n),
        "tempo": np.random.uniform(60, 200, n),
    })


@pytest.fixture
def classification_dataset():
    """Synthetic dataset for classification with imbalanced classes (80/20)."""
    np.random.seed(42)
    n = 100

    # Create features
    X = pd.DataFrame({
        "duration (ms)": np.random.randint(60000, 300000, n),
        "danceability": np.random.uniform(0, 1, n),
        "energy": np.random.uniform(0, 1, n),
        "loudness": np.random.uniform(-60, 0, n),
        "speechiness": np.random.uniform(0, 1, n),
        "acousticness": np.random.uniform(0, 1, n),
        "instrumentalness": np.random.uniform(0, 1, n),
        "liveness": np.random.uniform(0, 1, n),
        "valence": np.random.uniform(0, 1, n),
        "tempo": np.random.uniform(60, 200, n),
    })

    # Create imbalanced labels: 80% class 0, 20% class 1
    y = pd.Series([0] * 80 + [1] * 20)

    return X, y


@pytest.fixture
def four_class_dataset():
    """Synthetic dataset with 4 balanced classes for mood classification."""
    np.random.seed(42)
    n = 400  # 100 per class

    X = pd.DataFrame({
        "duration (ms)": np.random.randint(60000, 300000, n),
        "danceability": np.random.uniform(0, 1, n),
        "energy": np.random.uniform(0, 1, n),
        "loudness": np.random.uniform(-60, 0, n),
        "speechiness": np.random.uniform(0, 1, n),
        "acousticness": np.random.uniform(0, 1, n),
        "instrumentalness": np.random.uniform(0, 1, n),
        "liveness": np.random.uniform(0, 1, n),
        "valence": np.random.uniform(0, 1, n),
        "tempo": np.random.uniform(60, 200, n),
    })

    y = pd.Series([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100)

    return X, y


@pytest.fixture
def recommendation_df():
    """DataFrame mimicking the structure needed for recommend_df()."""
    return pd.DataFrame({
        "spotify_id": ["id1", "id2", "id3", "id4", "id5"],
        "name": ["Song A", "Song B", "Song C", "Song D", "Song E"],
        "artists": ["Artist 1", "Artist 1", "Artist 2", "Artist 3", "Artist 3"],
        "country": ["US", "US", "US", "VN", "VN"],
        "is_explicit": [True, False, False, False, True],
        "popularity": [80, 70, 60, 50, 40],
        "daily_rank": [1, 2, 3, 1, 2],
        "mood_conf": [0.9, 0.8, 0.7, 0.6, 0.5],
        "P_0": [0.9, 0.1, 0.2, 0.3, 0.4],
        "P_1": [0.05, 0.8, 0.3, 0.2, 0.1],
        "P_2": [0.03, 0.05, 0.4, 0.3, 0.2],
        "P_3": [0.02, 0.05, 0.1, 0.2, 0.3],
        "trend_score": [0.8, 0.7, 0.6, 0.5, 0.4],
        "pop_norm": [1.0, 0.8, 0.6, 0.5, 0.3],
    })


@pytest.fixture
def trend_single_day_df():
    """DataFrame with single snapshot_date for trend score fallback testing."""
    return pd.DataFrame({
        "spotify_id": ["id1", "id2", "id3"],
        "country": ["US", "US", "VN"],
        "snapshot_date": ["2024-01-15", "2024-01-15", "2024-01-15"],
        "daily_rank": [1, 2, 1],
        "daily_movement": [5, -2, 3],
        "weekly_movement": [2, -1, 1],
    })
