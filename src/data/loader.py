from pathlib import Path
import pandas as pd
from src.models.registry import predict_proba_df


def load_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["country"] = d["country"].astype(str).str.strip().str.upper()
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"], errors="coerce")
    if "duration_ms" in d.columns and "duration (ms)" not in d.columns:
        d = d.rename(columns={"duration_ms": "duration (ms)"})
    num_cols = [
        "daily_rank",
        "daily_movement",
        "weekly_movement",
        "popularity",
        "loudness",
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration (ms)",
    ]
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["spotify_id", "snapshot_date", "popularity", "daily_rank"])


def add_pop_norm(d: pd.DataFrame) -> pd.DataFrame:
    d["pop_norm"] = d.groupby("country")["popularity"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )
    d["pop_norm"] = d["pop_norm"].fillna(0.0)
    return d


def add_mood_scores(d: pd.DataFrame, model_name: str) -> pd.DataFrame:
    proba, classes = predict_proba_df(d, model_name)
    for i, c in enumerate(classes):
        d[f"P_{c}"] = proba[:, i]
    d["mood_pred"] = [classes[i] for i in proba.argmax(axis=1)]
    d["mood_conf"] = proba.max(axis=1)
    return d


def build_latest_df(d: pd.DataFrame):
    latest_date = d["snapshot_date"].max()
    latest = d[d["snapshot_date"] == latest_date].copy()
    latest = latest.sort_values(["country", "spotify_id", "daily_rank"]).drop_duplicates(
        subset=["country", "spotify_id"], keep="first"
    )
    return latest, latest_date
