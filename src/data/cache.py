from pathlib import Path
import json
import os
import pandas as pd
from src.data.loader import load_dataset, clean_dataset, add_pop_norm, add_mood_scores, build_latest_df
from src.recommender.trends import compute_trend_score

_DF_CACHE = {}
_LATEST_DATE = {}


def _data_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    env_path = os.getenv("DATA_B_PATH")
    return Path(env_path) if env_path else (root / "data/processed/filtered_countries.csv")

def _cache_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    env_dir = os.getenv("DF_LATEST_CACHE_DIR")
    return Path(env_dir) if env_dir else (root / "data/processed")


def _cache_paths(model_name: str):
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"df_latest_{model_name}.csv"
    meta_path = cache_dir / f"df_latest_{model_name}.meta.json"
    return data_path, meta_path


def _source_mtime() -> float:
    return _data_path().stat().st_mtime


def _load_cached_latest(model_name: str):
    data_path, meta_path = _cache_paths(model_name)
    if not data_path.exists() or not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("source_mtime") != _source_mtime():
        return None, None
    latest = pd.read_csv(data_path)
    return latest, meta.get("latest_date")


def _save_cached_latest(model_name: str, latest: pd.DataFrame, latest_date):
    data_path, meta_path = _cache_paths(model_name)
    latest.to_csv(data_path, index=False)
    meta = {"source_mtime": _source_mtime(), "latest_date": str(latest_date)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def get_df_latest(model_name: str, reload: bool = False):
    if reload or model_name not in _DF_CACHE:
        if not reload:
            cached, cached_date = _load_cached_latest(model_name)
            if cached is not None:
                _DF_CACHE[model_name] = cached
                _LATEST_DATE[model_name] = cached_date
                return _DF_CACHE[model_name]
        df = load_dataset(_data_path())
        df = clean_dataset(df)
        df = add_pop_norm(df)
        df = compute_trend_score(df)
        df = add_mood_scores(df, model_name)
        latest, latest_date = build_latest_df(df)
        _save_cached_latest(model_name, latest, latest_date)
        _DF_CACHE[model_name] = latest
        _LATEST_DATE[model_name] = latest_date
    return _DF_CACHE[model_name]


def get_latest_date(model_name: str):
    return _LATEST_DATE.get(model_name)
