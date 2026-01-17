import numpy as np
import pandas as pd

def _minmax_norm(x: pd.Series) -> pd.Series:
    if x.notna().any():
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-6)
    return pd.Series([0.5] * len(x), index=x.index)


def compute_trend_score(df: pd.DataFrame, window: int = 30, min_pts: int = 7) -> pd.DataFrame:
    d = df.copy()
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"], errors="coerce")
    # Multi-day => rolling slope; single-day => movement-based fallback.
    if d["snapshot_date"].nunique() <= 1:
        daily_col = d["daily_movement"] if "daily_movement" in d.columns else pd.Series(0, index=d.index)
        weekly_col = d["weekly_movement"] if "weekly_movement" in d.columns else pd.Series(0, index=d.index)
        daily = pd.to_numeric(daily_col, errors="coerce").fillna(0)
        weekly = pd.to_numeric(weekly_col, errors="coerce").fillna(0)
        d["trend_raw"] = 0.7 * daily + 0.3 * weekly
        d["trend_score"] = d.groupby("country")["trend_raw"].transform(_minmax_norm).fillna(0.5)
        d["rank_roll_mean"] = np.nan
        d["rank_trend_slope"] = np.nan
        return d
    d = d.sort_values(["country","spotify_id","snapshot_date"])
    def _rolling_slope(arr: np.ndarray) -> float:
        if len(arr) < min_pts:
            return np.nan
        t = np.arange(len(arr), dtype=float)
        return np.polyfit(t, arr.astype(float), 1)[0]
    d["rank_roll_mean"] = d.groupby(["country","spotify_id"])["daily_rank"].transform(
        lambda x: x.rolling(window, min_periods=min_pts).mean()
    )
    d["rank_trend_slope"] = d.groupby(["country","spotify_id"])["rank_roll_mean"].transform(
        lambda x: x.rolling(window, min_periods=min_pts).apply(lambda w: _rolling_slope(w.values), raw=False)
    )
    d["trend_raw"] = -d["rank_trend_slope"]
    d["trend_score"] = d.groupby("country")["trend_raw"].transform(_minmax_norm).fillna(0.5)
    return d
