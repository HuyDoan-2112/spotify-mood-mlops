import pandas as pd

MODE_WEIGHTS = {
    "popular": (0.55, 0.20, 0.25),
    "trending": (0.45, 0.40, 0.15),
    "discovery": (0.70, 0.20, 0.10),
}


def recommend_df(
    df_latest: pd.DataFrame,
    mood_idx: int,
    k: int = 20,
    country: str | None = None,
    allow_explicit: bool = True,
    min_conf: float = 0.0,
    diversify_artist: bool = True,
    max_per_artist: int = 2,
    mode: str = "popular",
) -> pd.DataFrame:
    w_mood, w_trend, w_pop = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["popular"])
    d = df_latest.copy()

    if country:
        d = d[d["country"] == str(country).strip().upper()]

    if not allow_explicit:
        d = d[d["is_explicit"] == False]

    if min_conf > 0:
        d = d[d["mood_conf"] >= min_conf]

    pcol = f"P_{mood_idx}"
    if pcol not in d.columns:
        raise KeyError(f"Missing {pcol} in df_latest")

    d["score"] = w_mood * d[pcol] + w_trend * d["trend_score"] + w_pop * d["pop_norm"]
    d = d.sort_values("score", ascending=False)

    if diversify_artist:
        out, counts = [], {}
        for _, row in d.iterrows():
            a = row["artists"]
            counts[a] = counts.get(a, 0)
            if counts[a] < max_per_artist:
                out.append(row)
                counts[a] += 1
            if len(out) >= k:
                break
        d = pd.DataFrame(out)
    else:
        d = d.head(k)

    out_cols = [
        "spotify_id",
        "name",
        "artists",
        "country",
        "score",
        "popularity",
        "daily_rank",
        "mood_conf",
        pcol,
    ]
    for c in out_cols:
        if c not in d.columns:
            d[c] = None
    return d[out_cols].reset_index(drop=True)
