import pandas as pd
import streamlit as st
from src.data.cache import get_df_latest, get_latest_date
from src.models.registry import list_models, predict_proba_df
from src.recommender.recommend import recommend_df

st.set_page_config(page_title="Mood Recs", layout="wide")
st.title("Spotify Mood Tester")

models = list_models()
if not models:
    st.error("No models found. Check MODEL_ROOT and artifacts.")
    st.stop()

model_name = st.sidebar.selectbox("Model", models)
reload_data = st.sidebar.button("Reload data cache")

try:
    df_latest = get_df_latest(model_name, reload=reload_data)
except Exception as exc:
    st.error(f"Data/model load failed: {exc}")
    st.stop()

latest_date = get_latest_date(model_name)
if latest_date is not None:
    st.sidebar.caption(f"Latest date: {latest_date}")

countries = ["ALL"] + sorted(df_latest["country"].dropna().unique().tolist())

st.sidebar.subheader("Recommend")
mood_idx = st.sidebar.selectbox("Mood index", [0, 1, 2, 3])
country = st.sidebar.selectbox("Country", countries)
allow_explicit = st.sidebar.checkbox("Allow explicit", True)
diversify = st.sidebar.checkbox("Diversify artists", True)
min_conf = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.2, 0.01)
k = st.sidebar.slider("Top K", 1, 50, 20)
mode = st.sidebar.selectbox("Mode", ["popular", "trending", "discovery"])

st.subheader("Model playground")
song = {
    "duration_ms": st.number_input("duration_ms", value=210000),
    "danceability": st.slider("danceability", 0.0, 1.0, 0.5),
    "energy": st.slider("energy", 0.0, 1.0, 0.6),
    "loudness": st.number_input("loudness", value=-7.0),
    "speechiness": st.slider("speechiness", 0.0, 1.0, 0.05),
    "acousticness": st.slider("acousticness", 0.0, 1.0, 0.1),
    "instrumentalness": st.slider("instrumentalness", 0.0, 1.0, 0.0),
    "liveness": st.slider("liveness", 0.0, 1.0, 0.1),
    "valence": st.slider("valence", 0.0, 1.0, 0.5),
    "tempo": st.number_input("tempo", value=120.0),
}


def infer_one(name: str) -> dict:
    df = pd.DataFrame([song])
    proba, classes = predict_proba_df(df, name)
    row = {f"P_{c}": float(proba[0, i]) for i, c in enumerate(classes)}
    row["mood_pred"] = int(classes[int(proba[0].argmax())])
    row["mood_conf"] = float(proba[0].max())
    row["model"] = name
    return row


col1, col2 = st.columns(2)
with col1:
    if st.button("Infer selected model"):
        st.dataframe([infer_one(model_name)], use_container_width=True)
with col2:
    if st.button("Compare all models"):
        rows = [infer_one(m) for m in models]
        st.dataframe(rows, use_container_width=True)

payload = {
    "mood_idx": mood_idx,
    "k": k,
    "country": None if country == "ALL" else country,
    "allow_explicit": allow_explicit,
    "min_conf": min_conf,
    "mode": mode,
    "diversify_artist": diversify,
}

if st.button("Recommend"):
    recs = recommend_df(
        df_latest,
        payload["mood_idx"],
        payload["k"],
        payload["country"],
        payload["allow_explicit"],
        payload["min_conf"],
        payload["diversify_artist"],
        mode=payload["mode"],
    )
    pcol = f"P_{mood_idx}"
    if pcol in recs.columns:
        recs = recs.rename(columns={pcol: "P_target_mood"})
    st.dataframe(recs, use_container_width=True)
