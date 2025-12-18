"""
Streamlit UI to send records to a trained model.
Run with: `streamlit run app/streamlit_app.py`
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src import config


@st.cache_resource
def load_model(model_path: Path) -> object:
    return joblib.load(model_path)


st.set_page_config(page_title="ML Project NTI", layout="wide")
st.title("ML Project NTI")
st.caption("Simple interface for running predictions.")

model_path = st.text_input("Model path", value=str(config.MODELS_DIR / "binary_model.joblib"))
file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
text_input = st.text_area("Or paste JSON records (list of objects)", height=150)

run = st.button("Predict")

if run:
    try:
        model = load_model(Path(model_path))
    except FileNotFoundError:
        st.error(f"Model not found at {model_path}")
        st.stop()

    df: pd.DataFrame
    if file is not None:
        df = pd.read_csv(file)
    else:
        try:
            records = json.loads(text_input or "[]")
            df = pd.DataFrame(records)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            st.stop()

    if df.empty:
        st.warning("No data to predict on.")
    else:
        preds = model.predict(df)
        st.subheader("Predictions")
        st.dataframe(pd.DataFrame({"prediction": preds}))
