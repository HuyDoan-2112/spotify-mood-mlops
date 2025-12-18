"""
FastAPI stub for serving a trained model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config


class PredictionRequest(BaseModel):
    records: List[dict[str, Any]]


app = FastAPI(title="ML Project NTI API")


def load_model(model_path: Path | None = None):
    path = model_path or config.MODELS_DIR / "binary_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


@app.get("/")
def healthcheck():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest, model_path: str | None = None):
    try:
        model = load_model(Path(model_path) if model_path else None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    df = pd.DataFrame(payload.records)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
