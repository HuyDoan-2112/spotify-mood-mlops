from pathlib import Path
import os
import json
import joblib
import pandas as pd
from src.config import FEATURE_COLS
from src.components.features import add_features


_CACHE = {}
MODEL_ORDER = ["lr", "rf", "xgb", "lgbm", "mlp"]


def _model_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    env_root = os.getenv("MODEL_ROOT")
    return Path(env_root) if env_root else (root / "artifacts/models")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "duration_ms" in d.columns and "duration (ms)" not in d.columns:
        d = d.rename(columns={"duration_ms": "duration (ms)"})
    missing = [c for c in FEATURE_COLS if c not in d.columns]
    if missing:
        raise ValueError(f"Missing base columns: {missing}")
    return d[FEATURE_COLS].copy()

def list_models() -> list[str]:
    root = _model_root()
    out = []
    for name in MODEL_ORDER:
        if (root/name/"model.joblib").exists() or (root/name/"model.pt").exists():
            out.append(name)
    return out

def _load_mlp(name: str):
    import torch
    from src.components.models import build_mlp
    root = _model_root()/name
    classes = json.loads((root/"classes.json").read_text(encoding="utf-8"))
    scaler = joblib.load(root/"scaler.joblib")
    model = build_mlp(scaler.mean_.shape[0], len(classes))
    state = torch.load(root/"model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return {"type":"mlp","model":model,"scaler":scaler,"classes":classes}

def get_model(name: str):
    if name in _CACHE:
        return _CACHE[name]
    root = _model_root()/name
    if (root/"model.joblib").exists():
        _CACHE[name] = {"type":"sklearn","model":joblib.load(root/"model.joblib")}
    elif (root/"model.pt").exists():
        _CACHE[name] = _load_mlp(name)
    else:
        raise FileNotFoundError(f"Model not found: {name}")
    return _CACHE[name]

def predict_proba_df(df: pd.DataFrame, model_name: str):
    info = get_model(model_name)
    X = prepare_features(df)
    if info["type"] == "sklearn":
        proba = info["model"].predict_proba(X)
        classes = [int(c) for c in info["model"].classes_]
        return proba, classes
    import torch
    X = add_features(X, FEATURE_COLS)
    Xs = info["scaler"].transform(X)
    with torch.no_grad():
        logits = info["model"](torch.tensor(Xs, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).numpy()
    return proba, info["classes"]
