from pathlib import Path
import json
import pandas as pd


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj
    clean = _convert(data)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")

def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def save_confusion_matrix(path: Path, cm, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df.to_csv(path, index=True)

def save_metadata(path: Path, cfg, model_name, classes, dataset_hash, extra=None):
    payload = {
        "model_name": model_name,
        "dataset_name": cfg.DATASET_NAME,
        "dataset_hash": dataset_hash,
        "feature_cols": cfg.FEATURE_COLS,
        "engineered_features": cfg.ENGINEERED_FEATURES,
        "target_col": cfg.TARGET_COL,
        "classes": [int(c) for c in classes],
        "random_state": cfg.RANDOM_STATE,
        "split_sizes": {
            "train": cfg.TRAIN_SIZE,
            "val": cfg.VAL_SIZE,
            "test": cfg.TEST_SIZE,
        },
    }
    if extra:
        payload.update(extra)
    save_json(path, payload)

def save_leaderboard(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
