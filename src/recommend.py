"""
Simple batch prediction utility to generate recommendations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from . import config


def load_inputs(filename: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED / filename
    return pd.read_csv(path)


def predict(model_path: Path, input_file: str, output_file: str | None = None) -> Path | None:
    model = joblib.load(model_path)
    df = load_inputs(input_file)
    preds = model.predict(df)

    out_path = Path(output_file) if output_file else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.copy()
        df_out["prediction"] = preds
        df_out.to_csv(out_path, index=False)
        print(f"Wrote predictions to {out_path}")
    else:
        print(preds)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions with a trained model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to joblib model.")
    parser.add_argument("--data", required=True, help="Processed CSV filename in data/processed/")
    parser.add_argument("--output", help="Optional path to save predictions CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.model, args.data, args.output)
