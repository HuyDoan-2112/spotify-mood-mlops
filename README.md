# ML Project NTI

Lightweight scaffold for a multi-problem ML workflow (binary, multiclass, regression) with training, evaluation, reporting, and simple API/UI endpoints.

## Layout
- `data/` — raw and processed data placeholders (gitignored).
- `notebooks/EDA.ipynb` — exploratory analysis starter notebook.
- `src/` — core Python modules (config, validation, preprocessing, features, training scripts, evaluation, recommendation).
- `models/` — serialized artifacts (gitignored).
- `api/main.py` — FastAPI service stub.
- `app/streamlit_app.py` — Streamlit interface stub.
- `reports/report_outline.md` — reporting scaffold.
- `requirements.txt` — Python dependencies.

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Place data into `data/raw/`; transform via your preprocessing pipeline into `data/processed/`.
4. Run a training script, e.g. `python src/train_binary.py`.
5. Serve the API: `uvicorn api.main:app --reload`
6. Launch the UI: `streamlit run app/streamlit_app.py`

## Notes
- Update `src/config.py` with your paths and experiment settings.
- Keep artifacts out of version control (already covered in `.gitignore`).
