# Spotify Mood MLOps

Train and compare multiple mood classifiers on the Moodify dataset, save artifacts and reports, and prepare outputs for downstream recommendation and UI layers.

## Project layout

```
artifacts/          # Model binaries and training outputs (ignored by git)
data/               # Raw and processed datasets
notebooks/          # Exploratory analysis
reports/            # Reports and experiment outputs (experiments ignored by git)
src/
  api/              # API entry points (optional)
  components/       # Reusable blocks (data loader, features, models)
  engine/           # Training, evaluation, prediction pipeline
  utils/            # IO helpers
  config.py         # Central configuration
```

## Requirements

- Python 3.12.5
- See `requirements.txt`

## Quickstart

1) Create and activate a virtual environment.
2) Install deps:

```
pip install -r requirements.txt
```

3) Put the Moodify CSV in `data/raw/278k_song_labelled.csv` (update `RAW_DATA_PATH` in `src/config.py` if needed).

4) Train all models (LR/RF/XGB/LGBM/MLP):

```
python -m src.engine.train --model all --run-id all_models
```

## Outputs

Training writes to:

- `artifacts/<run_id>/<model>/` (model artifacts)
- `reports/experiments/<run_id>/metrics/` (metrics JSON)
- `reports/experiments/<run_id>/reports/` (classification reports)
- `reports/experiments/<run_id>/confusion/` (confusion matrices)
- `reports/experiments/<run_id>/leaderboard.csv`

## Configuration

All configuration lives in `src/config.py`:

- feature list and target column
- train/val/test split sizes
- random seed
- artifact and report paths
- preferred model (default: LGBM)

## Notes

- Feature order must be consistent between training and inference.
- If you change the feature list, retrain all models.

## License

Add your license here.
