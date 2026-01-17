from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "278k_song_labelled.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = REPORTS_DIR / "experiments"

DATASET_NAME = "Moodify_278k"
VN_POOL_RAW = PROJECT_ROOT / "data" / "raw" / "vn_spotify.csv"
VN_POOL_PROCESSED = PROJECT_ROOT / "data" / "processed" / "vn_pool.parquet"

TARGET_COL = "labels"
FEATURE_COLS = [
    "duration (ms)", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
ENGINEERED_FEATURES = ["intensity", "rhythm_drive", "calm_score"]

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 42

USE_FEATURE_ENGINEERING = False
EPOCHS = 25
BATCH_SIZE = 1024
VAL_BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4

MODEL_NAMES = ["lr", "rf", "xgb", "lgbm", "mlp"]
PREFERRED_MODEL = "lgbm"
