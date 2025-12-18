"""
Centralized configuration for paths and common hyperparameters.
Adjust values to match your local environment and experiment needs.
"""
from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class TrainingConfig:
    target: str
    test_size: float = 0.2
    random_state: int = 42


DEFAULT_BINARY_CONFIG = TrainingConfig(target="target_binary")
DEFAULT_MULTICLASS_CONFIG = TrainingConfig(target="target_multiclass")
DEFAULT_REGRESSION_CONFIG = TrainingConfig(target="target_regression")
