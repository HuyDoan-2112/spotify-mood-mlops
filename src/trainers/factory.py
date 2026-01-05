from src.trainers.sklearn_trainer import SklearnTrainer
from src.trainers.mlp_trainer import MLPTrainer


def build_trainer(model_name, cfg, classes, sample_weight=None):
    if model_name in ["lr", "rf", "xgb", "lgbm"]:
        return SklearnTrainer(cfg, model_name, classes, sample_weight)
    if model_name == "mlp":
        return MLPTrainer(cfg, model_name)
    raise ValueError(f"Unknown model: {model_name}")
