from pathlib import Path
import joblib
from src.components.models import build_model_specs
from src.engine.evaluate import (
    compute_metrics,
    classification_report_text,
    confusion_matrix_array,
)

class SklearnTrainer:
    def __init__(self, cfg, model_name, classes, sample_weight):
        self.cfg = cfg
        self.model_name = model_name
        self.classes = classes
        self.sample_weight = sample_weight
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        specs = build_model_specs(
            self.cfg.FEATURE_COLS,
            self.classes,
            self.sample_weight,
            use_feature_engineering=self.cfg.USE_FEATURE_ENGINEERING
        )
        spec = specs[self.model_name]
        self.model = spec.pipeline
        fit_kwargs = spec.fit_kwargs or {}
        self.model.fit(X_train, y_train, **fit_kwargs)

        # Evaluate on val + test
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)

        return {
            "val": compute_metrics(y_val, val_pred),
            "test": compute_metrics(y_test, test_pred),
            "val_report": classification_report_text(y_val, val_pred),
            "test_report": classification_report_text(y_test, test_pred),
            "val_cm": confusion_matrix_array(y_val, val_pred),
            "test_cm": confusion_matrix_array(y_test, test_pred),
        }
    
    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "model.joblib")
