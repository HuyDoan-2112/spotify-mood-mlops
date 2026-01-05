from src.data import load_and_split, load_raw_csv, dataset_hash
from src.trainers.factory import build_trainer
from src.io import save_json, save_text, save_confusion_matrix, save_leaderboard, save_metadata
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import numpy as np

class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self, model_name="all", run_id=None):
        run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.cfg.RUNS_DIR / run_id

        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(self.cfg)
        df = load_raw_csv(self.cfg.RAW_DATA_PATH)   
        data_hash = dataset_hash(df, self.cfg.FEATURE_COLS, self.cfg.TARGET_COL)
        
        classes = np.array(sorted(y_train.unique()))
        sample_weight = make_sample_weight(y_train, classes)

        names = self.cfg.MODEL_NAMES if model_name == "all" else [model_name]
        leaderboard = []
        for name in names:
            trainer = build_trainer(name, self.cfg, classes, sample_weight)
            out = trainer.fit(X_train, y_train, X_val, y_val, X_test, y_test)

            model_dir = self.cfg.ARTIFACTS_DIR / run_id / name
            trainer.save_model(model_dir)

            # Save reports  
            save_json(run_dir / "metrics" / f"{name}.json",
                      {"val": out["val"], "test": out["test"]})
            save_text(run_dir / "reports" / f"{name}_val.txt", out["val_report"])
            save_text(run_dir / "reports" / f"{name}_test.txt", out["test_report"])
            save_confusion_matrix(run_dir / "confusion" / f"{name}_val.csv",
                                  out["val_cm"], classes)
            save_confusion_matrix(run_dir / "confusion" / f"{name}_test.csv",
                                  out["test_cm"], classes)

            save_metadata(run_dir / "metadata" / f"{name}.json",
                          self.cfg, name, classes, data_hash)

            leaderboard.append({
                "model": name,
                "val_macro_f1": out["val"]["macro_f1"],
                "test_macro_f1": out["test"]["macro_f1"],
            })

        save_leaderboard(run_dir / "leaderboard.csv", leaderboard)
        save_json(run_dir / "best_model.json",
                  {"model": self.cfg.PREFERRED_MODEL, "run_id": run_id})
        return run_id
    
def make_sample_weight(y_train, classes):
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_map = {c: w for c, w in zip(classes, cw)}
    return  y_train.map(class_weight_map).to_numpy()

