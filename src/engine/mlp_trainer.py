import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.components.features import add_features
from src.components.models import build_mlp
from src.engine.evaluate import (
    compute_metrics,
    classification_report_text,
    confusion_matrix_array,
)


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MLPTrainer:
    def __init__(self, cfg, model_name):
        self.cfg = cfg
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.cfg.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=self.cfg.VAL_BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t),
            batch_size=self.cfg.VAL_BATCH_SIZE, shuffle=False
        )
        return train_loader, val_loader, test_loader
    
    def _eval(self, loader, criterion):
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                _ = criterion(logits, yb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_true.append(yb.cpu().numpy())
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_preds)
        return y_true, y_pred
    
    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Set seeds for reproducibility
        _set_seed(self.cfg.RANDOM_STATE)

        if self.cfg.USE_FEATURE_ENGINEERING:
            X_train = add_features(X_train, self.cfg.FEATURE_COLS)
            X_val = add_features(X_val, self.cfg.FEATURE_COLS)
            X_test = add_features(X_test, self.cfg.FEATURE_COLS)

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)

        train_loader, val_loader, test_loader = self._make_loaders(
            X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
        )

        classes = np.unique(y_train)
        class_w = compute_class_weight("balanced", classes=classes, y=y_train)
        class_w_t = torch.tensor(class_w, dtype=torch.float32).to(self.device)

        self.model = build_mlp(X_train_s.shape[1],len(classes)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_w_t)
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY
            )
        best_f1 = -1.0
        best_state = None
        for _ in range(self.cfg.EPOCHS):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_true, val_pred = self._eval(val_loader, criterion)
            val_f1 = compute_metrics(val_true, val_pred)["macro_f1"]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        val_true, val_pred = self._eval(val_loader, criterion)
        test_true, test_pred = self._eval(test_loader, criterion)

        return {
            "val": compute_metrics(val_true, val_pred),
            "test": compute_metrics(test_true, test_pred),
            "val_report": classification_report_text(val_true, val_pred),
            "test_report": classification_report_text(test_true, test_pred),
            "val_cm": confusion_matrix_array(val_true, val_pred),
            "test_cm": confusion_matrix_array(test_true, test_pred),
        }
    
    def predict(self, X):
        if self.cfg.USE_FEATURE_ENGINEERING:
            X = add_features(X, self.cfg.FEATURE_COLS)
        X_s = self.scaler.transform(X)
        X_t = torch.tensor(X_s, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def save_model(self, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "model.pt")
        joblib.dump(self.scaler, out_dir / "scaler.joblib")
