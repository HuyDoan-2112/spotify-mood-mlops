import logging
from dataclasses import dataclass
from typing import Any, Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from src.components.features import add_features

logger = logging.getLogger(__name__)

@dataclass
class ModelSpec:
    name: str
    pipeline: Any
    fit_kwargs: Dict[str, Any]


def build_feature_transformer(feature_cols):
    return FunctionTransformer(
        add_features,
        validate=False,
        kw_args={"feature_cols": feature_cols}
    )

def build_model_specs(feature_cols, classes, sample_weight, use_feature_engineering=False):
    # Build pipeline steps based on whether feature engineering is enabled
    if use_feature_engineering:
        feat_eng = build_feature_transformer(feature_cols)
        lr_steps = [("feat_eng", feat_eng), ("scaler", StandardScaler())]
        rf_steps = [("feat_eng", feat_eng)]
    else:
        lr_steps = [("scaler", StandardScaler())]
        rf_steps = []

    lr = Pipeline(lr_steps + [
        ("model", OneVsRestClassifier(LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=2000
        )))
    ])

    rf = Pipeline(rf_steps + [
        ("model", RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, n_jobs=-1, random_state=42,
            class_weight="balanced_subsample"
        ))
    ])

    specs = {
        "lr": ModelSpec("lr", lr, {}),
        "rf": ModelSpec("rf", rf, {"model__sample_weight": sample_weight}),
    }
    specs = try_add_xgb_lgbm(specs, classes, sample_weight, use_feature_engineering)
    return specs

def build_mlp(input_dim: int, num_classes: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.20),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(32, num_classes),
    )

def try_add_xgb_lgbm(specs, classes, sample_weight, use_feature_engineering=False):
    from src.config import FEATURE_COLS

    if use_feature_engineering:
        feat_eng = build_feature_transformer(FEATURE_COLS)
        xgb_steps = [("feat_eng", feat_eng)]
        lgbm_steps = [("feat_eng", feat_eng)]
    else:
        xgb_steps = []
        lgbm_steps = []

    try:
        from xgboost import XGBClassifier
        specs["xgb"] = ModelSpec(
            "xgb",  Pipeline(xgb_steps + [
                ("model", XGBClassifier(
                    n_estimators=600,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="multi:softprob",
                    num_class=len(classes),
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            {"model__sample_weight": sample_weight},
        )
    except ImportError as e:
        logger.warning("XGBoost not available, skipping xgb model: %s", e)
    except Exception as e:
        logger.error("Failed to initialize XGBoost model: %s", e, exc_info=True)

    try:
        from lightgbm import LGBMClassifier
        specs["lgbm"] = ModelSpec(
            "lgbm",
            Pipeline(lgbm_steps + [
                ("model", LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.03,
                    num_leaves=63,
                    subsample=0.9,
                    max_depth=8,
                    reg_lambda=1.2,
                    colsample_bytree=0.9,
                    objective="multiclass",
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            {"model__sample_weight": sample_weight},
        )
    except ImportError as e:
        logger.warning("LightGBM not available, skipping lgbm model: %s", e)
    except Exception as e:
        logger.error("Failed to initialize LightGBM model: %s", e, exc_info=True)

    return specs




    
