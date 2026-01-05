from dataclasses import dataclass
from typing import Any, Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from src.features import add_features

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

def build_model_specs(feature_cols, classes, sample_weight):
    feat_eng = build_feature_transformer(feature_cols)

    lr = Pipeline([
        ("feat_eng", feat_eng),
        ("scaler", StandardScaler()),
        ("model", OneVsRestClassifier(LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=2000
        )))
    ])

    rf = Pipeline([
        ("feat_eng", feat_eng),
        ("model", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, n_jobs=-1, random_state=42,
            class_weight="balanced_subsample"
        ))
    ])

    specs = {
    "lr": ModelSpec("lr", lr, {}),
    "rf": ModelSpec("rf", rf, {"model__sample_weight": sample_weight}),
    }
    specs = try_add_xgb_lgbm(specs, feat_eng, classes, sample_weight)
    return specs

def try_add_xgb_lgbm(specs, feat_eng, classes, sample_weight):
    try: 
        from xgboost import XGBClassifier
        specs["xgb"] = ModelSpec(
            "xgb",  Pipeline([
                ("feat_eng", feat_eng),
                ("model", XGBClassifier(
                    n_estimators=800,
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
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        specs["lgbm"] = ModelSpec(
            "lgbm",
            Pipeline([
                ("feat_eng", feat_eng),
                ("model", LGBMClassifier(
                    n_estimators=1500,
                    learning_rate=0.03,
                    num_leaves=63,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multiclass",
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            {"model__sample_weight": sample_weight},
        )
    except Exception:
        pass


    return specs




    