import pandas as pd

def add_features(X_in, feature_cols):
    # Handle ndarray input from sklearn
    if not isinstance(X_in, pd.DataFrame):
        X_in = pd.DataFrame(X_in, columns=feature_cols)

    X_ = X_in[feature_cols].copy()

    X_["intensity"] = X_["energy"] * (-X_["loudness"])
    X_["rhythm_drive"] = X_["danceability"] * X_["tempo"]
    X_["calm_score"] = X_["acousticness"] + X_["instrumentalness"]

    return X_
