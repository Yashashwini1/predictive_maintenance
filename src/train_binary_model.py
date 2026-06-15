from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE


def train_binary_failure_model(X_train, y_train, preprocessor):
    rf_params = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
    }

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_params)),
        ]
    )

    model.fit(X_train, y_train)
    return model, rf_params
