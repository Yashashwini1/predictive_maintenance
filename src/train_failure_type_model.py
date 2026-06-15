from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import RANDOM_STATE


def train_failure_type_model(X_train, y_train, preprocessor):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    xgb_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
    }

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**xgb_params)),
        ]
    )

    model.fit(X_train, y_train_encoded)
    return model, label_encoder, xgb_params
