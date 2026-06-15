import numpy as np
import pandas as pd

from config import FAILURE_THRESHOLD, MODEL_DIR
from utils import load_pickle


def load_models():
    binary_model = load_pickle(MODEL_DIR / "binary_failure_model.pkl")
    failure_type_model = load_pickle(MODEL_DIR / "failure_type_model.pkl")
    label_encoder = load_pickle(MODEL_DIR / "failure_type_label_encoder.pkl")
    feature_columns = load_pickle(MODEL_DIR / "feature_columns.pkl")
    return binary_model, failure_type_model, label_encoder, feature_columns


def predict_single_machine(input_data, threshold=FAILURE_THRESHOLD):
    """
    input_data can be a dict or a one-row pandas DataFrame.
    """
    binary_model, failure_type_model, label_encoder, feature_columns = load_models()

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    input_df = input_df[feature_columns]

    failure_probability = float(binary_model.predict_proba(input_df)[0][1])
    binary_prediction = int(failure_probability >= threshold)

    if binary_prediction == 0:
        return {
            "final_prediction": "No Failure",
            "failure_probability": failure_probability,
            "predicted_failure_type": "No Failure",
            "failure_type_confidence": np.nan,
        }

    failure_type_proba = failure_type_model.predict_proba(input_df)[0]
    failure_type_encoded = int(failure_type_proba.argmax())
    failure_type = label_encoder.inverse_transform([failure_type_encoded])[0]
    failure_type_confidence = float(failure_type_proba.max())

    return {
        "final_prediction": "Failure",
        "failure_probability": failure_probability,
        "predicted_failure_type": failure_type,
        "failure_type_confidence": failure_type_confidence,
    }
