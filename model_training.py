# train_hierarchical_mlflow.py

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================

DATA_PATH = "data/predictive_maintenance.csv"
EXPERIMENT_NAME = "Predictive_Maintenance_Hierarchical_"
RUN_NAME = "Hierarchical_RF_XGBoost_failure_probability_0.3"

mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


# =========================
# HELPER FUNCTION
# =========================

def save_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(xticks_rotation=45)
    plt.tight_layout()

    path = f"artifacts/{filename}"
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

# Remove ID columns if present
df = df.drop(columns=[col for col in ["UDI", "Product ID"] if col in df.columns])

# Remove Random Failures only for failure-type model later, NOT globally
features = [col for col in df.columns if col not in ["Target", "Failure Type"]]


# =========================
# SINGLE PRODUCTION-STYLE SPLIT
# =========================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Target"],
    random_state=42
)


# =========================
# PREPROCESSOR
# Same feature columns for both models
# =========================

X_train_all = train_df[features]

categorical_cols = X_train_all.select_dtypes(
    include=["object", "category"]
).columns

numeric_cols = X_train_all.select_dtypes(
    include=["int64", "float64"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


# =========================
# HIERARCHICAL RUN
# =========================

print(RUN_NAME)

with mlflow.start_run(run_name=RUN_NAME):

    mlflow.log_param("architecture", "hierarchical")
    mlflow.log_param("binary_model", "RandomForestClassifier")
    mlflow.log_param("failure_type_model", "XGBoostClassifier")
    mlflow.log_param("split_strategy", "single train/test split")
    mlflow.log_param("random_failure_handling", "excluded only from multiclass model")

    # =====================================================
    # MODEL 1: BINARY FAILURE DETECTION
    # =====================================================

    X_train_binary = train_df[features]
    y_train_binary = train_df["Target"]

    X_test_binary = test_df[features]
    y_test_binary = test_df["Target"]

    rf_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42
    }

    for key, value in rf_params.items():
        mlflow.log_param(f"rf_{key}", value)

    binary_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_params))
        ]
    )

    binary_model.fit(X_train_binary, y_train_binary)

    y_pred_binary = binary_model.predict(X_test_binary)
    y_proba_binary = binary_model.predict_proba(X_test_binary)[:, 1]

    binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
    binary_precision = precision_score(y_test_binary, y_pred_binary)
    binary_recall = recall_score(y_test_binary, y_pred_binary)
    binary_f1 = f1_score(y_test_binary, y_pred_binary)

    mlflow.log_metric("binary_accuracy", binary_accuracy)
    mlflow.log_metric("binary_precision", binary_precision)
    mlflow.log_metric("binary_recall", binary_recall)
    mlflow.log_metric("binary_f1", binary_f1)

    print("\n==============================")
    print("MODEL 1: BINARY FAILURE DETECTION")
    print("==============================")
    print(classification_report(y_test_binary, y_pred_binary))

    save_confusion_matrix(
        y_test_binary,
        y_pred_binary,
        labels=["No Failure", "Failure"],
        filename="binary_confusion_matrix.png"
    )

    # =====================================================
    # MODEL 2: FAILURE TYPE DIAGNOSIS
    # Train only on failure rows from TRAIN split
    # Test only on failure rows from TEST split
    # =====================================================

    failure_train_df = train_df[
        (train_df["Target"] == 1) &
        (train_df["Failure Type"] != "No Failure") &
        (train_df["Failure Type"] != "Random Failures")
    ].copy()

    failure_test_df = test_df[
        (test_df["Target"] == 1) &
        (test_df["Failure Type"] != "No Failure") &
        (test_df["Failure Type"] != "Random Failures")
    ].copy()

    X_train_multi = failure_train_df[features]
    y_train_multi = failure_train_df["Failure Type"]

    X_test_multi = failure_test_df[features]
    y_test_multi = failure_test_df["Failure Type"]

    label_encoder = LabelEncoder()
    y_train_multi_encoded = label_encoder.fit_transform(y_train_multi)
    y_test_multi_encoded = label_encoder.transform(y_test_multi)

    xgb_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42
    }

    for key, value in xgb_params.items():
        mlflow.log_param(f"xgb_{key}", value)

    failure_type_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**xgb_params))
        ]
    )

    failure_type_model.fit(X_train_multi, y_train_multi_encoded)

    y_pred_multi_encoded = failure_type_model.predict(X_test_multi)
    y_proba_multi = failure_type_model.predict_proba(X_test_multi)

    multi_accuracy = accuracy_score(y_test_multi_encoded, y_pred_multi_encoded)
    multi_precision = precision_score(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        average="macro"
    )
    multi_recall = recall_score(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        average="macro"
    )
    multi_f1 = f1_score(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        average="macro"
    )

    mlflow.log_metric("multiclass_accuracy", multi_accuracy)
    mlflow.log_metric("multiclass_macro_precision", multi_precision)
    mlflow.log_metric("multiclass_macro_recall", multi_recall)
    mlflow.log_metric("multiclass_macro_f1", multi_f1)

    print("\n==============================")
    print("MODEL 2: FAILURE TYPE DIAGNOSIS")
    print("==============================")
    print(classification_report(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        target_names=label_encoder.classes_
    ))

    save_confusion_matrix(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        labels=label_encoder.classes_,
        filename="failure_type_confusion_matrix.png"
    )

    # =====================================================
    # HIERARCHICAL INFERENCE ON TEST DATA
    # This connects both models together
    # =====================================================

    final_predictions = []

    for idx, row in X_test_binary.iterrows():
        input_row = pd.DataFrame([row])

        failure_probability = binary_model.predict_proba(input_row)[0][1]
        binary_prediction = int(failure_probability >= 0.3)

        if binary_prediction == 0:
            final_predictions.append({
                "Final Prediction": "No Failure",
                "Failure Probability": failure_probability,
                "Failure Type": "No Failure",
                "Failure Type Confidence": None
            })
        else:
            failure_type_proba = failure_type_model.predict_proba(input_row)[0]
            failure_type_encoded = failure_type_proba.argmax()
            failure_type = label_encoder.inverse_transform(
                [failure_type_encoded]
            )[0]
            failure_type_confidence = failure_type_proba.max()

            final_predictions.append({
                "Final Prediction": "Failure",
                "Failure Probability": failure_probability,
                "Failure Type": failure_type,
                "Failure Type Confidence": failure_type_confidence
            })

    hierarchical_results = pd.DataFrame(final_predictions)

    hierarchical_results.to_csv(
        "artifacts/hierarchical_test_predictions.csv",
        index=False
    )

    mlflow.log_artifact("artifacts/hierarchical_test_predictions.csv")

    print("\nSample Hierarchical Predictions:")
    print(hierarchical_results.head(10))

    # =====================================================
    # SAVE MODELS
    # =====================================================

    with open("models/binary_failure_model.pkl", "wb") as f:
        pickle.dump(binary_model, f)

    with open("models/failure_type_model.pkl", "wb") as f:
        pickle.dump(failure_type_model, f)

    with open("models/failure_type_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    mlflow.sklearn.log_model(
        binary_model,
        artifact_path="binary_failure_model"
    )

    mlflow.sklearn.log_model(
        failure_type_model,
        artifact_path="failure_type_model"
    )

    mlflow.log_artifact("models/binary_failure_model.pkl")
    mlflow.log_artifact("models/failure_type_model.pkl")
    mlflow.log_artifact("models/failure_type_label_encoder.pkl")

print("\nTraining complete. Check MLflow UI.")