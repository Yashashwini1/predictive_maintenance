"""
train_hierarchical_mlflow_metrics.py

Production-style hierarchical predictive maintenance pipeline with MLflow tracking.

Model 1: Binary failure detection
    - RandomForestClassifier
    - Target: Target

Model 2: Failure type diagnosis
    - XGBoostClassifier
    - Target: Failure Type
    - Trained only on confirmed failure rows
    - Random Failures excluded only from multiclass model

Logs to MLflow:
    - Parameters
    - Accuracy, precision, recall, F1
    - ROC AUC, PR AUC
    - False positive rate, false negative rate
    - Threshold comparison metrics
    - Confusion matrices
    - Classification reports
    - Feature importance CSV + plots
    - Hierarchical predictions CSV
    - Saved models
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================

DATA_PATH = "data/predictive_maintenance.csv"
EXPERIMENT_NAME = "Predictive_Maintenance_Hierarchical_Advanced_Metrics"
RUN_NAME = "Hierarchical_RF_XGBoost_threshold_0.50"

FAILURE_THRESHOLD = 0.50
# THRESHOLDS_TO_TEST = [0.50, 0.40, 0.30, 0.25, 0.20]

RANDOM_STATE = 42
TEST_SIZE = 0.20

ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_experiment(EXPERIMENT_NAME)


# =========================
# HELPER FUNCTIONS
# =========================

def log_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    path = ARTIFACT_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(path))
    return cm


def save_classification_report(y_true, y_pred, filename, target_names=None):
    report_text = classification_report(y_true, y_pred, target_names=target_names)
    path = ARTIFACT_DIR / filename
    with open(path, "w") as f:
        f.write(report_text)
    mlflow.log_artifact(str(path))
    return report_text


def get_feature_names(preprocessor):
    """Extract feature names from ColumnTransformer after fitting."""
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(columns)
            feature_names.extend(names)
        elif transformer == "passthrough":
            feature_names.extend(columns)
        else:
            feature_names.extend(columns)

    return list(feature_names)


def log_feature_importance(model_pipeline, filename_prefix, top_n=20):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        return None

    feature_names = get_feature_names(preprocessor)
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    csv_path = ARTIFACT_DIR / f"{filename_prefix}_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path))

    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title(f"Top {top_n} Feature Importances - {filename_prefix}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plot_path = ARTIFACT_DIR / f"{filename_prefix}_feature_importance.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(plot_path))

    return importance_df


def log_binary_threshold_metrics(y_true, y_proba, thresholds):
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        rows.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })

        suffix = str(threshold).replace(".", "_")
        mlflow.log_metric(f"threshold_{suffix}_accuracy", accuracy)
        mlflow.log_metric(f"threshold_{suffix}_precision", precision)
        mlflow.log_metric(f"threshold_{suffix}_recall", recall)
        mlflow.log_metric(f"threshold_{suffix}_f1", f1)
        mlflow.log_metric(f"threshold_{suffix}_false_negative_rate", false_negative_rate)
        mlflow.log_metric(f"threshold_{suffix}_false_positive_rate", false_positive_rate)

    threshold_df = pd.DataFrame(rows)
    csv_path = ARTIFACT_DIR / "binary_threshold_comparison.csv"
    threshold_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path))

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    plt.xlabel("Failure Probability Threshold")
    plt.ylabel("Score")
    plt.title("Binary Model Threshold Comparison")
    plt.legend()
    plt.tight_layout()
    plot_path = ARTIFACT_DIR / "binary_threshold_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(plot_path))

    return threshold_df


def log_roc_pr_curves(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("ROC Curve - Binary Failure Model")
    plt.tight_layout()
    roc_path = ARTIFACT_DIR / "binary_roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(roc_path))

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Binary Failure Model")
    plt.tight_layout()
    pr_path = ARTIFACT_DIR / "binary_precision_recall_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(pr_path))


# =========================
# LOAD DATA
# =========================

print("Running script:", __file__)
print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", EXPERIMENT_NAME)
print("Run name:", RUN_NAME)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Remove ID columns if present because they are identifiers, not predictive features.
df = df.drop(columns=[col for col in ["UDI", "Product ID"] if col in df.columns])

required_cols = {"Target", "Failure Type"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

features = [col for col in df.columns if col not in ["Target", "Failure Type"]]

# Data quality: label contradictions
contradiction_mask = (
    ((df["Target"] == 1) & (df["Failure Type"] == "No Failure")) |
    ((df["Target"] == 0) & (df["Failure Type"] != "No Failure"))
)
contradiction_count = int(contradiction_mask.sum())
contradiction_rate = contradiction_count / len(df)

class_distribution_target = df["Target"].value_counts().rename_axis("Target").reset_index(name="count")
class_distribution_failure_type = df["Failure Type"].value_counts().rename_axis("Failure Type").reset_index(name="count")

class_distribution_target.to_csv(ARTIFACT_DIR / "target_class_distribution.csv", index=False)
class_distribution_failure_type.to_csv(ARTIFACT_DIR / "failure_type_class_distribution.csv", index=False)


# =========================
# SINGLE PRODUCTION-STYLE SPLIT
# =========================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["Target"],
    random_state=RANDOM_STATE,
)

X_train_all = train_df[features]
categorical_cols = X_train_all.select_dtypes(include=["object", "category"]).columns
numeric_cols = X_train_all.select_dtypes(include=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)


# =========================
# MLflow RUN
# =========================

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("architecture", "hierarchical")
    mlflow.log_param("binary_model", "RandomForestClassifier")
    mlflow.log_param("failure_type_model", "XGBoostClassifier")
    mlflow.log_param("split_strategy", "single train/test split")
    mlflow.log_param("failure_threshold", FAILURE_THRESHOLD)
    mlflow.log_param("random_failure_handling", "excluded only from multiclass model")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("n_rows", len(df))
    mlflow.log_param("n_features", len(features))

    mlflow.log_metric("contradictory_records", contradiction_count)
    mlflow.log_metric("contradiction_rate", contradiction_rate)
    mlflow.log_artifact(str(ARTIFACT_DIR / "target_class_distribution.csv"))
    mlflow.log_artifact(str(ARTIFACT_DIR / "failure_type_class_distribution.csv"))

    # =====================================================
    # MODEL 1: BINARY FAILURE DETECTION
    # =====================================================

    X_train_binary = train_df[features]
    y_train_binary = train_df["Target"]
    X_test_binary = test_df[features]
    y_test_binary = test_df["Target"]

    rf_params = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
    }

    for key, value in rf_params.items():
        mlflow.log_param(f"rf_{key}", value)

    binary_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_params)),
        ]
    )

    print("\nStarting binary Random Forest training...")
    binary_model.fit(X_train_binary, y_train_binary)
    print("Binary Random Forest training complete.")

    y_proba_binary = binary_model.predict_proba(X_test_binary)[:, 1]
    y_pred_binary = (y_proba_binary >= FAILURE_THRESHOLD).astype(int)

    binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
    binary_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    binary_recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    binary_f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    binary_roc_auc = roc_auc_score(y_test_binary, y_proba_binary)
    binary_pr_auc = average_precision_score(y_test_binary, y_proba_binary)

    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    failure_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

    mlflow.log_metric("binary_accuracy", binary_accuracy)
    mlflow.log_metric("binary_precision", binary_precision)
    mlflow.log_metric("binary_recall", binary_recall)
    mlflow.log_metric("binary_f1", binary_f1)
    mlflow.log_metric("binary_roc_auc", binary_roc_auc)
    mlflow.log_metric("binary_pr_auc", binary_pr_auc)
    mlflow.log_metric("binary_false_negative_rate", false_negative_rate)
    mlflow.log_metric("binary_false_positive_rate", false_positive_rate)
    mlflow.log_metric("binary_failure_detection_rate", failure_detection_rate)
    mlflow.log_metric("binary_true_positives", tp)
    mlflow.log_metric("binary_false_negatives", fn)
    mlflow.log_metric("binary_true_negatives", tn)
    mlflow.log_metric("binary_false_positives", fp)

    print("\n==============================")
    print("MODEL 1: BINARY FAILURE DETECTION")
    print("==============================")
    print(classification_report(y_test_binary, y_pred_binary))

    save_classification_report(
        y_test_binary,
        y_pred_binary,
        filename="binary_classification_report.txt",
    )

    log_confusion_matrix(
        y_test_binary,
        y_pred_binary,
        labels=["No Failure", "Failure"],
        filename="binary_confusion_matrix.png",
    )

    log_roc_pr_curves(y_test_binary, y_proba_binary)
    # threshold_df = log_binary_threshold_metrics(y_test_binary, y_proba_binary, THRESHOLDS_TO_TEST)
    log_feature_importance(binary_model, "binary_random_forest")

    # =====================================================
    # MODEL 2: FAILURE TYPE DIAGNOSIS
    # =====================================================

    failure_train_df = train_df[
        (train_df["Target"] == 1)
        & (train_df["Failure Type"] != "No Failure")
        & (train_df["Failure Type"] != "Random Failures")
    ].copy()

    failure_test_df = test_df[
        (test_df["Target"] == 1)
        & (test_df["Failure Type"] != "No Failure")
        & (test_df["Failure Type"] != "Random Failures")
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
        "random_state": RANDOM_STATE,
    }

    for key, value in xgb_params.items():
        mlflow.log_param(f"xgb_{key}", value)

    failure_type_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**xgb_params)),
        ]
    )

    print("\nStarting multiclass XGBoost training...")
    failure_type_model.fit(X_train_multi, y_train_multi_encoded)
    print("Multiclass XGBoost training complete.")

    y_pred_multi_encoded = failure_type_model.predict(X_test_multi)
    y_proba_multi = failure_type_model.predict_proba(X_test_multi)

    multi_accuracy = accuracy_score(y_test_multi_encoded, y_pred_multi_encoded)
    multi_precision_macro = precision_score(y_test_multi_encoded, y_pred_multi_encoded, average="macro", zero_division=0)
    multi_recall_macro = recall_score(y_test_multi_encoded, y_pred_multi_encoded, average="macro", zero_division=0)
    multi_f1_macro = f1_score(y_test_multi_encoded, y_pred_multi_encoded, average="macro", zero_division=0)
    multi_precision_weighted = precision_score(y_test_multi_encoded, y_pred_multi_encoded, average="weighted", zero_division=0)
    multi_recall_weighted = recall_score(y_test_multi_encoded, y_pred_multi_encoded, average="weighted", zero_division=0)
    multi_f1_weighted = f1_score(y_test_multi_encoded, y_pred_multi_encoded, average="weighted", zero_division=0)

    mlflow.log_metric("multiclass_accuracy", multi_accuracy)
    mlflow.log_metric("multiclass_macro_precision", multi_precision_macro)
    mlflow.log_metric("multiclass_macro_recall", multi_recall_macro)
    mlflow.log_metric("multiclass_macro_f1", multi_f1_macro)
    mlflow.log_metric("multiclass_weighted_precision", multi_precision_weighted)
    mlflow.log_metric("multiclass_weighted_recall", multi_recall_weighted)
    mlflow.log_metric("multiclass_weighted_f1", multi_f1_weighted)

    class_report_dict = classification_report(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    for class_name in label_encoder.classes_:
        mlflow.log_metric(f"{class_name}_precision", class_report_dict[class_name]["precision"])
        mlflow.log_metric(f"{class_name}_recall", class_report_dict[class_name]["recall"])
        mlflow.log_metric(f"{class_name}_f1", class_report_dict[class_name]["f1-score"])

    print("\n==============================")
    print("MODEL 2: FAILURE TYPE DIAGNOSIS")
    print("==============================")
    print(classification_report(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    save_classification_report(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        filename="failure_type_classification_report.txt",
        target_names=label_encoder.classes_,
    )

    log_confusion_matrix(
        y_test_multi_encoded,
        y_pred_multi_encoded,
        labels=label_encoder.classes_,
        filename="failure_type_confusion_matrix.png",
    )

    log_feature_importance(failure_type_model, "failure_type_xgboost")

    # =====================================================
    # CONNECTED HIERARCHICAL INFERENCE ON TEST DATA
    # =====================================================

    final_predictions = []

    for idx, row in X_test_binary.iterrows():
        input_row = pd.DataFrame([row])

        failure_probability = binary_model.predict_proba(input_row)[0][1]
        binary_prediction = int(failure_probability >= FAILURE_THRESHOLD)

        if binary_prediction == 0:
            final_predictions.append({
                "index": idx,
                "actual_target": int(test_df.loc[idx, "Target"]),
                "actual_failure_type": test_df.loc[idx, "Failure Type"],
                "final_prediction": "No Failure",
                "failure_probability": failure_probability,
                "predicted_failure_type": "No Failure",
                "failure_type_confidence": np.nan,
            })
        else:
            failure_type_proba = failure_type_model.predict_proba(input_row)[0]
            failure_type_encoded = int(failure_type_proba.argmax())
            failure_type = label_encoder.inverse_transform([failure_type_encoded])[0]
            failure_type_confidence = float(failure_type_proba.max())

            final_predictions.append({
                "index": idx,
                "actual_target": int(test_df.loc[idx, "Target"]),
                "actual_failure_type": test_df.loc[idx, "Failure Type"],
                "final_prediction": "Failure",
                "failure_probability": failure_probability,
                "predicted_failure_type": failure_type,
                "failure_type_confidence": failure_type_confidence,
            })

    hierarchical_results = pd.DataFrame(final_predictions)
    hierarchical_path = ARTIFACT_DIR / "hierarchical_test_predictions.csv"
    hierarchical_results.to_csv(hierarchical_path, index=False)
    mlflow.log_artifact(str(hierarchical_path))

    print("\nSample Hierarchical Predictions:")
    print(hierarchical_results.head(10))

    # =====================================================
    # SAVE MODELS
    # =====================================================

    binary_model_path = MODEL_DIR / "binary_failure_model.pkl"
    failure_type_model_path = MODEL_DIR / "failure_type_model.pkl"
    label_encoder_path = MODEL_DIR / "failure_type_label_encoder.pkl"

    with open(binary_model_path, "wb") as f:
        pickle.dump(binary_model, f)

    with open(failure_type_model_path, "wb") as f:
        pickle.dump(failure_type_model, f)

    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    mlflow.sklearn.log_model(binary_model, artifact_path="binary_failure_model")
    mlflow.sklearn.log_model(failure_type_model, artifact_path="failure_type_model")
    mlflow.log_artifact(str(binary_model_path))
    mlflow.log_artifact(str(failure_type_model_path))
    mlflow.log_artifact(str(label_encoder_path))

print("\nTraining complete. Check MLflow UI.")
