import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    ARTIFACT_DIR,
    EXPERIMENT_NAME,
    FAILURE_THRESHOLD,
    FAILURE_TYPE_COLUMN,
    MODEL_DIR,
    NO_FAILURE_LABEL,
    RANDOM_FAILURE_LABEL,
    RANDOM_STATE,
    RUN_NAME,
    TARGET_COLUMN,
    TEST_SIZE,
)
from data_loader import load_clean_data
from preprocessing import build_preprocessor, get_feature_columns, split_data
from train_binary_model import train_binary_failure_model
from train_failure_type_model import train_failure_type_model
from utils import (
    log_confusion_matrix,
    log_feature_importance,
    log_roc_pr_curves,
    save_classification_report,
    save_pickle,
)


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_clean_data()
    features = get_feature_columns(df)

    train_df, test_df = split_data(df)

    X_train_binary = train_df[features]
    y_train_binary = train_df[TARGET_COLUMN]
    X_test_binary = test_df[features]
    y_test_binary = test_df[TARGET_COLUMN]

    preprocessor = build_preprocessor(X_train_binary)

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("architecture", "hierarchical")
        mlflow.log_param("binary_model", "RandomForestClassifier")
        mlflow.log_param("failure_type_model", "XGBoostClassifier")
        mlflow.log_param("failure_threshold", FAILURE_THRESHOLD)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_features", len(features))

        # -------------------------
        # Model 1: Binary RF model
        # -------------------------
        print("Training binary Random Forest model...")
        binary_model, rf_params = train_binary_failure_model(
            X_train_binary,
            y_train_binary,
            preprocessor,
        )

        for key, value in rf_params.items():
            mlflow.log_param(f"rf_{key}", value)

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

        mlflow.log_metric("binary_accuracy", binary_accuracy)
        mlflow.log_metric("binary_precision", binary_precision)
        mlflow.log_metric("binary_recall", binary_recall)
        mlflow.log_metric("binary_f1", binary_f1)
        mlflow.log_metric("binary_roc_auc", binary_roc_auc)
        mlflow.log_metric("binary_pr_auc", binary_pr_auc)
        mlflow.log_metric("binary_false_negative_rate", false_negative_rate)
        mlflow.log_metric("binary_false_positive_rate", false_positive_rate)
        mlflow.log_metric("binary_true_positives", tp)
        mlflow.log_metric("binary_false_negatives", fn)
        mlflow.log_metric("binary_true_negatives", tn)
        mlflow.log_metric("binary_false_positives", fp)

        print("\nMODEL 1: BINARY FAILURE DETECTION")
        print(classification_report(y_test_binary, y_pred_binary, zero_division=0))

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
        log_feature_importance(binary_model, "binary_random_forest")

        # -------------------------
        # Model 2: XGBoost failure type model
        # -------------------------
        failure_train_df = train_df[
            (train_df[TARGET_COLUMN] == 1)
            & (train_df[FAILURE_TYPE_COLUMN] != NO_FAILURE_LABEL)
            & (train_df[FAILURE_TYPE_COLUMN] != RANDOM_FAILURE_LABEL)
        ].copy()

        failure_test_df = test_df[
            (test_df[TARGET_COLUMN] == 1)
            & (test_df[FAILURE_TYPE_COLUMN] != NO_FAILURE_LABEL)
            & (test_df[FAILURE_TYPE_COLUMN] != RANDOM_FAILURE_LABEL)
        ].copy()

        X_train_multi = failure_train_df[features]
        y_train_multi = failure_train_df[FAILURE_TYPE_COLUMN]
        X_test_multi = failure_test_df[features]
        y_test_multi = failure_test_df[FAILURE_TYPE_COLUMN]

        print("\nTraining multiclass XGBoost model...")
        failure_type_model, label_encoder, xgb_params = train_failure_type_model(
            X_train_multi,
            y_train_multi,
            preprocessor,
        )

        for key, value in xgb_params.items():
            mlflow.log_param(f"xgb_{key}", value)

        y_test_multi_encoded = label_encoder.transform(y_test_multi)
        y_pred_multi_encoded = failure_type_model.predict(X_test_multi)

        multi_accuracy = accuracy_score(y_test_multi_encoded, y_pred_multi_encoded)
        multi_precision_macro = precision_score(
            y_test_multi_encoded,
            y_pred_multi_encoded,
            average="macro",
            zero_division=0,
        )
        multi_recall_macro = recall_score(
            y_test_multi_encoded,
            y_pred_multi_encoded,
            average="macro",
            zero_division=0,
        )
        multi_f1_macro = f1_score(
            y_test_multi_encoded,
            y_pred_multi_encoded,
            average="macro",
            zero_division=0,
        )

        mlflow.log_metric("multiclass_accuracy", multi_accuracy)
        mlflow.log_metric("multiclass_macro_precision", multi_precision_macro)
        mlflow.log_metric("multiclass_macro_recall", multi_recall_macro)
        mlflow.log_metric("multiclass_macro_f1", multi_f1_macro)

        print("\nMODEL 2: FAILURE TYPE DIAGNOSIS")
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

        # -------------------------
        # Save hierarchical test predictions
        # -------------------------
        final_predictions = []

        for idx, row in X_test_binary.iterrows():
            input_row = pd.DataFrame([row])
            failure_probability = float(binary_model.predict_proba(input_row)[0][1])
            binary_prediction = int(failure_probability >= FAILURE_THRESHOLD)

            if binary_prediction == 0:
                predicted_failure_type = "No Failure"
                failure_type_confidence = None
                final_prediction = "No Failure"
            else:
                failure_type_proba = failure_type_model.predict_proba(input_row)[0]
                failure_type_encoded = int(failure_type_proba.argmax())
                predicted_failure_type = label_encoder.inverse_transform([failure_type_encoded])[0]
                failure_type_confidence = float(failure_type_proba.max())
                final_prediction = "Failure"

            final_predictions.append({
                "index": idx,
                "actual_target": int(test_df.loc[idx, TARGET_COLUMN]),
                "actual_failure_type": test_df.loc[idx, FAILURE_TYPE_COLUMN],
                "final_prediction": final_prediction,
                "failure_probability": failure_probability,
                "predicted_failure_type": predicted_failure_type,
                "failure_type_confidence": failure_type_confidence,
            })

        hierarchical_results = pd.DataFrame(final_predictions)
        hierarchical_path = ARTIFACT_DIR / "hierarchical_test_predictions.csv"
        hierarchical_results.to_csv(hierarchical_path, index=False)
        mlflow.log_artifact(str(hierarchical_path))

        # -------------------------
        # Save models
        # -------------------------
        save_pickle(binary_model, MODEL_DIR / "binary_failure_model.pkl")
        save_pickle(failure_type_model, MODEL_DIR / "failure_type_model.pkl")
        save_pickle(label_encoder, MODEL_DIR / "failure_type_label_encoder.pkl")
        save_pickle(features, MODEL_DIR / "feature_columns.pkl")

        mlflow.sklearn.log_model(binary_model, artifact_path="binary_failure_model")
        mlflow.sklearn.log_model(failure_type_model, artifact_path="failure_type_model")

    print("\nTraining complete. Models saved in models/ and artifacts saved in artifacts/.")


if __name__ == "__main__":
    main()
