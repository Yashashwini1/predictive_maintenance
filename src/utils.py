import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from config import ARTIFACT_DIR


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    path = ARTIFACT_DIR / filename
    with open(path, "w") as f:
        f.write(report_text)
    mlflow.log_artifact(str(path))
    return report_text


def get_feature_names(preprocessor):
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
