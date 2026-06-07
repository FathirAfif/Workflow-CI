import argparse
import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)


def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train = train.drop("Outcome", axis=1)
    y_train = train["Outcome"]
    X_test  = test.drop("Outcome", axis=1)
    y_test  = test["Outcome"]
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    path = "training_confusion_matrix.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    path = "roc_curve.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    path = "classification_report.json"
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path


def main(args):
    # Tidak perlu set tracking URI atau start_run manual
    # MLflow Project sudah handle ini via environment variable
    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    # Langsung log tanpa with mlflow.start_run()
    mlflow.log_param("n_estimators",  args.n_estimators)
    mlflow.log_param("max_depth",     args.max_depth)
    mlflow.log_param("random_state",  args.random_state)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1_score" : f1_score(y_test, y_pred),
        "roc_auc"  : roc_auc_score(y_test, y_prob),
    }
    mlflow.log_metrics(metrics)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    mlflow.sklearn.log_model(model, artifact_path="model")

    cm_path     = save_confusion_matrix(y_test, y_pred)
    roc_path    = save_roc_curve(y_test, y_prob)
    report_path = save_classification_report(y_test, y_pred)

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(report_path)

    mlflow.set_tag("model", "RandomForestClassifier")
    mlflow.set_tag("dataset", "Pima Indians Diabetes")
    print("[MLflow] Run selesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     type=str, default="diabetes_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
