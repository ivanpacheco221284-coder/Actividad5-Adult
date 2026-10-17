#!/usr/bin/env python3
"""
train.py
Entrena y registra experimentos en MLflow para el dataset Adult.
Uso:
  python fuentes/train.py --model logreg
  python fuentes/train.py --model rf
Parámetros importantes:
  --data-path : ruta al CSV limpio (por defecto datos/datos_limp/clean_adult.csv)
  --mlflow-uri : URI de tracking (por defecto file:./mlruns) pero puede ser sobrescrito por la variable de entorno MLFLOW_TRACKING_URI
"""
import argparse
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    return pd.read_csv(path)


def get_feature_lists(df):
    y = df["income"]
    X = df.drop(columns=["income"])
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, numeric, categorical


def make_preprocessor(numeric, categorical, sparse_ohe=False):
    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    # OneHotEncoder: use sparse=False for compatibility, but allow sparse output if requested
    if sparse_ohe:
        cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))])
    else:
        # sparse=False is widely supported across sklearn versions
        cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", cat_pipe, categorical)])
    return preprocessor


def plot_confusion(cm, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def read_dataset_sha(path="datos/datos_ini/dataset_version.txt"):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def main(args):
    # Allow MLFLOW_TRACKING_URI environment variable to override CLI argument (useful in Colab)
    mlflow_uri = args.mlflow_uri
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        mlflow_uri = env_uri

    mlflow.set_tracking_uri(mlflow_uri)

    data = load_data(args.data_path)
    X, y, numeric, categorical = get_feature_lists(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = make_preprocessor(numeric, categorical, sparse_ohe=args.ohe_sparse)

    if args.model == "logreg":
        clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
        param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif args.model == "rf":
        clf = RandomForestClassifier(random_state=42)
        param_grid = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]}
    else:
        raise ValueError("model must be 'logreg' or 'rf'")

    pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)

    with mlflow.start_run():
        mlflow.log_param("model", args.model)
        # log mlflow tracking uri used
        mlflow.log_param("mlflow_tracking_uri", mlflow_uri)
        ds_sha = read_dataset_sha()
        if ds_sha:
            mlflow.log_param("dataset_sha256", ds_sha)

        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best.named_steps["clf"], "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        if roc is not None:
            mlflow.log_metric("roc_auc", roc)

        # log best params
        for k, v in gs.best_params_.items():
            mlflow.log_param(k, v)

        # save model artifact
        os.makedirs("artifacts", exist_ok=True)
        model_path = f"artifacts/model_{args.model}.pkl"
        joblib.dump(best, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_path = f"artifacts/cm_{args.model}.png"
        plot_confusion(cm, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        print("Finished. Metrics:")
        print({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="datos/datos_limp/clean_adult.csv")
    parser.add_argument("--model", type=str, choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--mlflow-uri", type=str, default="file:./mlruns")
    parser.add_argument("--ohe-sparse", action="store_true", help="Use sparse OneHotEncoder output to save memory (useful in Colab)")
    args = parser.parse_args()
    main(args)
