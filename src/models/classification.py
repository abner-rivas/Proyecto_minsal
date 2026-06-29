"""
src/models/classification.py

Entrenamiento y serializacion de modelos de Clasificacion (Tarea B: Riesgo Salud Mental).
Entrena 3 modelos (2 obligatorios de desafio.md + 1 opcional):
  1. LogisticRegression (baseline, class_weight="balanced")
  2. RandomForestClassifier (class_weight="balanced", RandomizedSearchCV)
  3. XGBClassifier (scale_pos_weight, RandomizedSearchCV)

Uso:
    python src/models/classification.py --features QN
    python src/models/classification.py --features Q --n-iter 30 --cv 5
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.console import console, make_progress, make_spinner, Panel, Table
from src.pipeline_factory import build_classification_pipeline
from src.evaluation.metrics import (
    evaluate_classification,
    find_optimal_threshold,
    print_classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento Clasificacion Salud Mental")
    parser.add_argument("--features", choices=["Q", "QN"], default="QN",
                        help="Tipo de features: Q (ordinales) o QN (binarias)")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Iteraciones de RandomizedSearchCV para RF/XGB")
    parser.add_argument("--cv", type=int, default=5,
                        help="Folds de validacion cruzada (StratifiedKFold)")
    return parser.parse_args()


def train_logistic_regression(X_train, y_train, X_val, y_val, feature_type, cv):
    """Entrena LogisticRegression con GridSearchCV sobre C."""
    pipeline = build_classification_pipeline(
        feature_type=feature_type, model="lr")

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
    }
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    console.print(f"    Mejor C: [bold]{search.best_params_['classifier__C']}[/bold]")

    y_proba_val = best_model.predict_proba(X_val)[:, 1]
    threshold, f1_val = find_optimal_threshold(y_val, y_proba_val)
    console.print(f"    Umbral optimo (val): {threshold:.4f} -> F1={f1_val:.4f}")

    val_metrics = evaluate_classification(best_model, X_val, y_val, threshold)
    return best_model, threshold, val_metrics


def train_random_forest_clf(X_train, y_train, X_val, y_val, feature_type, n_iter, cv):
    """Entrena RandomForestClassifier con RandomizedSearchCV."""
    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [5, 10, 15, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }

    pipeline = build_classification_pipeline(
        feature_type=feature_type, model="rf")

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    console.print(f"    Mejores params: [dim]{search.best_params_}[/dim]")

    y_proba_val = best_model.predict_proba(X_val)[:, 1]
    threshold, f1_val = find_optimal_threshold(y_val, y_proba_val)
    console.print(f"    Umbral optimo (val): {threshold:.4f} -> F1={f1_val:.4f}")

    val_metrics = evaluate_classification(best_model, X_val, y_val, threshold)
    return best_model, threshold, val_metrics


def train_xgboost(X_train, y_train, X_val, y_val, feature_type, scale_pos_weight, n_iter, cv):
    """Entrena XGBClassifier con RandomizedSearchCV."""
    param_dist = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__gamma": [0, 0.1, 0.3],
    }

    pipeline = build_classification_pipeline(
        feature_type=feature_type, model="xgb",
        scale_pos_weight=scale_pos_weight)

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    console.print(f"    Mejores params: [dim]{search.best_params_}[/dim]")

    y_proba_val = best_model.predict_proba(X_val)[:, 1]
    threshold, f1_val = find_optimal_threshold(y_val, y_proba_val)
    console.print(f"    Umbral optimo (val): {threshold:.4f} -> F1={f1_val:.4f}")

    val_metrics = evaluate_classification(best_model, X_val, y_val, threshold)
    return best_model, threshold, val_metrics


def main():
    args = parse_args()
    ft = args.features
    os.makedirs(args.models_dir, exist_ok=True)

    console.print(Panel.fit(
        f"[bold]Entrenamiento Clasificacion Riesgo Mental[/bold]\n"
        f"FEATURE_TYPE = [cyan bold]{ft}[/cyan bold]",
        border_style="blue",
    ))

    X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.data_dir, "y_clf_train.csv")).squeeze()
    X_val = pd.read_csv(os.path.join(args.data_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(args.data_dir, "y_clf_val.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(args.data_dir, "y_clf_test.csv")).squeeze()

    console.print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    console.print(f"  Distribucion train: Sin Riesgo=[bold]{n_neg}[/bold] | Con Riesgo=[bold]{n_pos}[/bold]")
    console.print(f"  scale_pos_weight = {scale_pos_weight:.2f}\n")

    models = {}

    with make_progress() as progress:
        main_task = progress.add_task("Entrenando modelos", total=3)

        # Modelo 1: LogisticRegression
        progress.update(main_task, description="[1/3] LogisticRegression (GridSearchCV)")
        pipe, thr, metrics = train_logistic_regression(
            X_train, y_train, X_val, y_val, ft, args.cv)
        models["lr"] = (pipe, thr, metrics)
        progress.advance(main_task)

        # Modelo 2: RandomForest
        progress.update(main_task, description=f"[2/3] RandomForest (SearchCV {args.n_iter}iter)")
        pipe, thr, metrics = train_random_forest_clf(
            X_train, y_train, X_val, y_val, ft, args.n_iter, args.cv)
        models["rf"] = (pipe, thr, metrics)
        progress.advance(main_task)

        # Modelo 3: XGBoost
        progress.update(main_task, description=f"[3/3] XGBoost (SearchCV {args.n_iter}iter)")
        pipe, thr, metrics = train_xgboost(
            X_train, y_train, X_val, y_val, ft, scale_pos_weight,
            args.n_iter, args.cv)
        models["xgb"] = (pipe, thr, metrics)
        progress.advance(main_task)

    # Tabla comparativa
    table = Table(title=f"Comparativa Clasificacion (validacion, {ft})",
                  show_header=True, header_style="bold cyan")
    table.add_column("Modelo", style="bold")
    table.add_column("F1", justify="right")
    table.add_column("AUC-ROC", justify="right")
    table.add_column("Umbral", justify="right")

    labels = {"lr": "LogisticRegression", "rf": "RandomForestClassifier", "xgb": "XGBClassifier"}
    best_key, best_f1 = None, -1.0
    for key, (pipe, thr, metrics) in models.items():
        table.add_row(labels[key], f"{metrics['f1']:.4f}", f"{metrics['auc']:.4f}", f"{thr:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_key = key

    console.print(table)

    # Serializar cada modelo como artefacto {pipeline, threshold}
    for key, (pipe, thr, _) in models.items():
        artifact = {"pipeline": pipe, "threshold": thr}
        path = os.path.join(args.models_dir, f"{key}_clasificacion_riesgo_{ft}.joblib")
        joblib.dump(artifact, path, compress=3)
        console.print(f"  [dim]Guardado: {path}[/dim]")

    # Evaluar mejor modelo en test
    best_pipe, best_thr, _ = models[best_key]
    console.print(f"\n  [bold green]Mejor modelo: {labels[best_key]}[/bold green] (F1 val={best_f1:.4f})")

    test_metrics = evaluate_classification(best_pipe, X_test, y_test, best_thr)
    print_classification_report(test_metrics, "Test")

    best_artifact = {"pipeline": best_pipe, "threshold": best_thr}
    best_path = os.path.join(args.models_dir, f"best_clasificacion_riesgo_{ft}.joblib")
    joblib.dump(best_artifact, best_path, compress=3)
    console.print(f"  [dim]Guardado: {best_path}[/dim]")

    console.print(Panel.fit(
        "[bold green]Entrenamiento clasificacion completado[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
