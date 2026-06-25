"""
src/models/regression.py

Entrenamiento y serializacion de modelos de Regresion (Tarea A: Prediccion de IMC).
Entrena 2 modelos obligatorios (desafio.md):
  1. LinearRegression (baseline)
  2. RandomForestRegressor (con RandomizedSearchCV)

Uso:
    python src/models/regression.py --features QN
    python src/models/regression.py --features Q --n-iter 30 --cv 5
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.console import console, make_progress, make_spinner, Panel, Table
from src.pipeline_factory import build_regression_pipeline
from src.evaluation.metrics import evaluate_regression, print_regression_report


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento Regresion IMC")
    parser.add_argument("--features", choices=["Q", "QN"], default="QN",
                        help="Tipo de features: Q (ordinales) o QN (binarias)")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Iteraciones de RandomizedSearchCV para RF")
    parser.add_argument("--cv", type=int, default=5,
                        help="Numero de folds de validacion cruzada")
    return parser.parse_args()


def train_linear_regression(X_train, y_train, X_val, y_val, feature_type):
    """Entrena LinearRegression como modelo baseline."""
    pipeline = build_regression_pipeline(feature_type=feature_type, model="lr")
    pipeline.fit(X_train, y_train)
    val_metrics = evaluate_regression(pipeline, X_val, y_val)
    return pipeline, val_metrics


def train_random_forest(X_train, y_train, X_val, y_val, feature_type, n_iter, cv):
    """Entrena RandomForestRegressor con RandomizedSearchCV."""
    param_dist = {
        "regressor__n_estimators": [100, 200, 300, 500],
        "regressor__max_depth": [5, 10, 15, 20, None],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", "log2", 0.5],
    }

    pipeline = build_regression_pipeline(feature_type=feature_type, model="rf")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    console.print(f"    Mejores params: [dim]{search.best_params_}[/dim]")

    val_metrics = evaluate_regression(best_model, X_val, y_val)
    return best_model, val_metrics


def main():
    args = parse_args()
    ft = args.features
    os.makedirs(args.models_dir, exist_ok=True)

    console.print(Panel.fit(
        f"[bold]Entrenamiento Regresion IMC[/bold]\n"
        f"FEATURE_TYPE = [cyan bold]{ft}[/cyan bold]",
        border_style="blue",
    ))

    X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.data_dir, "y_reg_train.csv")).squeeze()
    X_val = pd.read_csv(os.path.join(args.data_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(args.data_dir, "y_reg_val.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(args.data_dir, "y_reg_test.csv")).squeeze()

    console.print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}\n")

    models = {}

    with make_progress() as progress:
        main_task = progress.add_task("Entrenando modelos", total=2)

        # Modelo 1: LinearRegression
        progress.update(main_task, description="[1/2] LinearRegression (baseline)")
        pipeline, val_metrics = train_linear_regression(
            X_train, y_train, X_val, y_val, ft)
        models["lr"] = (pipeline, val_metrics)
        console.print(f"    LR val -> RMSE={val_metrics['rmse']:.4f}  R^2={val_metrics['r2']:.4f}")
        progress.advance(main_task)

        # Modelo 2: RandomForest
        progress.update(main_task, description=f"[2/2] RandomForest (SearchCV {args.n_iter}iter, {args.cv}-fold)")
        pipeline, val_metrics = train_random_forest(
            X_train, y_train, X_val, y_val, ft, args.n_iter, args.cv)
        models["rf"] = (pipeline, val_metrics)
        console.print(f"    RF val -> RMSE={val_metrics['rmse']:.4f}  R^2={val_metrics['r2']:.4f}")
        progress.advance(main_task)

    # Tabla comparativa
    table = Table(title=f"Comparativa Regresion (validacion, {ft})",
                  show_header=True, header_style="bold cyan")
    table.add_column("Modelo", style="bold")
    table.add_column("RMSE", justify="right")
    table.add_column("R^2", justify="right")

    best_key, best_rmse = None, float("inf")
    for key, (pipe, metrics) in models.items():
        label = "LinearRegression" if key == "lr" else "RandomForest"
        rmse_str = f"{metrics['rmse']:.4f}"
        r2_str = f"{metrics['r2']:.4f}"
        table.add_row(label, rmse_str, r2_str)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_key = key

    console.print(table)

    # Serializar
    for key, (pipe, _) in models.items():
        path = os.path.join(args.models_dir, f"{key}_regresion_imc_{ft}.joblib")
        joblib.dump(pipe, path, compress=3)
        console.print(f"  [dim]Guardado: {path}[/dim]")

    # Mejor modelo en test
    best_model = models[best_key][0]
    best_label = "LinearRegression" if best_key == "lr" else "RandomForest"
    console.print(f"\n  [bold green]Mejor modelo: {best_label}[/bold green] (RMSE val={best_rmse:.4f})")

    test_metrics = evaluate_regression(best_model, X_test, y_test)
    print_regression_report(test_metrics, "Test")

    best_path = os.path.join(args.models_dir, f"best_regresion_imc_{ft}.joblib")
    joblib.dump(best_model, best_path, compress=3)
    console.print(f"  [dim]Guardado: {best_path}[/dim]")

    console.print(Panel.fit(
        "[bold green]Entrenamiento regresion completado[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
