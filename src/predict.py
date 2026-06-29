"""
src/predict.py

Script de inferencia y evaluacion final sobre el conjunto de test.
Soporta seleccion de features Q o QN y carga el mejor modelo correspondiente.

Modos de uso:
    python src/predict.py --task reg --evaluate --features QN
    python src/predict.py --task clf --evaluate --features Q
    python src/predict.py --task all --evaluate --features QN
    python src/predict.py --task reg --input data/raw/nuevos_datos.csv --output predictions.csv --features QN
"""

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.console import console, make_spinner, Panel
from src.features.engineering import select_features, get_feature_cols, feature_names_readable
from src.evaluation.metrics import (
    evaluate_regression,
    evaluate_classification,
    print_regression_report,
    print_classification_report,
    generate_all_figures,
)

DATA_DIR = "data/processed"


def model_path(models_dir, prefix, feature_type):
    return os.path.join(models_dir, f"best_{prefix}_{feature_type}.joblib")


def load_reg_model(models_dir, feature_type):
    path = model_path(models_dir, "regresion_imc", feature_type)
    return joblib.load(path)


def load_clf_artifact(models_dir, feature_type):
    path = model_path(models_dir, "clasificacion_riesgo", feature_type)
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        return artifact["pipeline"], artifact["threshold"]
    return artifact, 0.5


def predict_regression(input_path, feature_type, models_dir, output_path=None):
    model = load_reg_model(models_dir, feature_type)
    df = pd.read_csv(input_path)
    X = select_features(df, feature_type)
    preds = pd.Series(model.predict(X), name="IMC_predicho")
    if output_path:
        preds.to_csv(output_path, index=False)
        console.print(f"  [dim]Predicciones guardadas: {output_path}[/dim]")
    return preds


def predict_classification(input_path, feature_type, models_dir, output_path=None):
    model, threshold = load_clf_artifact(models_dir, feature_type)
    df = pd.read_csv(input_path)
    X = select_features(df, feature_type)
    proba = model.predict_proba(X)[:, 1]
    preds = pd.Series((proba >= threshold).astype(int), name="Riesgo_Salud_Mental_predicho")
    if output_path:
        pd.DataFrame({"probabilidad": proba, "prediccion": preds}).to_csv(output_path, index=False)
        console.print(f"  [dim]Predicciones guardadas: {output_path}[/dim]")
    return preds


def evaluate_all(task, feature_type, data_dir, models_dir, figures=True):
    ft = feature_type

    console.print(Panel.fit(
        f"[bold]Evaluacion Final[/bold]\n"
        f"FEATURE_TYPE = [cyan bold]{ft}[/cyan bold]",
        border_style="blue",
    ))

    reg_model, clf_model, clf_threshold = None, None, 0.5
    X_test, y_reg_test, y_clf_test = None, None, None

    if task in ("reg", "all"):
        with make_spinner() as progress:
            t = progress.add_task("Evaluando regresion")
            reg_model = load_reg_model(models_dir, ft)
            X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
            y_reg_test = pd.read_csv(os.path.join(data_dir, "y_reg_test.csv")).squeeze()
            metrics = evaluate_regression(reg_model, X_test, y_reg_test)
            progress.update(t, description="[bold green]Regresion evaluada")
        print_regression_report(metrics, "Test")

    if task in ("clf", "all"):
        with make_spinner() as progress:
            t = progress.add_task("Evaluando clasificacion")
            clf_model, clf_threshold = load_clf_artifact(models_dir, ft)
            if X_test is None:
                X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
            y_clf_test = pd.read_csv(os.path.join(data_dir, "y_clf_test.csv")).squeeze()
            metrics = evaluate_classification(clf_model, X_test, y_clf_test, clf_threshold)
            progress.update(t, description="[bold green]Clasificacion evaluada")
        print_classification_report(metrics, "Test")

    if figures and reg_model is not None and clf_model is not None:
        feature_cols = get_feature_cols(ft)
        readable = feature_names_readable(feature_cols)
        generate_all_figures(
            reg_model=reg_model,
            clf_model=clf_model,
            X_test=X_test,
            y_reg_test=y_reg_test,
            y_clf_test=y_clf_test,
            clf_threshold=clf_threshold,
            feature_names=feature_cols,
            readable_names=readable,
            feature_type=ft,
        )

    console.print(Panel.fit(
        "[bold green]Evaluacion completada[/bold green]",
        border_style="green",
    ))


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia y evaluacion GSHS 2013")
    parser.add_argument("--task", choices=["reg", "clf", "all"], default="all",
                        help="Tarea: reg (IMC), clf (Salud Mental), all (ambas)")
    parser.add_argument("--features", choices=["Q", "QN"], default="QN",
                        help="Tipo de features: Q o QN")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluar sobre el test set")
    parser.add_argument("--no-figures", action="store_true",
                        help="Omitir generacion de figuras")
    parser.add_argument("--input", help="Ruta al CSV de entrada para inferencia")
    parser.add_argument("--output", help="Ruta al CSV de salida para predicciones")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.evaluate:
        evaluate_all(
            args.task, args.features, args.data_dir, args.models_dir,
            figures=not args.no_figures)
    else:
        if not args.input:
            console.print("[bold red]Error:[/bold red] --input es obligatorio cuando no se usa --evaluate")
            sys.exit(1)
        if args.task == "reg":
            predict_regression(args.input, args.features, args.models_dir, args.output)
        elif args.task == "clf":
            predict_classification(args.input, args.features, args.models_dir, args.output)
        else:
            predict_regression(args.input, args.features, args.models_dir)
            predict_classification(args.input, args.features, args.models_dir, args.output)


if __name__ == "__main__":
    main()
