"""
src/evaluation/metrics.py

Funciones de evaluacion y generacion de graficos para las dos tareas del proyecto GSHS 2013.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
if matplotlib.get_backend() == "agg" or not hasattr(matplotlib, "_called_from_pytest"):
    try:
        shell = get_ipython().__class__.__name__
    except NameError:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.console import console, make_progress, Panel, Table


def find_optimal_threshold(y_true, y_probs) -> tuple[float, float]:
    """
    Calcula el umbral optimo de clasificacion maximizando el F1-Score
    a partir de la curva de Precision-Recall.

    Returns:
        (umbral_optimo, f1_maximo)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    idx = np.argmax(f1_scores)
    return float(thresholds[idx]), float(f1_scores[idx])


def evaluate_regression(model, X, y) -> dict:
    """
    Evalua un modelo de regresion sobre el conjunto (X, y).

    Returns:
        dict con 'rmse', 'r2', 'y_pred'
    """
    y_pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = float(r2_score(y, y_pred))
    return {"rmse": rmse, "r2": r2, "y_pred": y_pred}


def evaluate_classification(model, X, y, threshold: float = 0.5) -> dict:
    """
    Evalua un modelo de clasificacion binaria sobre el conjunto (X, y).

    Returns:
        dict con 'f1', 'auc', 'threshold', 'report', 'confusion_matrix', 'y_proba'
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    f1 = float(f1_score(y, y_pred, zero_division=0))
    auc = float(roc_auc_score(y, y_proba))
    report = classification_report(y, y_pred, target_names=["Sin Riesgo", "Con Riesgo"])
    cm = confusion_matrix(y, y_pred)
    return {
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "report": report,
        "confusion_matrix": cm,
        "y_proba": y_proba,
    }


def print_regression_report(metrics: dict, split_name: str = "Test") -> None:
    table = Table(title=f"Regresion IMC [{split_name}]",
                  show_header=True, header_style="bold cyan")
    table.add_column("Metrica", style="bold")
    table.add_column("Valor", justify="right")
    table.add_row("RMSE (kg/m^2)", f"{metrics['rmse']:.4f}")
    table.add_row("R^2", f"{metrics['r2']:.4f}")
    console.print(table)


def print_classification_report(metrics: dict, split_name: str = "Test") -> None:
    table = Table(title=f"Clasificacion Riesgo Mental [{split_name}]",
                  show_header=True, header_style="bold cyan")
    table.add_column("Metrica", style="bold")
    table.add_column("Valor", justify="right")
    table.add_row("F1 (minoritaria)", f"{metrics['f1']:.4f}")
    table.add_row("AUC-ROC", f"{metrics['auc']:.4f}")
    table.add_row("Umbral aplicado", f"{metrics['threshold']:.4f}")
    console.print(table)
    console.print(metrics["report"])


# --- Generacion de figuras ---

def plot_feature_importance(
    model,
    feature_names: list[str],
    readable_names: list[str],
    model_label: str,
    feature_type: str,
    output_dir: str = "reports/figures",
    top_n: int = 15,
) -> str:
    """Genera barplot horizontal de feature importance con etiquetas del codebook."""
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        console.print(f"  [dim]Modelo {model_label} no soporta feature importance[/dim]")
        return ""

    n_features = min(len(importances), len(feature_names), len(readable_names))
    importances = importances[:n_features]
    names = readable_names[:n_features]
    codes = feature_names[:n_features]

    indices = np.argsort(importances)[-top_n:]
    top_names = [f"{codes[i]} - {names[i][:40]}" for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_names)), top_vals, color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Importancia")
    ax.set_title(f"Feature Importance - {model_label} ({feature_type})")
    plt.tight_layout()

    path = os.path.join(output_dir, f"feature_importance_{model_label}_{feature_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curve(
    y_true,
    y_proba,
    auc_score: float,
    model_label: str,
    feature_type: str,
    output_dir: str = "reports/figures",
) -> str:
    """Genera curva ROC y la guarda como PNG."""
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Curva ROC - {model_label} ({feature_type})")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"roc_curve_{model_label}_{feature_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm,
    model_label: str,
    feature_type: str,
    output_dir: str = "reports/figures",
) -> str:
    """Genera heatmap de la matriz de confusion."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Sin Riesgo", "Con Riesgo"],
                yticklabels=["Sin Riesgo", "Con Riesgo"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusion - {model_label} ({feature_type})")
    plt.tight_layout()

    path = os.path.join(output_dir, f"confusion_matrix_{model_label}_{feature_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residuals(
    y_true,
    y_pred,
    model_label: str,
    feature_type: str,
    output_dir: str = "reports/figures",
) -> str:
    """Genera scatter plot de residuos para regresion."""
    os.makedirs(output_dir, exist_ok=True)

    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="steelblue")
    axes[0].axhline(y=0, color="red", linestyle="--", lw=1)
    axes[0].set_xlabel("Prediccion IMC")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title("Residuos vs Prediccion")

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribucion de Residuos")

    fig.suptitle(f"Analisis de Residuos - {model_label} ({feature_type})")
    plt.tight_layout()

    path = os.path.join(output_dir, f"residuos_{model_label}_{feature_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_figures(
    reg_model,
    clf_model,
    X_test,
    y_reg_test,
    y_clf_test,
    clf_threshold: float,
    feature_names: list[str],
    readable_names: list[str],
    feature_type: str,
    reg_label: str = "best",
    clf_label: str = "best",
    output_dir: str = "reports/figures",
) -> list[str]:
    """Genera todas las figuras del informe tecnico y retorna las rutas."""
    console.print(Panel.fit(
        f"[bold]Generando figuras[/bold] ({feature_type})",
        border_style="blue",
    ))

    paths = []
    figure_steps = [
        "Feature importance (regresion)",
        "Residuos (regresion)",
        "Feature importance (clasificacion)",
        "Curva ROC",
        "Matriz de confusion",
    ]

    with make_progress() as progress:
        task = progress.add_task("Generando figuras", total=len(figure_steps))

        # Regresion: feature importance + residuos
        reg_estimator = reg_model
        if hasattr(reg_model, "named_steps"):
            reg_estimator = reg_model.named_steps.get("regressor", reg_model)

        progress.update(task, description=figure_steps[0])
        p = plot_feature_importance(
            reg_estimator, feature_names, readable_names,
            f"reg_{reg_label}", feature_type, output_dir)
        if p:
            paths.append(p)
        progress.advance(task)

        progress.update(task, description=figure_steps[1])
        reg_metrics = evaluate_regression(reg_model, X_test, y_reg_test)
        p = plot_residuals(y_reg_test, reg_metrics["y_pred"],
                           f"reg_{reg_label}", feature_type, output_dir)
        paths.append(p)
        progress.advance(task)

        # Clasificacion: feature importance + ROC + confusion matrix
        if hasattr(clf_model, "named_steps"):
            clf_estimator = clf_model.named_steps.get("classifier", clf_model)
        else:
            clf_estimator = clf_model

        progress.update(task, description=figure_steps[2])
        p = plot_feature_importance(
            clf_estimator, feature_names, readable_names,
            f"clf_{clf_label}", feature_type, output_dir)
        if p:
            paths.append(p)
        progress.advance(task)

        progress.update(task, description=figure_steps[3])
        clf_metrics = evaluate_classification(clf_model, X_test, y_clf_test, clf_threshold)
        p = plot_roc_curve(y_clf_test, clf_metrics["y_proba"], clf_metrics["auc"],
                           f"clf_{clf_label}", feature_type, output_dir)
        paths.append(p)
        progress.advance(task)

        progress.update(task, description=figure_steps[4])
        p = plot_confusion_matrix(clf_metrics["confusion_matrix"],
                                  f"clf_{clf_label}", feature_type, output_dir)
        paths.append(p)
        progress.advance(task)

    console.print(f"  Total figuras generadas: [bold]{len(paths)}[/bold]")
    for p in paths:
        console.print(f"  [dim]{p}[/dim]")

    return paths
