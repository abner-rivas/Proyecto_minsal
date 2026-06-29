"""
src/data/cleaner.py

Carga, limpieza, construccion de targets y particion del dataset GSHS 2013.
Soporta seleccion dinamica de features Q o QN via parametro --features.
Todas las operaciones de imputacion de target se ajustan unicamente sobre
el conjunto de entrenamiento para prevenir data leakage.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.console import console, make_progress, make_spinner, Panel, Table
from src.features.engineering import get_feature_cols

ANOMALOUS_VALUE = 1.79769313486232e+308
RANDOM_STATE = 42


def load_and_clean_survey(filepath: str) -> pd.DataFrame:
    """Carga el CSV crudo y reemplaza el valor anomalo de punto flotante por NaN.

    El valor anomalo aparece en el CSV como texto, lo que fuerza a las columnas
    a tipo ``object``. Se reemplaza tanto la version float como la string y luego
    se coercionan todas las columnas a numerico (el dataset GSHS es completamente
    numerico una vez removido el centinela).
    """
    df = pd.read_csv(filepath)
    df = df.replace([ANOMALOUS_VALUE, str(ANOMALOUS_VALUE)], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    console.print(f"  Cargado: [bold]{df.shape[0]}[/bold] filas x [bold]{df.shape[1]}[/bold] columnas")
    return df


def engineer_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las dos variables objetivo:
      - IMC = Q5 / Q4^2 (regresion, continua)
      - Riesgo_Salud_Mental: 1 si Q25==1 (ideacion suicida), 0 en caso contrario
    """
    out = df.copy()

    if "Q4" in out.columns and "Q5" in out.columns:
        q4 = pd.to_numeric(out["Q4"], errors="coerce")
        q5 = pd.to_numeric(out["Q5"], errors="coerce")
        out["IMC"] = q5 / (q4 ** 2)
        console.print(f"  IMC calculado: [bold]{out['IMC'].notna().sum()}[/bold] valores validos")

    if "Q25" in out.columns:
        q25 = pd.to_numeric(out["Q25"], errors="coerce")
        out["Riesgo_Salud_Mental"] = (q25 == 1).astype(int)
        n0 = int((out["Riesgo_Salud_Mental"] == 0).sum())
        n1 = int((out["Riesgo_Salud_Mental"] == 1).sum())
        console.print(f"  Riesgo: Sin Riesgo=[bold]{n0}[/bold] | Con Riesgo=[bold]{n1}[/bold]")

    return out


def split_and_save(
    df: pd.DataFrame,
    output_dir: str = "data/processed",
    feature_type: str = "QN",
) -> None:
    """
    Particiona el dataset en train (70%) / val (20%) / test (10%).
    La imputacion del target IMC se ajusta (fit) solo sobre train.
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_cols = get_feature_cols(feature_type)

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y_reg = df["IMC"].copy()
    y_clf = df["Riesgo_Salud_Mental"].copy()

    console.print(f"  Features: [bold]{len(available)}[/bold] columnas [cyan]{feature_type}[/cyan]")
    console.print(f"  Nulos IMC: [bold]{y_reg.isnull().sum()}[/bold]")

    X_train, X_temp, y_reg_train, y_reg_temp, y_clf_train, y_clf_temp = train_test_split(
        X, y_reg, y_clf, test_size=0.30, random_state=RANDOM_STATE,
    )
    X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = train_test_split(
        X_temp, y_reg_temp, y_clf_temp, test_size=1 / 3, random_state=RANDOM_STATE,
    )

    imputer_imc = SimpleImputer(strategy="median")
    y_reg_train = pd.Series(
        imputer_imc.fit_transform(y_reg_train.values.reshape(-1, 1)).flatten(),
        index=y_reg_train.index, name="IMC",
    )
    y_reg_val = pd.Series(
        imputer_imc.transform(y_reg_val.values.reshape(-1, 1)).flatten(),
        index=y_reg_val.index, name="IMC",
    )
    y_reg_test = pd.Series(
        imputer_imc.transform(y_reg_test.values.reshape(-1, 1)).flatten(),
        index=y_reg_test.index, name="IMC",
    )

    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_reg_train": y_reg_train, "y_reg_val": y_reg_val, "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train, "y_clf_val": y_clf_val, "y_clf_test": y_clf_test,
    }

    with make_progress() as progress:
        task = progress.add_task("Guardando splits", total=len(splits))
        for name, data in splits.items():
            path = os.path.join(output_dir, f"{name}.csv")
            data.to_csv(path, index=False)
            progress.advance(task)

    table = Table(title="Particion 70/20/10", show_header=True, header_style="bold cyan")
    table.add_column("Conjunto", style="bold")
    table.add_column("Filas", justify="right")
    table.add_column("Columnas", justify="right")
    table.add_row("Train", str(X_train.shape[0]), str(X_train.shape[1]))
    table.add_row("Validacion", str(X_val.shape[0]), str(X_val.shape[1]))
    table.add_row("Test", str(X_test.shape[0]), str(X_test.shape[1]))
    console.print(table)


def run_pipeline(
    raw_path: str = "data/raw/SLV2013_Public_Use.csv",
    output_dir: str = "data/processed",
    feature_type: str = "QN",
) -> None:
    """Punto de entrada principal: carga -> targets -> particion -> guardado."""
    console.print(Panel.fit(
        f"[bold]Pipeline de Datos GSHS 2013[/bold]\n"
        f"FEATURE_TYPE = [cyan bold]{feature_type}[/cyan bold]",
        border_style="blue",
    ))

    with make_spinner() as progress:
        t = progress.add_task("Cargando dataset crudo")
        df = load_and_clean_survey(raw_path)
        progress.update(t, description="[bold green]Dataset cargado")

    with make_spinner() as progress:
        t = progress.add_task("Construyendo variables objetivo")
        df = engineer_targets(df)
        progress.update(t, description="[bold green]Targets construidos")

    split_and_save(df, output_dir, feature_type)

    clean_path = os.path.join(output_dir, "SLV2013_Limpios_Targets.csv")
    df.to_csv(clean_path, index=False)
    console.print(f"\n  Dataset completo: [dim]{clean_path}[/dim]")

    console.print(Panel.fit(
        "[bold green]Pipeline de datos completado[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de datos GSHS 2013")
    parser.add_argument(
        "--features", choices=["Q", "QN"], default="QN",
        help="Tipo de features: Q (ordinales) o QN (binarias recodificadas)",
    )
    parser.add_argument("--raw-path", default="data/raw/SLV2013_Public_Use.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    run_pipeline(args.raw_path, args.output_dir, args.features)
