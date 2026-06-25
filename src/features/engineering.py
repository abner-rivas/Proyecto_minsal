"""
src/features/engineering.py

Define los conjuntos de features Q y QN para ambas tareas y utilidades de seleccion.
Soporta seleccion dinamica via feature_type ("Q" o "QN").

Exclusiones:
  - Q4 (estatura) y Q5 (peso): son componentes directos del target IMC (data leakage)
  - Q25 / QN25: espejo del target de clasificacion (ideacion suicida)
"""

import json
import os
import pandas as pd


# --- Variables Q (ordinales Likert, 1-7) ---
_Q_NUMBERS = [
    1, 2, 3,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24,
    26, 27, 34, 35, 36, 37, 38, 39, 40,
    44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58,
]
Q_FEATURE_COLS: list[str] = [f"Q{n}" for n in _Q_NUMBERS]  # 46 columnas


# --- Variables QN (binarias recodificadas, 0/1) ---
_QN_NUMBERS = [
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24,
    26, 27, 34, 35, 36, 37, 38, 39, 40,
    44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58,
]
QN_FEATURE_COLS: list[str] = [f"QN{n}" for n in _QN_NUMBERS]  # 43 columnas

FEATURE_COLS: list[str] = QN_FEATURE_COLS


def get_feature_cols(feature_type: str = "QN") -> list[str]:
    """Retorna la lista de columnas segun el tipo de features solicitado."""
    if feature_type.upper() == "Q":
        return Q_FEATURE_COLS
    return QN_FEATURE_COLS


def select_features(df: pd.DataFrame, feature_type: str = "QN") -> pd.DataFrame:
    """
    Retorna unicamente las columnas del feature_type presentes en df.
    No aplica ninguna transformacion -- el Pipeline de sklearn se encarga de eso.
    """
    cols = get_feature_cols(feature_type)
    available = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available)
    if missing:
        print(f"Advertencia: {len(missing)} columnas {feature_type} ausentes: {sorted(missing)}")
    return df[available].copy()


def load_codebook(path: str = "data/codebook.json") -> dict:
    """Carga el mapeo Q/QN -> descripcion semantica desde codebook.json."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_names_readable(
    feature_cols: list[str],
    codebook_path: str = "data/codebook.json",
) -> list[str]:
    """
    Traduce una lista de codigos Q/QN a sus descripciones semanticas.
    Util para etiquetas en graficos de feature importance.
    """
    codebook = load_codebook(codebook_path)
    return [codebook.get(col, col) for col in feature_cols]
