import pandas as pd
import numpy as np

def load_and_clean_survey(filepath):
    """
    Carga el dataset de la encuesta y limpia la anomalía de coma flotante.
    """
    try:
        df = pd.read_csv(filepath)
        valor_anomalo = 1.79769313486232e+308
        df = df.replace(valor_anomalo, np.nan)
        print(f"Datos cargados y limpiados exitosamente. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {filepath}")
        return None

def engineer_targets(df):
    """
    Construye el target de Regresión (IMC) y Clasificación (Riesgo Salud Mental).
    """
    df_procesado = df.copy()

    # Target A: IMC
    if 'Q4' in df_procesado.columns and 'Q5' in df_procesado.columns:
        df_procesado['IMC'] = df_procesado['Q5'] / (df_procesado['Q4'] ** 2)

    # Target B: Riesgo Salud Mental — unión de tristeza persistente (Q25==1)
    # e ideación suicida (QN24==1). Ambas señales son clínicamente independientes:
    # Q25 captura duración del episodio depresivo y QN24 captura el comportamiento
    # suicida activo. Usar solo Q25 subestimaba la clase positiva en ~55 casos.
    q25 = df_procesado.get('Q25', pd.Series(dtype=float))
    qn24 = df_procesado.get('QN24', pd.Series(dtype=float))
    df_procesado['Riesgo_Salud_Mental'] = (
        (q25 == 1) | (qn24 == 1)
    ).astype(int)

    return df_procesado