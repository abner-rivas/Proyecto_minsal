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
        
    # Target B: Riesgo Salud Mental
    if 'Q25' in df_procesado.columns:
        df_procesado['Riesgo_Salud_Mental'] = df_procesado['Q25'].apply(lambda x: 1 if x == 1 else 0)
        
    return df_procesado