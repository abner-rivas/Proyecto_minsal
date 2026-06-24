# src/pipeline_factory.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

def build_preprocessor():
    """
    Construye el preprocesador unificado para datos categóricos y numéricos.
    
    NOTA TÉCNICA: Dado que en la Fase 2 (Feature Engineering) convertimos todas 
    las variables predictoras a tipo 'str', el selector 'make_column_selector' 
    las detectará automáticamente como categóricas. Si en el futuro se añaden 
    columnas numéricas (ej. 'int64'), el flujo numérico aplicará RobustScaler.
    """
    # Pipeline especializado para variables numéricas (edad, peso, etc.)
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    # Pipeline especializado para variables categóricas o de texto
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
    
    # El procesador inteligente: Aplica cada flujo según el tipo de columna automáticamente
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category', 'bool']))
    ], remainder='passthrough')
    
    return preprocessor


def build_classification_pipeline(scale_pos_weight, best_params=None):
    """
    Construye el Pipeline para la Tarea B: Clasificación de Riesgo en Salud Mental.
    Utiliza XGBoost con balanceo de clases interno (scale_pos_weight).
    
    Args:
        scale_pos_weight (float): Relación entre la clase negativa y la positiva.
        best_params (dict, optional): Hiperparámetros extra para el clasificador.
    """
    if best_params is None:
        best_params = {}
        
    return Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            **best_params
        ))
    ])


def build_regression_pipeline(best_params=None):
    """
    Construye el Pipeline para la Tarea A: Regresión del Índice de Masa Corporal (IMC).
    Utiliza RandomForestRegressor como modelo base, el cual maneja bien la no linealidad.
    
    Args:
        best_params (dict, optional): Hiperparámetros extra para el regresor.
    """
    if best_params is None:
        best_params = {}
        
    return Pipeline([
        ('preprocessor', build_preprocessor()),
        ('regressor', RandomForestRegressor(random_state=42, **best_params))
    ])