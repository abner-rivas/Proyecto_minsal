from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

def build_health_pipeline(scale_pos_weight, best_params=None):
    """
    Construye un Pipeline empaquetado de Scikit-Learn que incluye:
    1. Imputación de nulos (Moda)
    2. Codificación Categórica (One-Hot Encoding seguro)
    3. Escalado de características (RobustScaler)
    4. Clasificador Avanzado (XGBoost)
    
    Evita por completo el Data Leakage al encapsular las transformaciones.
    """
    if best_params is None:
        best_params = {}
        
    # Definimos la secuencia de pasos secuenciales
    pipeline = Pipeline([
        # Paso 1: Tratar nulos restantes con la moda
        ('imputer', SimpleImputer(strategy='most_frequent')),
        
        # Paso 2: Convertir categorías numéricas a dummis (One-Hot) de forma segura
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)),
        
        # Paso 3: Escalar los rangos de manera robusta ante outliers
        ('scaler', RobustScaler()),
        
        # Paso 4: El motor de predicción optimizado con sus pesos nativos
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            **best_params
        ))
    ])
    
    return pipeline