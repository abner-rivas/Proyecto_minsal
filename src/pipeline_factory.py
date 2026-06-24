from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

def build_health_pipeline(scale_pos_weight, best_params=None):
    """
    Construye una arquitectura limpia y empaquetada que separa flujos de datos:
    1. Flujo Numérico: Imputación por mediana + Escalado robusto ante outliers.
    2. Flujo Categórico: Imputación por moda + One-Hot Encoding seguro.
    
    Evita de forma absoluta el Data Leakage utilizando selectores dinámicos por tipo de dato.
    """
    if best_params is None:
        best_params = {}
        
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
    
    # Pipeline final consolidado
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            **best_params
        ))
    ])
    
    return pipeline