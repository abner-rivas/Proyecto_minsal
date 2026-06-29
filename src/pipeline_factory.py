"""
src/pipeline_factory.py

Constructores de Pipelines de Scikit-Learn para las dos tareas del proyecto.
Soporta dos tipos de features:

  QN (binarias 0/1): SimpleImputer(median) -> RobustScaler()
  Q  (ordinales 1-7): SimpleImputer(median) -> StandardScaler()

Y multiples estimadores por tarea:
  Regresion:     LinearRegression (baseline), RandomForestRegressor
  Clasificacion: LogisticRegression (baseline), RandomForestClassifier, XGBClassifier
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier


def build_preprocessor(feature_type: str = "QN") -> Pipeline:
    """
    Preprocesador adaptado al tipo de features:
      QN (binarias): mediana + RobustScaler (robusto a distribucion sesgada 0/1)
      Q  (ordinales): mediana + StandardScaler (preserva distancia ordinal)
    """
    if feature_type.upper() == "Q":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])


def build_regression_pipeline(
    feature_type: str = "QN",
    model: str = "rf",
    best_params: dict | None = None,
) -> Pipeline:
    """
    Pipeline para Tarea A: Prediccion de IMC.

    Args:
        feature_type: "Q" o "QN"
        model: "lr" (LinearRegression) o "rf" (RandomForestRegressor)
        best_params: hiperparametros para el estimador (solo RF)
    """
    prep = build_preprocessor(feature_type)
    if model == "lr":
        return Pipeline([
            ("preprocessor", prep),
            ("regressor", LinearRegression()),
        ])
    return Pipeline([
        ("preprocessor", prep),
        ("regressor", RandomForestRegressor(
            random_state=42,
            **(best_params or {}),
        )),
    ])


def build_classification_pipeline(
    feature_type: str = "QN",
    model: str = "xgb",
    scale_pos_weight: float = 1.0,
    best_params: dict | None = None,
) -> Pipeline:
    """
    Pipeline para Tarea B: Clasificacion de Riesgo en Salud Mental.

    Args:
        feature_type: "Q" o "QN"
        model: "lr" (LogisticRegression), "rf" (RandomForestClassifier), "xgb" (XGBClassifier)
        scale_pos_weight: ratio de desbalance (solo para XGB)
        best_params: hiperparametros para el estimador (RF o XGB)
    """
    prep = build_preprocessor(feature_type)
    if model == "lr":
        return Pipeline([
            ("preprocessor", prep),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ])
    elif model == "rf":
        return Pipeline([
            ("preprocessor", prep),
            ("classifier", RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                **(best_params or {}),
            )),
        ])
    return Pipeline([
        ("preprocessor", prep),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            **(best_params or {}),
        )),
    ])
