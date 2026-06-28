# Proyecto_minsal
# Sistema de Predicción de Factores de Riesgo en Adolescentes Salvadoreños

Este repositorio contiene la solución al "Desafío de Machine Learning" propuesto para el Ministerio de Salud (MINSAL) de El Salvador, basado en la Encuesta Global de Salud Escolar (GSHS) de la OMS/OPS de 2013.

## Objetivos del Proyecto
El objetivo central es desarrollar un sistema predictivo para identificar patrones de riesgo (alerta temprana) y permitir la focalización de recursos en las escuelas más vulnerables. El proyecto se divide en dos tareas principales dentro de un único pipeline de Machine Learning:

1. **Tarea A (Regresión):** Predecir el Índice de Masa Corporal (IMC) a partir de variables de comportamiento (alimentación, actividad física, etc.), evitando usar peso y estatura directamente.
2. **Tarea B (Clasificación):** Predecir el riesgo de problemas de salud mental (ej. sentimiento de soledad, ideación suicida) basándose en factores de protección y riesgo.

## Estructura del Proyecto
```text
proyecto_minsal_gshs/
│
├── data/
│   ├── raw/                 # SLV2013_Public_Use.csv (Datos originales, "sucios")
│   └── processed/           # Datos post-limpieza (sin el valor 1.79e+308) y con variables derivadas
│
├── notebooks/
│   ├── 01_limpieza_y_eda.ipynb      # Carga, reemplazo de nulos extremos y Análisis Exploratorio
│   ├── 02_feature_engineering.ipynb # Imputación, Target/One-Hot encoding y escalado
│   └── 03_modelado_y_evaluacion.ipynb # Entrenamiento (Regresión y Clasificación) y métricas
│
├── src/                     # Scripts modulares en Python
│   ├── preprocess.py        # Funciones para limpieza e imputación
│   └── train.py             # Funciones para validación cruzada y entrenamiento de modelos
│
├── docs/
│   └── reporte_IEEE_MINSAL.pdf  # Documento técnico final en formato IEEE
│
├── requirements.txt         # Dependencias necesarias para ejecutar el proyecto
└── README.md                # Este archivo
