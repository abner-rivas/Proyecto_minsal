### Estructura de Directorios Orientada a la Reproducibilidad

Esta organización aísla el código fuente de los datos y los entornos de prueba, asegurando que las transformaciones y modelos se ejecuten sistemáticamente.
```text
proyecto_gshs_2013/
├── data/
│   ├── raw/                    # Ubicación exclusiva para textSLV2013_Public_Use.csv
│   ├── processed/              # Matrices particionadas y limpias (train_X.csv, test_y.csv)
│   └── codebook.json           # Nuevo: Diccionario de metadatos estático (Mapeo QN a nombres reales)
├── models/                     # Artefactos binarios serializados (.joblib)
├── notebooks/                  # Entorno interactivo para análisis
│   ├── 01_exploracion.ipynb    # Carga de codebook.json para EDA legible
│   └── 02_prototipos.ipynb     
├── src/                        # Código fuente modular
│   ├── __init__.py
│   ├── data/
│   │   └── cleaner.py          # Limpieza, partición e inyección de metadatos
│   ├── features/
│   │   └── engineering.py      # Transformaciones y codificación (Pipelines)
│   ├── models/
│   │   ├── regression.py       # Algoritmos de predicción IMC
│   │   └── classification.py   # Algoritmos de Salud Mental
│   ├── evaluation/
│   │   └── metrics.py          # Auditoría de métricas (RMSE, F1-Score)
│   └── predict.py              # Script final de inferencia automatizada
├── reports/
│   ├── figures/                # Exportación de gráficos legibles (usando codebook)
│   └── informe_tecnico.pdf     # Entregable IEEE
├── Makefile                    # Orquestador maestro para ejecución desde consola
└── requirements.txt            # Dependencias del entorno
```
---

A continuación, presento la planificación de las fases de desarrollo actualizada, integrando explícitamente la funcionalidad de selección dinámica de variables (Q vs QN) y detallando la organización de archivos para garantizar la reproducibilidad del pipeline.

### Fases de Desarrollo Actualizadas (Soporte Dinámico Q vs QN)

#### Fase 0: Definición de Metadatos y Orquestación Base

* **Metadatos (`data/codebook.json`):** Creación del diccionario estático para mapear los códigos a descripciones reales.
* **Orquestador (`Makefile`):** Configuración del archivo maestro en la raíz del proyecto para aceptar variables de entorno (`FEATURE_TYPE ?= QN`). Esto permite ejecutar los scripts de entrenamiento inyectando el parámetro `Q` o `QN` directamente desde la consola, sin modificar el código fuente.

#### Fase 1: Ingesta, Limpieza y Partición (`src/data/cleaner.py`)

* **Limpieza de Nulos:** Reemplazo de la constante `1.79769313486232e+308` por `np.nan`.
* **Imputación:** Relleno de valores faltantes mediante la mediana o la moda.
* **Partición Ciega:** División del dataset en conjuntos de entrenamiento, validación y prueba, ejecutada antes de cualquier transformación para prevenir la fuga de datos (*Data Leakage*).

#### Fase 2: Análisis Exploratorio Legible (Directorio `notebooks/`)

* **Traducción Temporal:** Uso de `data/codebook.json` para renombrar columnas en memoria y generar un análisis univariado y bivariado comprensible.
* **Definición de Targets:** Cálculo del IMC y creación de la variable binaria de riesgo en salud mental.
* **Evaluación de Colinealidad:** Generación de matrices de correlación para evaluar el comportamiento de las variables originales frente a las recodificadas.

#### Fase 3: Ingeniería de Características Dinámica (`src/features/engineering.py`)

* **Aislamiento del Target:** Eliminación estricta de `Q4` y `Q5` del conjunto de predictores para la tarea de regresión.
* **Selector Dinámico de Columnas:** Implementación de una clase personalizada dentro de Scikit-Learn (ej. `ColumnSelector`) que filtre la matriz de datos en tiempo de ejecución para retener únicamente las columnas correspondientes al prefijo solicitado (`Q` o `QN`).
* **Pipelines de Transformación:** Configuración de `StandardScaler` y *One-Hot Encoding* ajustados exclusivamente sobre el conjunto de entrenamiento, adaptándose al volumen de columnas filtradas en el paso anterior.

#### Fase 4: Entrenamiento Parametrizado y Serialización (`src/models/regression.py` y `src/models/classification.py`)

* **Recepción de Argumentos:** Integración de la librería `argparse` en ambos scripts para capturar el argumento `--features` enviado por el `Makefile`.
* **Regresión (IMC):** Parametrización y entrenamiento de Regresión Lineal y Random Forest Regressor.
* **Clasificación (Salud Mental):** Entrenamiento de Regresión Logística y Random Forest Classifier. Integración de técnicas de balanceo (`SMOTE` o `class_weight`) dentro de validaciones con `StratifiedKFold`.
* **Almacenamiento Diferenciado:** Exportación de los modelos entrenados como archivos `.joblib` en el directorio `models/`, utilizando un sufijo en el nombre del archivo para distinguir con qué conjunto de datos fueron entrenados (ej. `regressor_imc_QN.joblib` o `classifier_mental_Q.joblib`).

#### Fase 5: Evaluación Automatizada e Inferencia (`src/evaluation/metrics.py` y `src/predict.py`)

* **Evaluación de Rendimiento:** Auditoría sobre el conjunto de prueba aislado, calculando RMSE y R² para regresión, y F1-Score junto con AUC-ROC para clasificación.
* **Interpretabilidad Traducida:** Extracción de *Feature Importance*, integrando el `codebook.json` para que los gráficos de exportación reflejen descripciones legibles.

#### Fase 6: Entregable Técnico (Directorio `reports/`)

* Consolidación de gráficos exportados en `reports/figures/`.
* Redacción del informe técnico final en formato IEEE, contrastando los resultados algorítmicos obtenidos al utilizar variables `Q` vs `QN` y formulando recomendaciones orientadas a la salud pública.