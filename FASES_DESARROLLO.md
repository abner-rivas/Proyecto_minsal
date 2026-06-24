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

### Fases de Desarrollo

#### Fase 0: Definición de Metadatos (`data/codebook.json`)
Antes de tocar el código en Python, el agente o desarrollador debe construir la matriz de referencia semántica.

* **Crear el archivo JSON con pares clave-valor (ej. `"QN24": "Ideacion_Suicida"`).** Esto permitirá que las fases posteriores traduzcan las columnas incomprensibles en variables de salud pública accionables.

#### Fase 1: Ingesta, Limpieza y Partición (`src/data/cleaner.py`)

El preprocesamiento inicial debe garantizar la sanidad matemática de la matriz base antes de cualquier modelado.

* **Limpieza de nulos:** Identificar la constante `1.79769313486232e+308` y sustituirla por `np.nan`.
* **Imputación:** Rellenar los valores faltantes utilizando la mediana o la moda según la naturaleza de la variable.
* **Partición Estratégica:** Dividir los datos en conjuntos de *Train*, *Validation* y *Test*. Esta separación estricta previene el error crítico de fuga de datos (*Data Leakage*) al asegurar que el modelo no vea datos futuros durante las transformaciones.

#### Fase 2: Análisis Exploratorio de Datos (Directorio `notebooks/`)

Exploración de la distribución estadística y topología de los datos en un entorno interactivo.

* **Variables Objetivo:** Calcular el IMC a partir del Peso y la Estatura. Construir la variable binaria `Riesgo_Salud_Mental` basándose en indicadores como ideación suicida o tristeza persistente.
* **Análisis Multivariado:** Utilizar histogramas y matrices de correlación para analizar relaciones bivariadas.
* **Selección Q vs QN:** Justificar estadísticamente la selección entre variables de respuestas originales (Q) y numéricas recodificadas (QN) para evadir la multicolinealidad.

#### Fase 3: Ingeniería de Características (`src/features/engineering.py`)

Creación de señales predictivas empaquetadas en un Pipeline de Scikit-Learn.

* **Restricción del Target:** Eliminar explícitamente `Q4` (Estatura) y `Q5` (Peso) del vector de características predictoras para la predicción de IMC.
* **Escalado y Codificación:** Aplicar `StandardScaler` o `RobustScaler` a variables continuas, y codificación categórica como *One-Hot Encoding* a variables cualitativas, asegurando que estas transformaciones se ajusten únicamente sobre el conjunto de entrenamiento.

#### Fase 4: Entrenamiento y Almacenamiento (`src/models/`)

Ajuste de los algoritmos predictivos y serialización de los pipelines hacia el directorio de modelos.

* **Modelos de Regresión (IMC):** Entrenar Regresión Lineal y Random Forest Regressor.
* **Modelos de Clasificación:** Entrenar Regresión Logística y Random Forest Classifier, implementando técnicas para tratar el desbalance, como `SMOTE` o `class_weight="balanced"`. Se utilizará `StratifiedKFold` de forma obligatoria para mantener las proporciones de clase constantes en los pliegues de validación.
* **Persistencia:** Exportar los modelos completos y optimizados mediante `joblib` y almacenarlos explícitamente en `models/`.

#### Fase 5: Evaluación y Pruebas Reales (`src/evaluation/metrics.py` y `src/predict.py`)

Medición rigurosa de los resultados matemáticos y la capacidad de generalización del sistema.

* **Métricas Evaluativas:** Para la tarea de regresión, reportar principalmente el RMSE y R². Para clasificación, descartar la métrica de *Accuracy* debido al desbalance y enfocarse en el F1-Score de la clase minoritaria y la curva AUC-ROC.
* **Pruebas Centralizadas:** Ejecutar `src/predict.py` mediante el `Makefile` para cargar el modelo pre-entrenado desde `models/` e inferir de forma aislada sobre el archivo de pruebas ubicado en `data/processed/`.
* **Interpretabilidad:** Extraer *Feature Importance* y coeficientes para identificar los factores biológicos y sociales dominantes en las predicciones.

#### Fase 6: Entregable Final (Directorio `reports/`)

* Consolidar los artefactos visuales (matrices de confusión, gráficos de la importancia de variables y análisis residual) en el directorio `reports/figures/`.
* Redactar el documento técnico final en formato IEEE dirigido al Ministro de Salud, detallando la metodología empleada, resultados algorítmicos y las recomendaciones de política pública derivadas del análisis.