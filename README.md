# Análisis Predictivo GSHS 2013: Riesgos en Adolescentes Salvadoreños

Pipeline de Machine Learning *end-to-end* para analizar los datos de la **Global School-based Student Health Survey (GSHS)** de El Salvador (2013). El sistema aplica aprendizaje supervisado para predecir indicadores críticos de salud pública con reproducibilidad garantizada.

## Objetivos

| Tarea | Objetivo | Métricas |
|---|---|---|
| **A - Regresión (IMC)** | Predecir el Índice de Masa Corporal usando solo variables conductuales (sin peso ni estatura) | RMSE, R² |
| **B - Clasificación (Salud Mental)** | Detectar adolescentes con alto riesgo en salud mental (ideación suicida) | F1-Score (clase minoritaria), AUC-ROC |

## Estructura del Proyecto

```text
ml-minsal-lab02/
├── data/
│   ├── raw/                        # SLV2013_Public_Use.csv (original, no modificar)
│   ├── processed/                  # Splits generados: X_train, X_val, X_test, y_*
│   └── codebook.json               # Mapeo Q1-Q58 y QN6-QN58 a descripciones semánticas
├── models/                         # Artefactos .joblib con sufijo _Q o _QN
├── notebooks/
│   ├── 01_exploracion.ipynb        # EDA unificado (variables Q y QN)
│   └── 02_prototipos.ipynb         # Experimentación interactiva con modelos
├── src/
│   ├── console.py                  # Consola rich (UTF-8 compatible con Windows)
│   ├── pipeline_factory.py         # Builders de pipelines sklearn (preprocesador + modelo)
│   ├── data/
│   │   └── cleaner.py              # Limpieza, targets (IMC, Riesgo), partición 70/20/10
│   ├── features/
│   │   └── engineering.py          # Columnas Q (46) y QN (43), selector dinámico
│   ├── models/
│   │   ├── regression.py           # LinearRegression + RandomForest (2 modelos)
│   │   └── classification.py       # LogisticRegression + RandomForest + XGBoost (3 modelos)
│   ├── evaluation/
│   │   └── metrics.py              # Evaluación, reportes, generación de figuras
│   └── predict.py                  # Inferencia y evaluación final
├── reports/
│   ├── figures/                    # Gráficos exportados (ROC, confusión, importance, residuos)
│   └── informe_tecnico.pdf         # Entregable IEEE
├── docs/
│   └── Reporte_Tecnico_MINSAL_IEEE.tex  # Fuente LaTeX del informe
├── Makefile                        # Orquestador del pipeline
└── requirements.txt                # Dependencias Python
```

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd ml-minsal-lab02

# 2. Crear entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

**Requisito:** El archivo `SLV2013_Public_Use.csv` debe estar en `data/raw/`.

## Uso del Pipeline (Makefile)

El pipeline se controla con `make` y la variable `FEATURE_TYPE` (default: `QN`).

### Ejecución completa

```bash
# Pipeline completo con features QN (binarias, default)
make all

# Pipeline completo con features Q (ordinales Likert)
make all FEATURE_TYPE=Q
```

### Targets individuales

| Comando | Descripción |
|---|---|
| `make data` | Limpia dataset, construye targets (IMC, Riesgo), genera splits 70/20/10 |
| `make features` | Muestra las columnas del feature set seleccionado |
| `make train` | Entrena todos los modelos (regresión + clasificación) |
| `make train-reg` | Solo entrena modelos de regresión (LR + RF) |
| `make train-clf` | Solo entrena modelos de clasificación (LogReg + RF + XGB) |
| `make evaluate` | Evalúa mejores modelos en test set y genera figuras en `reports/figures/` |
| `make test` | Alias de evaluate |
| `make clean` | Elimina datos procesados, modelos y figuras |
| `make help` | Muestra ayuda con todos los targets disponibles |

Todos los targets aceptan `FEATURE_TYPE=Q` o `FEATURE_TYPE=QN`:

```bash
make train-clf FEATURE_TYPE=Q
make evaluate FEATURE_TYPE=QN
```

## Uso Directo de Scripts

Cada script acepta parámetros vía `argparse` para ejecución independiente del Makefile.

### Limpieza y partición de datos

```bash
python src/data/cleaner.py --features QN
python src/data/cleaner.py --features Q --raw-path data/raw/SLV2013_Public_Use.csv --output-dir data/processed
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--features` | `QN` | Tipo de features: `Q` (ordinales 1-7) o `QN` (binarias 0/1) |
| `--raw-path` | `data/raw/SLV2013_Public_Use.csv` | Ruta al CSV crudo |
| `--output-dir` | `data/processed` | Directorio de salida para los splits |

### Entrenamiento de regresión (Tarea A - IMC)

```bash
python src/models/regression.py --features QN
python src/models/regression.py --features Q --n-iter 50 --cv 10
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--features` | `QN` | Tipo de features: `Q` o `QN` |
| `--n-iter` | `20` | Iteraciones de RandomizedSearchCV para RandomForest |
| `--cv` | `5` | Número de folds de validación cruzada |
| `--data-dir` | `data/processed` | Directorio con los splits |
| `--models-dir` | `models` | Directorio donde guardar los modelos |

Modelos entrenados:
- `models/lr_regresion_imc_{Q|QN}.joblib` - LinearRegression (baseline)
- `models/rf_regresion_imc_{Q|QN}.joblib` - RandomForest (optimizado)
- `models/best_regresion_imc_{Q|QN}.joblib` - Mejor modelo por RMSE en validación

### Entrenamiento de clasificación (Tarea B - Salud Mental)

```bash
python src/models/classification.py --features QN
python src/models/classification.py --features Q --n-iter 30 --cv 5
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--features` | `QN` | Tipo de features: `Q` o `QN` |
| `--n-iter` | `20` | Iteraciones de RandomizedSearchCV para RF y XGBoost |
| `--cv` | `5` | Folds de validación cruzada (StratifiedKFold) |
| `--data-dir` | `data/processed` | Directorio con los splits |
| `--models-dir` | `models` | Directorio donde guardar los modelos |

Modelos entrenados:
- `models/lr_clasificacion_riesgo_{Q|QN}.joblib` - LogisticRegression (class_weight="balanced")
- `models/rf_clasificacion_riesgo_{Q|QN}.joblib` - RandomForest (class_weight="balanced")
- `models/xgb_clasificacion_riesgo_{Q|QN}.joblib` - XGBoost (scale_pos_weight)
- `models/best_clasificacion_riesgo_{Q|QN}.joblib` - Mejor modelo por F1 en validación

Cada artefacto de clasificación es un dict `{"pipeline": ..., "threshold": ...}` con el umbral óptimo calibrado.

### Evaluación e inferencia

```bash
# Evaluar ambas tareas en test set con figuras
python src/predict.py --task all --evaluate --features QN

# Evaluar solo regresión, sin generar figuras
python src/predict.py --task reg --evaluate --features Q --no-figures

# Evaluar solo clasificación
python src/predict.py --task clf --evaluate --features QN

# Inferencia sobre datos nuevos
python src/predict.py --task reg --input datos_nuevos.csv --output predicciones.csv --features QN
python src/predict.py --task clf --input datos_nuevos.csv --output predicciones.csv --features QN
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--task` | `all` | Tarea a ejecutar: `reg`, `clf` o `all` |
| `--features` | `QN` | Tipo de features: `Q` o `QN` |
| `--evaluate` | - | Evaluar sobre el test set (genera métricas y figuras) |
| `--no-figures` | - | Omitir generación de figuras PNG |
| `--input` | - | CSV de entrada para inferencia (obligatorio sin `--evaluate`) |
| `--output` | - | CSV de salida con predicciones |
| `--data-dir` | `data/processed` | Directorio con los splits de test |
| `--models-dir` | `models` | Directorio con los modelos entrenados |

## Variables Q vs QN

El pipeline soporta dos conjuntos de features seleccionables con `FEATURE_TYPE`:

| Aspecto | Q (ordinales) | QN (binarias) |
|---|---|---|
| Columnas | Q1-Q58 (46 features, sin Q4, Q5, Q25) | QN6-QN58 (43 features, sin QN25) |
| Valores | Likert ordinal (1-7) | Binario (0/1) |
| Preprocesamiento | SimpleImputer(median) + StandardScaler | SimpleImputer(median) + RobustScaler |
| Ventaja | Preserva granularidad ordinal | Interpretación directa, menor dimensionalidad |

Variables excluidas: Q4 (estatura) y Q5 (peso) para evitar data leakage en regresión; Q25/QN25 (ideación suicida) porque es el target de clasificación.

## Consideraciones Técnicas

- **Data Leakage:** Imputación y escalado encapsulados en Pipelines sklearn, ajustados solo sobre train.
- **Partición:** 70% train / 20% validación / 10% test con `random_state=42`.
- **Desbalance de clases:** ~7.5:1 ratio, mitigado con `class_weight="balanced"` (LogReg/RF) y `scale_pos_weight` (XGBoost).
- **Umbral óptimo:** Calculado en validación maximizando F1 vía curva Precisión-Recall, serializado junto al modelo.
- **Serialización:** `.joblib` con `compress=3`, sufijo `_Q` o `_QN` para diferenciar artefactos.
- **Consola:** Salida profesional con `rich` (paneles, tablas, barras de progreso), compatible con Windows cp1252 vía wrapper UTF-8.

## Figuras Generadas

Al ejecutar `make evaluate`, se generan en `reports/figures/`:

| Figura | Archivo |
|---|---|
| Feature Importance (regresión) | `feature_importance_reg_best_{Q\|QN}.png` |
| Feature Importance (clasificación) | `feature_importance_clf_best_{Q\|QN}.png` |
| Curva ROC | `roc_curve_clf_best_{Q\|QN}.png` |
| Matriz de Confusión | `confusion_matrix_clf_best_{Q\|QN}.png` |
| Análisis de Residuos | `residuos_reg_best_{Q\|QN}.png` |

## Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
rich>=13.0.0
joblib>=1.3.0
jupyter>=1.0.0
notebook>=7.0.0
```
