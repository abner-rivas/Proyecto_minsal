El archivo `README.md` es la carta de presentación de tu proyecto. Para un portafolio de Machine Learning o para que un agente de IA entienda el repositorio, este documento no solo debe explicar *qué* hace el código, sino *cómo* reproducirlo y *por qué* se tomaron ciertas decisiones arquitectónicas.

Aquí tienes una plantilla estructurada y profesional, lista para que la copies y pegues en tu archivo `README.md`. He incorporado tu nombre y he hecho especial énfasis en la organización de los directorios para garantizar que cualquier persona o IA pueda reproducir el pipeline sin ambigüedades.

---

```markdown
# 📊 Análisis Predictivo GSHS 2013: Riesgos en Adolescentes Salvadoreños

Este repositorio contiene un pipeline de Machine Learning *end-to-end* diseñado para analizar los datos de la encuesta **Global School-based Student Health Survey (GSHS)** de El Salvador (2013). El sistema aplica técnicas de aprendizaje supervisado para predecir indicadores críticos de salud pública, garantizando objetividad matemática y reproducibilidad.

## 🎯 Objetivos del Proyecto

El desarrollo se divide en dos tareas analíticas principales:

* **Tarea A (Regresión - IMC):** Predicción del Índice de Masa Corporal (IMC) utilizando exclusivamente factores de comportamiento (dieta, sedentarismo, etc.), excluyendo métricas directas como el peso y la estatura. Métricas clave: `RMSE` y `R²`.
* **Tarea B (Clasificación - Salud Mental):** Detección de adolescentes con alto riesgo en salud mental (ideación suicida, tristeza severa). Implementa técnicas de balanceo de clases (SMOTE/class weights) priorizando la sensibilidad. Métricas clave: `F1-Score (clase minoritaria)` y `AUC-ROC`.

## 📁 Estructura del Directorio

Para asegurar la reproducibilidad del pipeline y evitar redundancia de código, el proyecto sigue una arquitectura estricta de separación de responsabilidades:

```text
proyecto_gshs_2013/
├── data/
│   ├── raw/                    # Dataset original (textSLV2013_Public_Use.csv) - NUNCA MODIFICAR
│   ├── processed/              # Matrices particionadas (train.csv, test.csv)
│   └── codebook.json           # Diccionario estático que mapea códigos QN a descripciones reales
├── models/                     # Artefactos binarios (.joblib) exportados tras el entrenamiento
├── notebooks/                  
│   ├── 01_exploracion.ipynb    # Análisis exploratorio (EDA) y justificación de variables
│   └── 02_prototipos.ipynb     # Pruebas de concepto interactivo
├── src/                        # Código fuente modular
│   ├── __init__.py
│   ├── data/
│   │   └── cleaner.py          # Script de limpieza de nulos y partición de datos
│   ├── features/
│   │   └── engineering.py      # Transformaciones (Scalers, One-Hot) encapsuladas en Pipelines
│   ├── models/
│   │   ├── regression.py       # Algoritmos de predicción de IMC
│   │   └── classification.py   # Algoritmos de Salud Mental
│   ├── evaluation/
│   │   └── metrics.py          # Auditoría de rendimiento (RMSE, F1-Score)
│   └── predict.py              # Script de inferencia para evaluar modelos pre-entrenados
├── reports/
│   ├── figures/                # Exportación de gráficos (ROC, Importancia de variables)
│   └── informe_tecnico.pdf     # Entregable final formato IEEE
├── Makefile                    # Orquestador maestro de ejecución por consola
└── requirements.txt            # Dependencias de Python

```

## ⚙️ Instalación y Configuración

1. Clonar el repositorio.
2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

```


3. Instalar las dependencias requeridas:
```bash
pip install -r requirements.txt

```


4. Asegurar que el archivo de datos original (`textSLV2013_Public_Use.csv`) esté ubicado dentro de `data/raw/`.

## 🚀 Uso y Ejecución del Pipeline

La orquestación del proyecto se maneja íntegramente a través del `Makefile` para asegurar que el orden de ejecución prevenga la fuga de datos (*Data Leakage*).

Desde la raíz del proyecto, ejecuta los siguientes comandos en tu terminal:

* `make data`: Limpia el dataset, imputa nulos y genera las particiones Train/Test.
* `make features`: Construye las variables derivadas (ej. IMC) y define los transformadores.
* `make train`: Entrena y ajusta los hiperparámetros de los modelos y los guarda en `models/`.
* `make evaluate`: Evalúa el rendimiento de los modelos y genera las métricas.
* `make test`: Lanza el script de inferencia (`predict.py`) para probar el sistema final contra datos aislados.
* **`make all`**: Ejecuta el ciclo de vida completo del proyecto de inicio a fin.
* `make clean`: Elimina archivos temporales, cachés y datos procesados para reiniciar el entorno.

## 🔬 Consideraciones Técnicas y Metodológicas

* **Prevención de Data Leakage:** Todo el procesamiento (escalado, imputación) se realiza *después* del split de datos y se encapsula en `Pipelines` de `scikit-learn` ajustados exclusivamente sobre el conjunto de entrenamiento.