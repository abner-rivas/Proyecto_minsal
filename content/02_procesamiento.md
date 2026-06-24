
# CONTEXTO DE REFERENCIA: FLUJO DE TRABAJO Y TÉCNICAS EN MACHINE LEARNING

Este documento define las etapas, herramientas, técnicas y mejores prácticas del ciclo de vida de un proyecto de Machine Learning y Big Data. Debe ser utilizado como marco de referencia para evaluar, proponer o auditar soluciones analíticas.

## 1. FUNDAMENTOS Y PREPARACIÓN DE DATOS
El rendimiento de un modelo depende directamente de la calidad del conjunto de datos ("dataset inteligente"). Un mal dataset produce un mal modelo, por lo que los datos crudos deben ser limpiados y estructurados.

*   **NumPy:** Estándar para procesamiento numérico y álgebra lineal. Transforma datos a forma matemática y permite operaciones vectoriales masivas sin ciclos `for` (ej. cálculo de medias, desviaciones estándar y normalización Z-Score).
*   **Pandas:** Herramienta principal para datos tabulares. Facilita la imputación de nulos (`fillna`), eliminación de ruido, filtrado de registros y transformación de tipos de datos.
*   **Polars:** Alternativa para grandes volúmenes de datos. Ofrece procesamiento ultrarrápido, ejecución multihilo y evaluación eficiente, ideal para "Big Data" tabular.

## 2. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
Fase crítica donde el conocimiento del negocio se convierte en señales predictivas robustas, mitigando el ruido y mejorando el rendimiento del algoritmo.

*   **Variables Derivadas:** Combinación matemática de columnas existentes para capturar mejor un fenómeno (ej. ratio deuda/ingreso).
*   **Transformaciones Numéricas:** Escalado (Standard Scaler para media 0 y varianza 1) y normalización de sesgos usando Logaritmo, Raíz Cuadrada o Box-Cox. También se utiliza la discretización (*Binning*) para agrupar variables continuas en intervalos.
*   **Codificación Categórica:** Conversión de texto a números. *One-Hot Encoding* para variables nominales (evita falsas jerarquías creando columnas binarias) y *Label Encoding* para variables ordinales.
*   **Variables Temporales:** Extracción de año, mes, día, hora o ciclos (fines de semana) para que el modelo capture estacionalidades.
*   **Vectorización de Texto (NLP):**
    *   *Bag of Words (BoW):* Conteo de frecuencias ignorando gramática.
    *   *TF-IDF:* Pondera relevancia penalizando palabras muy comunes en el corpus.
    *   *Embeddings:* Representación en vectores densos de baja dimensión que capturan la semántica y contexto relacional.

### 2.1 Selección de Características
Retener solo variables con valor predictivo significativo para evitar sobreajuste y reducir coste computacional.
*   **Heatmap de Correlación:** Detecta relaciones estadísticas y multicolinealidad.
*   **SelectKBest:** Filtrado estadístico riguroso (ej. ANOVA, Chi2).
*   **Importancia del Árbol (Random Forest Importance):** Ranking automático basado en impureza de Gini.
*   **Antipatrones:** Evitar redundancia (columnas que aportan lo mismo), variables sin relación causal directa (correlaciones espurias) y codificaciones incorrectas.

## 3. PARTICIÓN DE DATOS Y ESTRATEGIAS DE VALIDACIÓN
Es imperativo medir la capacidad de **generalización** del modelo para evitar el sobreajuste (memorización de datos).

*   **Estructura Train/Validation/Test:** 
    *   *Train (70-80%):* Enseña patrones al modelo.
    *   *Validation (10-15%):* Ajusta hiperparámetros y compara modelos.
    *   *Test (10-15%):* Simula datos futuros. El modelo **nunca** debe ver estos datos durante su desarrollo.
*   **Validación Cruzada (K-Fold):** El dataset se divide en *K* partes. Se entrena iterativamente usando diferentes combinaciones, promediando los resultados para reducir sesgos.
*   **Stratified K-Fold:** Obligatorio ante *datasets desbalanceados* (ej. 90% vs 10%). Garantiza que la proporción de las clases se mantenga idéntica en todos los pliegues de entrenamiento y validación.
*   **Data Leakage (Fuga de Datos):** Error crítico donde el modelo accede a información del futuro o a la variable objetivo durante el entrenamiento, logrando un rendimiento artificialmente alto que fracasará en producción.

## 4. ENTRENAMIENTO Y OPTIMIZACIÓN DE HIPERPARÁMETROS
A diferencia de los parámetros (pesos y sesgos que el modelo aprende por sí solo), los **hiperparámetros** son las configuraciones manuales que controlan cómo ocurre el aprendizaje.

*   **Hiperparámetros Base:**
    *   *Learning Rate (Tasa de Aprendizaje):* Tamaño del paso del optimizador. Muy alto genera saltos inestables; muy bajo atrapa al modelo en mínimos locales (típicamente 0.001-0.1 para Redes Neuronales, 0.01-0.3 para XGBoost).
    *   *Batch Size (Tamaño de Lote):* Muestras procesadas antes de actualizar pesos. Lotes pequeños (16-64) ayudan a la generalización aportando ruido; lotes grandes (256+) paralelizan rápido pero pueden sobreajustar. *Regla: A mayor batch size, se requiere un learning rate proporcionalmente más alto*.
    *   *Epochs (Épocas):* Iteraciones sobre todo el dataset. Evita el underfitting (muy pocas) o overfitting (demasiadas) combinándose con metodologías de *Early Stopping*.
    *   *Optimizador:* Algoritmo de minimización de error (Adam, SGD, RMSprop).
*   **Hiperparámetros de Arquitectura (Redes Neuronales):** Capas ocultas (profundidad), neuronas por capa (ancho), y funciones de activación (ReLU, GELU).
*   **Regularización:** Penaliza la complejidad para evitar el sobreajuste. Incluye L1/L2, *Dropout*, *Weight decay* para NN, y *max_depth* o *min_child_weight* para árboles como XGBoost.
*   **Estrategias de Búsqueda:**
    *   *Grid Search:* Fuerza bruta sobre una malla.
    *   *Random Search:* Muestreo aleatorio más eficiente.
    *   *Bayesian Optimization:* Modelo probabilístico avanzado que aprende de resultados previos para minimizar tiempos de búsqueda.

## 5. EVALUACIÓN Y AUDITORÍA
Una evaluación robusta dicta si el modelo está listo para producción y requiere diagnosticar el equilibrio a través de una Matriz de Confusión.
*   **Métricas principales:** *Accuracy* (aciertos globales), *Precision* (exactitud en la predicción positiva), *Recall* (capacidad de encontrar todos los positivos) y *F1-Score* (balance armónico). Solamente medir *Accuracy* es un error severo, en especial en datos desbalanceados.

## 6. ARQUITECTURA DE BIG DATA E INFRAESTRUCTURA
Si la naturaleza del problema escapa a las herramientas tradicionales, se debe considerar una infraestructura analítica de Big Data, definida por las "5 Vs": Volumen (Petabytes), Velocidad (Tiempo real), Variedad (80% datos no estructurados), Veracidad y Valor.

*   **Tecnologías requeridas:**
    *   Procesamiento Distribuido: Apache Hadoop o Apache Spark (este último maneja procesamiento en memoria hasta 100 veces más rápido).
    *   Almacenamiento: Data Lakes, bases de datos NoSQL (MongoDB, Cassandra) para esquemas flexibles.
    *   Streaming: Apache Kafka para ingesta de eventos de bajísima latencia.
    *   Infraestructura: Entornos nativos en Cloud Público que dominan el despliegue del mercado actual.