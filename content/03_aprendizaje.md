
# BASE DE CONOCIMIENTO: EXPERTO EN MACHINE LEARNING TRADICIONAL
**Propósito:** Este documento proporciona el contexto técnico normativo para el análisis, diseño y evaluación de pipelines de Machine Learning. El agente debe utilizar estos principios para formular recomendaciones frente a cualquier problemática de datos.

## 1. PREPROCESAMIENTO Y ARQUITECTURA DEL PIPELINE
Antes de modelar, los datos deben someterse a tratamientos estrictos para garantizar la validez matemática y evitar fugas de información.

*   **Prevención de "Data Leakage" (Fuga de Datos):** Es un error crítico aplicar transformaciones (como escalado completo) antes de realizar validación cruzada. Se deben utilizar **Pipelines** (ej. Scikit-Learn) que encadenen transformadores y estimadores, asegurando que el método `fit()` se llame dinámicamente solo en los *folds* de entrenamiento.
*   **Normalización y Escalado:** Obligatorio para algoritmos basados en distancias (K-Means, DBSCAN, KNN, SVM).
    *   *StandardScaler:* Asume distribución gaussiana, vital para Kernel RBF.
    *   *MinMaxScaler:* Ideal para límites conocidos (ej. imágenes 0-255) o para mantener ceros (datos dispersos).
    *   *RobustScaler:* Utiliza el rango intercuartil, diseñado para datasets con muchos valores atípicos (outliers).
*   **Maldición de la Dimensionalidad:** En espacios de alta dimensión, las distancias euclidianas convergen y pierden significado estadístico, lo que encarece el cómputo y arruina modelos como DBSCAN o K-Means. Se requiere Reducción de Dimensionalidad (ej. PCA, t-SNE) antes del modelado.

## 2. APRENDIZAJE SUPERVISADO Y REGULARIZACIÓN
La elección del modelo depende de la topología de los datos y la necesidad de interpretabilidad versus precisión.

### 2.1 Modelos Basados en Distancia y Planos
*   **KNN (K-Nearest Neighbors):** Algoritmo no paramétrico que captura relaciones no lineales basado en cercanía. Requiere calibrar $K$ (vecinos) y considerar el uso de pesos (`weights='distance'`) para mitigar la influencia excesiva de outliers lejanos.
*   **SVM (Support Vector Machines):** Busca el hiperplano con el **máximo margen** de separación entre clases, apoyándose en los "Vectores de Soporte".
    *   **Margen Suave y parámetro C:** Introducir holgura permite violaciones controladas. Un $C$ bajo crea fronteras suaves y prioriza un margen amplio (riesgo de underfitting), mientras que un $C$ alto se ajusta rígidamente a los datos de entrenamiento (riesgo de overfitting).
    *   **Kernel Trick:** Permite mapear datos a dimensiones superiores para separabilidad lineal sin un costo computacional masivo.
    *   **Kernel RBF (Gaussiano):** Proyecta a dimensión infinita. El parámetro **Gamma ($\gamma$)** controla la influencia; un gamma alto hace que la frontera se encoja alrededor de puntos individuales (overfitting).

### 2.2 Modelos Basados en Árboles
*   **Árboles de Decisión:** Interpretabilidad alta pero riesgo crítico de sobreajuste al "memorizar" datos. Se controla podando hiperparámetros: `max_depth`, `min_samples_split`, y `min_samples_leaf`.
*   **Random Forest:** Ensamble que utiliza *Bagging* (muestreo Bootstrap) y selección aleatoria de características en cada nodo para crear diversidad. Reduce drásticamente la varianza y suele ganar en precisión tabular, además de ofrecer puntuaciones de importancia de variables.

### 2.3 Trade-off Sesgo-Varianza y Regularización (L1/L2)
Introducir sesgo deliberadamente reduce la varianza y mejora la generalización en datos nuevos, crucial en campos sensibles como la medicina.
*   **Lasso (L1):** Fuerza algunos coeficientes exactamente a cero. Excelente para selección automática de características en datasets de alta dimensión, creando modelos interpretables y "parsimoniosos".
*   **Ridge (L2):** Reduce la magnitud de todos los coeficientes hacia cero sin eliminarlos. Ideal cuando existe alta multicolinealidad entre variables.
*   **Elastic Net:** Híbrido que maneja colinealidad agrupada y a la vez realiza selección de variables.

## 3. APRENDIZAJE NO SUPERVISADO (CLUSTERING)
Utilizado para descubrir estructuras intrínsecas sin usar etiquetas predefinidas. La métrica de similitud (Euclidiana L2 o Manhattan L1) dictamina el éxito.

### 3.1 Modelos Geométricos y por Densidad
*   **K-Means:** Particiona mediante centroides buscando reducir la varianza interna. Asume formas esféricas, es altamente sensible a ruido y requiere definir $K$ a priori. Obligatoria la inicialización `k-means++` para evitar mínimos locales.
*   **DBSCAN:** Basado en densidad, une áreas continuas de alta concentración de puntos núcleo. Detecta formas geométricas arbitrarias, aísla el ruido (outliers) y no requiere $K$. Su limitación principal es fallar ante clústeres de densidades variables. Parámetros: $\epsilon$ (Epsilon, distancia radial) y $MinPts$ (umbral mínimo).

### 3.2 Modelos Topológicos y Jerárquicos
*   **Clustering Jerárquico:** Aglomerativo (Bottom-Up) o Divisivo, construye una estructura jerárquica visible mediante **Dendrogramas**. Permite seleccionar el número de clústeres a posteriori midiendo distancias de enlace (*Single, Complete, Average, Ward*). Alto costo computacional.
*   **Self-Organizing Maps (SOM):** Basado en redes neuronales, reduce dimensionalidad preservando la topología de los datos multidimensionales, organizando nodos en mapas de aprendizaje competitivo.

## 4. MÉTRICAS DE EVALUACIÓN
Nunca se debe confiar en una sola métrica. El contexto de evaluación dicta la idoneidad del modelo.
*   **Para Regresión:**
    *   **RMSE:** Mide la magnitud de fallo en las *unidades originales* del problema, penalizando desviaciones grandes.
    *   **Coeficiente R²:** Mide la bondad de ajuste general (porcentaje de variabilidad de los datos explicada por el modelo, entre 0 y 1).
*   **Para Clustering:**
    *   **Método del Codo (Elbow):** Evalúa la inercia para encontrar el punto de inflexión de rendimiento al sumar clústeres.
    *   **Análisis de Silueta (-1 a +1):** Evalúa cohesión y separación. Un valor cercano a +1 indica correcta clasificación; negativo indica solapamiento y asignación incorrecta.