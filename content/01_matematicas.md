
***

# Contexto Base: Especialización en Machine Learning
**Propósito de este documento:** Este documento contiene los fundamentos teóricos, matemáticos y metodológicos que un Agente de IA debe utilizar como marco de referencia para analizar, diagnosticar y proponer soluciones a cualquier problemática de Machine Learning (ML).

---

## 1. Representación Matemática de Datos (Álgebra Lineal)
Para el agente, el mundo se traduce a estructuras algebraicas. Los algoritmos no procesan tablas ni textos, sino **vectores y matrices**. 
*   **Matriz de Datos ($X$):** El dataset completo se transforma en una matriz donde **cada fila representa un ejemplo (observación) y cada columna representa una característica (feature)**.
*   **Vectores ($w, x, y$):** Las características individuales y los pesos del modelo son vectores.
*   **Operación Fundamental:** La predicción base surge del producto matriz-vector: $y = Xw$ (donde $y$ es la predicción, $X$ los datos y $w$ los pesos).

## 2. Análisis Exploratorio de Datos (EDA) y Estadística Descriptiva
Antes de modelar, los datos deben ser comprendidos y limpiados usando estadística descriptiva.
*   **Distribución y Tendencia Central:** Se utilizan medidas como media, mediana, moda y desviación estándar para entender el comportamiento base de los datos.
*   **Tratamiento de Outliers (Valores Atípicos):** Si un outlier es un **error de medición**, debe eliminarse o imputarse. Si es un **evento real** (ej. fraude o clima extremo), debe mantenerse, pues contiene información predictiva valiosa para el modelo.

## 3. Inferencia, Probabilidad y Selección de Variables
No todas las variables de la matriz $X$ son útiles. El agente debe usar pruebas estadísticas para selección de *features* y validación probabilística.
*   **Teorema de Bayes:** Piedra angular para calcular el riesgo real y actualizar probabilidades ante nueva evidencia (ideal para NLP, Naive Bayes y diagnóstico).
*   **Prueba T de Student (t-test):** Se usa para comparar si existe una diferencia estadísticamente significativa entre las medias de **dos grupos**.
*   **ANOVA:** Se aplica para comparar eficientemente las medias de **tres o más grupos** simultáneamente.
*   **Chi-cuadrado:** Especializado en evaluar la independencia estadística entre **variables categóricas**.

## 4. Definición del Error: Funciones de Costo y Pérdida
El modelo aprende únicamente porque puede medir sus equivocaciones. El agente debe seleccionar la función correcta según el problema empírico.
*   **Diferencia conceptual:** La **función de pérdida (Loss)** mide el error en una sola observación, mientras que la **función de costo (Cost)** promedia y resume el error global de todo el conjunto de entrenamiento.
*   **Problemas de Regresión:**
    *   **MSE (Mean Squared Error):** Penaliza fuertemente los errores grandes al elevarlos al cuadrado. Estándar, pero sensible a outliers.
    *   **MAE (Mean Absolute Error):** Robusto ante valores atípicos, penaliza de forma lineal.
    *   **RMSE:** Mide el error en las mismas unidades interpretables que la variable objetivo.
*   **Problemas de Clasificación Probabilística:**
    *   **Binary Cross Entropy (Log Loss):** Para problemas con 2 clases (ej. Spam o No Spam).
    *   **Categorical Cross Entropy:** Para problemas con más de 2 clases.
*   **Máquinas de Soporte Vectorial (SVM):** Utilizan la **Hinge Loss** para maximizar el margen geométrico.

## 5. Optimización de Parámetros
El entrenamiento es un problema de optimización cuyo objetivo es navegar la superficie de la función de costo hasta encontrar el mínimo global o local ajustando los pesos.
*   **Algoritmos de Descenso:**
    *   **Gradient Descent (Batch):** Usa todo el dataset. Preciso pero lento.
    *   **Stochastic Gradient Descent (SGD):** Actualiza con una muestra a la vez. Muy rápido, pero con trayectoria ruidosa.
    *   **Mini-batch GD:** Divide los datos en pequeños lotes. Es el estándar de la industria que equilibra estabilidad y velocidad computacional (aprovechando GPUs).
    *   **Momentum:** Añade "inercia" ($\beta \approx 0.9$) a los gradientes pasados, amortiguando oscilaciones y acelerando la convergencia.
    *   **Adam:** Optimizador adaptativo que combina Momentum y RMSProp. Es la opción recomendada y estándar absoluto en Redes Neuronales y problemas complejos.
*   **Learning Rate ($\alpha$):** Parámetro crítico que define el tamaño del paso. Si es muy alto el modelo diverge; si es muy bajo, el aprendizaje se estanca o es demasiado lento.

## 6. Evaluación del Modelo y Dilema Bias-Variance
Al auditar los resultados de entrenamiento y validación (Curvas de Aprendizaje), el agente debe diagnosticar fallas de generalización.
*   **Underfitting (Alto Bias / Sesgo):**
    *   *Síntoma:* El modelo es demasiado simple y no aprende (error alto en entrenamiento y prueba).
    *   *Solución:* Aumentar complejidad (modelos no lineales), hacer *Feature Engineering*, reducir regularización o incrementar iteraciones.
*   **Overfitting (Alta Variance / Varianza):**
    *   *Síntoma:* El modelo memoriza el ruido del set de entrenamiento (error de entrenamiento casi cero), pero fracasa ante datos nuevos.
    *   *Solución:* Obtener más datos, simplificar el modelo, hacer poda de árboles (*pruning*), aplicar validación cruzada, o usar **regularización (Ridge/L2 o Lasso/L1)**.
*   **Objetivo (Trade-off):** Encontrar el *Sweet Spot* o punto de equilibrio donde la complejidad del modelo minimiza el Error Total.

## 7. Despliegue en Producción y Ética
El agente debe asegurar que el modelo no solo sea preciso matemáticamente, sino también justo y equitativo socialmente.
*   **Diferenciación de Sesgos:** El **sesgo (bias) estadístico** es un modelo matemático deficiente (underfitting), mientras que el **sesgo algorítmico/ético** ocurre cuando se perpetúa la discriminación humana (ej. por datos históricos injustos).
*   **Acciones Éticas:** Eliminar "variables encubiertas" y sensibles (como género, raza o códigos postales que deriven en marginación/redlining), revisar que los grupos subrepresentados estén compensados y auditar periódicamente el sistema.