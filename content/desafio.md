
# Desafío de Machine Learning: Predicción de Riesgos en Adolescentes Salvadoreños (GSHS 2013)

## Objetivo General
Construir un sistema de Machine Learning basado en los datos de la encuesta **GSHS El Salvador 2013 (OMS/OPS)** para identificar patrones de riesgo en adolescentes salvadoreños.El proyecto debe implementar un pipeline completo de ciencia de datos que incluya:
- Limpieza de datos.
- Análisis exploratorio (EDA).
- Ingeniería de características.
- Modelado.
- Optimización.
- Interpretación de resultados.
- Elaboración de informe técnico.

## DatasetArchivo principal:
textSLV2013_Public_Use.csv


### Variables relevantes

- `Q1 - Q58`: respuestas originales de la encuesta.
- `QN6 - QN58`: versiones numéricas recodificadas.
- Variables derivadas (`qnowtg`, `qnpa7g`, etc.): indicadores ya calculados.

### Valor nulo especial

El valor:

```
1.79769313486232e+308
```

representa datos faltantes (`missing values`) y debe ser reemplazado por:

```
np.nan
```

antes de cualquier análisis.

---

# Tarea A: Regresión (Predicción de IMC)

## Construcción del target

Calcular:

```
IMC = Peso(Q5) / (Estatura(Q4) ** 2)
```

## Objetivo

Predecir el IMC utilizando únicamente variables de comportamiento, por ejemplo:

- Alimentación.
- Actividad física.
- Tiempo frente a pantallas.
- Hábitos de salud.

### Restricción

No utilizar directamente:

- Peso (`Q5`)
- Estatura (`Q4`)

como variables predictoras.

## Métricas

### Principal

- RMSE (minimizar)

### Secundaria

- R² (maximizar)

---

# Tarea B: Clasificación (Riesgo en Salud Mental)

## Construcción del target

Identificar una variable relacionada con riesgo grave de salud mental, por ejemplo:

- Ideación suicida.
- Sentimientos persistentes de tristeza.
- Soledad.

Crear:

```
Riesgo_Salud_Mental
```

donde:

```
1 = Riesgo0 = No riesgo
```

## Objetivo

Predecir el riesgo utilizando factores asociados como:

- Bullying.
- Consumo de alcohol.
- Relación con padres o tutores.
- Factores de protección.
- Conductas de riesgo.

## Métricas

### Principal

- F1-Score de la clase minoritaria

### Secundaria

- AUC-ROC

---

# Requisitos Técnicos

## 1. Limpieza de Datos

### Obligatorio

- Cargar dataset.
- Reemplazar `1.79769313486232e+308` por `np.nan`.
- Detectar y gestionar valores faltantes.

### Imputación

Usar estrategias justificadas:

- Mediana.
- Moda.

---

## 2. Análisis Exploratorio (EDA)

Realizar análisis:

### Univariado

Ejemplos:

- Distribución de edad.
- Distribución de IMC.
- Distribución de la variable de riesgo.

### Bivariado

Analizar relaciones entre:

- Variables predictoras.
- Targets.

### Visualizaciones recomendadas

- Histogramas.
- Boxplots.
- Countplots.
- Matrices de correlación.

---

## 3. Ingeniería de Características

### Decisión Q vs QN

Elegir entre:

- Variables originales (`Q`)
- Variables recodificadas (`QN`)

Justificar técnicamente la elección para evitar:

- Redundancia.
- Colinealidad.
- Data leakage.

### Transformaciones

- Codificación categórica:
- One-Hot Encoding.
- Target Encoding.
- Escalado:
- StandardScaler.
- RobustScaler.

---

## 4. Modelado

### Regresión

Entrenar al menos 2 modelos:

- Linear Regression
- Random Forest Regressor

Opcional:

- XGBoost Regressor
- Gradient Boosting
- Extra Trees

### Clasificación

Entrenar al menos 2 modelos:

- Logistic Regression
- Random Forest Classifier

Opcional:

- XGBoost
- LightGBM

### Manejo del desbalance

Usar al menos una estrategia:

```
class_weight="balanced"
```

o

```pythonrunsmote
SMOTE
```

---

## 5. Validación

Utilizar:

```
Cross-Validation
```

para todos los modelos.

Recomendado:

```
KFoldStratifiedKFold
```

según la tarea.

---

## 6. Optimización

Aplicar búsqueda de hiperparámetros mediante:

```
GridSearchCV
```

o

```
RandomizedSearchCV
```

---

## 7. Evaluación

### Regresión

Reportar:

- RMSE
- MAE (opcional)
- R²

Analizar:

- Residuos.
- Sobreajuste.

### Clasificación

Reportar:

- F1-Score
- Precision
- Recall
- AUC-ROC

Generar:

- Matriz de confusión.

---

## 8. Interpretabilidad

Identificar variables más influyentes mediante:

- Feature Importance.
- Coeficientes del modelo.
- SHAP (opcional).

Responder:

- ¿Qué factores predicen mejor el IMC?
- ¿Qué factores predicen mejor el riesgo en salud mental?

---

# Entregable Final

## Documento técnico (formato IEEE)

Dirigido a un hipotético Ministro de Salud.

Debe incluir:

1. Problema abordado.
2. Descripción del dataset.
3. Metodología.
4. Resultados de regresión.
5. Resultados de clasificación.
6. Variables más importantes.
7. Recomendaciones de política pública.
8. Conclusiones.

---

# Criterios de Éxito

## Limpieza y EDA (20%)

- Manejo correcto de valores nulos.
- Visualizaciones útiles.
- Identificación de patrones relevantes.

## Feature Engineering (15%)

- Cálculo correcto del IMC.
- Ausencia de data leakage.
- Manejo adecuado de Q y QN.

## Modelo de Regresión (20%)

- Múltiples modelos.
- Cross-validation.
- Buen desempeño y análisis.

## Modelo de Clasificación (25%)

- Tratamiento del desbalance.
- Optimización de hiperparámetros.
- Interpretación adecuada de F1 y AUC.

## Interpretación y Reporte (20%)

- Explicación clara de resultados.
- Aplicabilidad para salud pública en El Salvador.
- Código modular y bien documentado.