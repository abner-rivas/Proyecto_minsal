import numpy as np
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_probs):
    """
    Calcula el umbral óptimo de clasificación maximizando el F1-Score
    a partir de la curva de Precisión-Recall.
    
    Retorna:
        - umbral_optimo (float)
        - f1_maximo (float)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Cálculo del F1-Score para todos los umbrales posibles
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Identificar el índice del valor máximo
    indice_optimo = np.argmax(f1_scores)
    umbral_optimo = thresholds[indice_optimo]
    f1_maximo = f1_scores[indice_optimo]
    
    return umbral_optimo, f1_maximo