import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from src.utils.logger import get_logger

logger = get_logger("model_evaluate")

def evaluate_regression(y_true, y_pred) -> dict:
    """Calculates regression performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE) handling zeros safely
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
    
    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE": float(mape)
    }
    logger.info(f"Regression Metrics: {metrics}")
    return metrics

def evaluate_classification(y_true, y_pred, labels=None) -> dict:
    """Calculates classification performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        "Accuracy": float(acc),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1),
        "Confusion_Matrix": cm.tolist()
    }
    logger.info(f"Classification Metrics: Accuracy={acc:.4f}, Macro_F1={macro_f1:.4f}")
    return metrics
