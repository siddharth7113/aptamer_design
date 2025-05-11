import numpy as np
from scipy.stats import spearmanr, pearsonr

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)

def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman Rank Correlation
    """
    corr, _ = spearmanr(y_true, y_pred)
    return corr

def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson Correlation Coefficient
    """
    corr, _ = pearsonr(y_true, y_pred)
    return corr

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns all supported metrics in a dictionary.
    """
    return {
        "mse": mse(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred)
    }
