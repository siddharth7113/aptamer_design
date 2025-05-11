import numpy as np
from aptamer_design.evaluation import compute_metrics

def test_metrics_output_shape():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    metrics = compute_metrics(y_true, y_pred)
    assert "mse" in metrics
    assert "spearman" in metrics
    assert "pearson" in metrics
    assert all(isinstance(v, float) for v in metrics.values())
