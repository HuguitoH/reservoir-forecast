"""
Model evaluation metrics for time series forecasting.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
) -> dict:
    """
    Compute regression metrics for time series forecast evaluation.

    Returns:
        model    — model name
        r2       — R squared
        mae      — Mean Absolute Error (hm³)
        rmse     — Root Mean Squared Error (hm³)
        mape     — Mean Absolute Percentage Error (%)
        accuracy — 100 - MAPE
    """
    mask = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )

    metrics = {
        "model":    name,
        "r2":       round(float(r2_score(y_true, y_pred)), 4),
        "mae":      round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse":     round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mape":     round(mape, 2),
        "accuracy": round(100 - mape, 2),
    }

    print(
        f"{name:<14} R²={metrics['r2']:.4f}  MAE={metrics['mae']:>7.1f}  "
        f"RMSE={metrics['rmse']:>7.1f}  MAPE={metrics['mape']:.2f}%  "
        f"Accuracy={metrics['accuracy']:.2f}%"
    )

    return metrics
