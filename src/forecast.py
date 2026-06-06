"""
Iterative forecast pipeline using the trained XGBoost model.
Used by the Streamlit app — extracted from app.py to keep it testeable
and reusable across pages.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.features import build_xgb_features


def iterative_forecast(
    series: pd.Series,
    model,
    feature_cols: list[str],
    n_lags: int,
    n_months: int,
) -> tuple[pd.DatetimeIndex, list[float]]:
    """
    Generate an iterative forecast by appending each prediction
    to the series before computing features for the next step.

    Each predicted value becomes an input for the next prediction —
    error accumulates with horizon. Suitable for short to medium
    horizons (up to ~24 months with confidence).

    Args:
        series       — historical time series (monthly, DatetimeIndex)
        model        — fitted XGBoost model with .predict() method
        feature_cols — list of feature column names the model expects
        n_lags       — number of lag features used during training
        n_months     — forecast horizon in months

    Returns:
        dates  — DatetimeIndex of forecast dates
        preds  — list of predicted values (hm³)
    """
    last_date  = series.index[-1]
    series_ext = series.copy()
    preds: list[float] = []
    dates: list[pd.Timestamp] = []

    for i in range(n_months):
        df_feat  = build_xgb_features(series_ext, n_lags=n_lags)
        last_row = df_feat.iloc[[-1]][feature_cols]
        pred     = float(model.predict(last_row)[0])
        next_dt  = last_date + pd.DateOffset(months=i + 1)

        preds.append(pred)
        dates.append(next_dt)

        series_ext = pd.concat([
            series_ext,
            pd.Series([pred], index=[next_dt]),
        ])

    return pd.DatetimeIndex(dates), preds


def classify_capacity(
    value: float,
    severe_threshold: float,
    moderate_threshold: float,
) -> tuple[str, str]:
    """
    Classify a capacity value into a drought zone.

    Returns:
        label — human-readable classification
        cls   — CSS class name (severe / moderate / normal)
    """
    if value < severe_threshold:
        return "Severe drought", "severe"
    elif value < moderate_threshold:
        return "Moderate drought", "moderate"
    else:
        return "Normal", "normal"
