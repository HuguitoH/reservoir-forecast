"""
Feature engineering functions for time series modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_xgb_features(series: pd.Series, n_lags: int = 12) -> pd.DataFrame:
    """
    Build supervised features from a time series for XGBoost.

    Features:
        lag_1 ... lag_n    — lagged values
        rolling_mean_3/6/12 — rolling means (shift=1 to avoid leakage)
        rolling_std_3/12   — rolling standard deviations
        month              — month number (1-12)
        month_sin/cos      — cyclic month encoding
    """
    df = pd.DataFrame({"y": series})

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["rolling_mean_3"]  = df["y"].shift(1).rolling(3).mean()
    df["rolling_mean_6"]  = df["y"].shift(1).rolling(6).mean()
    df["rolling_mean_12"] = df["y"].shift(1).rolling(12).mean()
    df["rolling_std_3"]   = df["y"].shift(1).rolling(3).std()
    df["rolling_std_12"]  = df["y"].shift(1).rolling(12).std()

    df["month"]     = series.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df.dropna()
