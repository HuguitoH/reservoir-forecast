import numpy as np
import pandas as pd
import pytest

from src.features import build_xgb_features


# Fixtures

@pytest.fixture
def monthly_series() -> pd.Series:
    """Synthetic monthly time series with DatetimeIndex."""
    idx = pd.date_range("2000-01-01", periods=48, freq="MS")
    np.random.seed(42)
    return pd.Series(np.random.uniform(400, 900, 48), index=idx, name="total_hm3")


# build_xgb_features

class TestBuildXgbFeatures:

    def test_all_expected_columns_created(self, monthly_series):
        out = build_xgb_features(monthly_series, n_lags=3)
        expected = [
            "lag_1", "lag_2", "lag_3",
            "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
            "rolling_std_3", "rolling_std_12",
            "month", "month_sin", "month_cos",
        ]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_nulls_in_output(self, monthly_series):
        out = build_xgb_features(monthly_series, n_lags=12)
        assert out.isna().sum().sum() == 0

    def test_lag_1_correct(self, monthly_series):
        out = build_xgb_features(monthly_series, n_lags=3)
        for i in range(5):
            idx = out.index[i]
            prev_idx = monthly_series.index[monthly_series.index.get_loc(idx) - 1]
            assert out.loc[idx, "lag_1"] == pytest.approx(monthly_series[prev_idx])

    def test_month_sin_cos_range(self, monthly_series):
        out = build_xgb_features(monthly_series, n_lags=3)
        assert out["month_sin"].between(-1, 1).all()
        assert out["month_cos"].between(-1, 1).all()

    def test_month_values_correct(self, monthly_series):
        out = build_xgb_features(monthly_series, n_lags=3)
        assert out["month"].between(1, 12).all()

    def test_does_not_mutate_input(self, monthly_series):
        original_len = len(monthly_series)
        build_xgb_features(monthly_series, n_lags=3)
        assert len(monthly_series) == original_len

    def test_n_lags_respected(self, monthly_series):
        out = build_xgb_features(monthly_series, n_lags=5)
        assert "lag_5" in out.columns
        assert "lag_6" not in out.columns
