import numpy as np
import pandas as pd
import pytest

from src.forecast import classify_capacity, iterative_forecast


# Fixtures

@pytest.fixture
def monthly_series() -> pd.Series:
    idx = pd.date_range("2000-01-01", periods=36, freq="MS")
    np.random.seed(42)
    return pd.Series(np.random.uniform(500, 800, 36), index=idx)

@pytest.fixture
def dummy_model():
    """Simple model that always predicts the mean of inputs."""
    class MeanModel:
        def predict(self, X):
            return np.array([650.0])
    return MeanModel()


# classify_capacity

class TestClassifyCapacity:

    def test_severe_below_threshold(self):
        label, cls = classify_capacity(400.0, 488.0, 574.0)
        assert cls == "severe"
        assert "Severe" in label

    def test_moderate_between_thresholds(self):
        label, cls = classify_capacity(530.0, 488.0, 574.0)
        assert cls == "moderate"
        assert "Moderate" in label

    def test_normal_above_moderate_threshold(self):
        label, cls = classify_capacity(700.0, 488.0, 574.0)
        assert cls == "normal"
        assert "Normal" in label

    def test_boundary_severe(self):
        label, cls = classify_capacity(488.0, 488.0, 574.0)
        assert cls == "moderate"

    def test_boundary_moderate(self):
        label, cls = classify_capacity(574.0, 488.0, 574.0)
        assert cls == "normal"


# iterative_forecast

class TestIterativeForecast:

    def test_returns_correct_length(self, monthly_series, dummy_model):
        dates, preds = iterative_forecast(
            series=monthly_series,
            model=dummy_model,
            feature_cols=["lag_1", "rolling_mean_3", "rolling_mean_6",
                            "rolling_mean_12", "rolling_std_3", "rolling_std_12",
                            "month", "month_sin", "month_cos",
                            "lag_2", "lag_3", "lag_4", "lag_5",
                            "lag_6", "lag_7", "lag_8", "lag_9",
                            "lag_10", "lag_11", "lag_12"],
            n_lags=12,
            n_months=6,
        )
        assert len(dates) == 6
        assert len(preds) == 6

    def test_dates_start_after_series(self, monthly_series, dummy_model):
        dates, _ = iterative_forecast(
            series=monthly_series,
            model=dummy_model,
            feature_cols=["lag_1", "rolling_mean_3", "rolling_mean_6",
                            "rolling_mean_12", "rolling_std_3", "rolling_std_12",
                            "month", "month_sin", "month_cos",
                            "lag_2", "lag_3", "lag_4", "lag_5",
                            "lag_6", "lag_7", "lag_8", "lag_9",
                            "lag_10", "lag_11", "lag_12"],
            n_lags=12,
            n_months=3,
        )
        assert dates[0] > monthly_series.index[-1]

    def test_dates_are_monthly(self, monthly_series, dummy_model):
        dates, _ = iterative_forecast(
            series=monthly_series,
            model=dummy_model,
            feature_cols=["lag_1", "rolling_mean_3", "rolling_mean_6",
                            "rolling_mean_12", "rolling_std_3", "rolling_std_12",
                            "month", "month_sin", "month_cos",
                            "lag_2", "lag_3", "lag_4", "lag_5",
                            "lag_6", "lag_7", "lag_8", "lag_9",
                            "lag_10", "lag_11", "lag_12"],
            n_lags=12,
            n_months=4,
        )
        for i in range(1, len(dates)):
            diff = (dates[i].year - dates[i-1].year) * 12 + \
                    (dates[i].month - dates[i-1].month)
            assert diff == 1
