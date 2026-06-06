import numpy as np
import pytest

from src.evaluate import compute_metrics


# Fixtures

@pytest.fixture
def perfect_predictions():
    y = np.array([500.0, 600.0, 700.0, 800.0])
    return y, y.copy()

@pytest.fixture
def imperfect_predictions():
    y_true = np.array([500.0, 600.0, 700.0, 800.0])
    y_pred = np.array([520.0, 580.0, 710.0, 790.0])
    return y_true, y_pred


# compute_metrics

class TestComputeMetrics:

    def test_perfect_predictions_r2_one(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred, "Test")
        assert result["r2"] == pytest.approx(1.0)

    def test_perfect_predictions_zero_errors(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred, "Test")
        assert result["mae"]  == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mape"] == pytest.approx(0.0)

    def test_returns_all_required_keys(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        result = compute_metrics(y_true, y_pred, "Test")
        assert set(result.keys()) == {"model", "r2", "mae", "rmse", "mape", "accuracy"}

    def test_accuracy_is_100_minus_mape(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        result = compute_metrics(y_true, y_pred, "Test")
        assert result["accuracy"] == pytest.approx(100 - result["mape"])

    def test_model_name_stored(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        result = compute_metrics(y_true, y_pred, "XGBoost")
        assert result["model"] == "XGBoost"
