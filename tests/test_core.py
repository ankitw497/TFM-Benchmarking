"""
Basic test suite for TFM-Bench.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_synthetic_fallback(self):
        """Loader should generate synthetic data when downloads fail."""
        from src.data.loader import _generate_synthetic_credit
        df = _generate_synthetic_credit(n=1000, seed=42)
        assert len(df) == 1000
        assert "SeriousDlqin2yrs" in df.columns
        assert df["SeriousDlqin2yrs"].isin([0, 1]).all()

    def test_load_returns_correct_shapes(self):
        """Loader should return properly shaped splits."""
        from src.data.loader import load_credit_dataset
        X_train, X_test, y_train, y_test = load_credit_dataset(
            "give_me_credit", max_rows=500, random_state=42
        )
        assert len(X_train) + len(X_test) <= 500
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_stratified_split(self):
        """Positive rates should be similar in train and test."""
        from src.data.loader import load_credit_dataset
        X_train, X_test, y_train, y_test = load_credit_dataset(
            "give_me_credit", max_rows=2000, random_state=42
        )
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.05

    def test_val_split(self):
        """Return validation set when requested."""
        from src.data.loader import load_credit_dataset
        result = load_credit_dataset(
            "give_me_credit", max_rows=500, return_val=True, random_state=42
        )
        assert len(result) == 6  # X_train, X_val, X_test, y_train, y_val, y_test

    def test_dataset_info(self):
        """Dataset info should return valid metadata."""
        from src.data.loader import get_dataset_info
        info = get_dataset_info("give_me_credit")
        assert info["n_samples"] == 150_000
        assert info["has_missing"] is True

    def test_unknown_dataset_raises(self):
        from src.data.loader import load_credit_dataset
        with pytest.raises(ValueError):
            load_credit_dataset("nonexistent_dataset")


# ---------------------------------------------------------------------------
# Base model tests
# ---------------------------------------------------------------------------

class TestBaseModel:
    def test_benchmark_result_to_dict(self):
        from src.models.base import BenchmarkResult
        r = BenchmarkResult(model_name="test", dataset_name="test", phase="zero_shot")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "model_name" in d
        assert "auc_roc" in d

    def test_ece_computation(self):
        from src.models.base import _compute_ece
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        ece = _compute_ece(y_true, y_prob, n_bins=5)
        assert 0 <= ece <= 1

    def test_can_handle_dataset(self):
        """Model should correctly check dataset limits."""
        from src.models.base import BaseModelWrapper, ModelLimitations

        class DummyModel(BaseModelWrapper):
            def fit(self, X, y): pass
            def predict_proba(self, X): return np.zeros((len(X), 2))
            def get_limitations(self):
                return ModelLimitations(max_rows=1000, max_features=50)

        m = DummyModel("dummy")
        assert m.can_handle_dataset(500, 30) is True
        assert m.can_handle_dataset(1500, 30) is False
        assert m.can_handle_dataset(500, 60) is False


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_compute_all_metrics(self):
        from src.evaluation.metrics import compute_all_metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])
        metrics = compute_all_metrics(y_true, y_prob)
        assert "auc_roc" in metrics
        assert "log_loss" in metrics
        assert "ece" in metrics
        assert 0 <= metrics["auc_roc"] <= 1
        assert metrics["log_loss"] > 0

    def test_ece_perfect_calibration(self):
        from src.evaluation.metrics import compute_ece
        # Perfectly calibrated: predicted prob matches actual frequency
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece < 0.1  # Should be very low

    def test_aggregate_cv_results(self):
        from src.evaluation.metrics import aggregate_cv_results
        folds = [
            {"auc_roc": 0.85, "log_loss": 0.4},
            {"auc_roc": 0.87, "log_loss": 0.38},
            {"auc_roc": 0.86, "log_loss": 0.39},
        ]
        agg = aggregate_cv_results(folds)
        assert "auc_roc" in agg
        assert abs(agg["auc_roc"]["mean"] - 0.86) < 0.01

    def test_compute_ranks(self):
        from src.evaluation.metrics import compute_ranks
        scores = np.array([
            [0.9, 0.8, 0.7],  # Dataset 1
            [0.8, 0.9, 0.7],  # Dataset 2
        ])
        ranks = compute_ranks(scores, higher_is_better=True)
        assert ranks.shape == (3,)
        assert ranks[2] == 3.0  # Third model always last


# ---------------------------------------------------------------------------
# GBDT wrapper tests (always available)
# ---------------------------------------------------------------------------

class TestGBDTWrappers:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((rng.randn(200) > 0).astype(int))
        return X[:160], X[160:], y[:160], y[160:]

    def test_xgboost_wrapper(self, sample_data):
        pytest.importorskip("xgboost", reason="XGBoost not installed")
        from src.models.gbdt_wrapper import XGBoostModel
        m = XGBoostModel(tuned=False)
        result = m.evaluate(*sample_data, "test_data", "test")
        assert result.success
        assert 0 <= result.auc_roc <= 1

    def test_lightgbm_wrapper(self, sample_data):
        pytest.importorskip("lightgbm", reason="LightGBM not installed")
        from src.models.gbdt_wrapper import LightGBMModel
        m = LightGBMModel()
        result = m.evaluate(*sample_data, "test_data", "test")
        assert result.success
