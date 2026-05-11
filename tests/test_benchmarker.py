"""
Integration tests for the public tfm_benchmark API.

These tests use only always-available models (sklearn) and bundled datasets
so they run without any optional dependencies.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

SKLEARN_MODELS = ["random_forest", "logistic_regression"]


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDatasetSynthetic:
    def test_synthetic_returns_four_tuple(self):
        from tfm_benchmark import load_dataset
        result = load_dataset("synthetic")
        assert len(result) == 4

    def test_synthetic_shapes(self):
        from tfm_benchmark import load_dataset
        X_train, X_test, y_train, y_test = load_dataset("synthetic")
        assert X_train.shape[1] == X_test.shape[1]
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_synthetic_nonempty(self):
        from tfm_benchmark import load_dataset
        X_train, X_test, y_train, y_test = load_dataset("synthetic")
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_synthetic_binary_labels(self):
        from tfm_benchmark import load_dataset
        _, _, y_train, y_test = load_dataset("synthetic")
        assert set(y_train.unique()) <= {0, 1}
        assert set(y_test.unique()) <= {0, 1}


class TestLoadDatasetFromDataFrame:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(7)
        return pd.DataFrame({
            "feat_a": rng.randn(100),
            "feat_b": rng.randn(100),
            "label":  rng.randint(0, 2, 100),
        })

    def test_dataframe_auto_splits(self, sample_df):
        from tfm_benchmark import load_dataset
        X_train, X_test, y_train, y_test = load_dataset(sample_df, target="label")
        assert len(X_train) + len(X_test) == 100

    def test_dataframe_target_not_in_features(self, sample_df):
        from tfm_benchmark import load_dataset
        X_train, X_test, _, _ = load_dataset(sample_df, target="label")
        assert "label" not in X_train.columns
        assert "label" not in X_test.columns

    def test_dataframe_correct_default_split(self, sample_df):
        from tfm_benchmark import load_dataset
        X_train, X_test, _, _ = load_dataset(sample_df, target="label")
        assert len(X_test) == pytest.approx(20, abs=2)


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_returns_list(self):
        from tfm_benchmark import list_datasets
        assert isinstance(list_datasets(), list)

    def test_includes_german_credit(self):
        from tfm_benchmark import list_datasets
        assert "german_credit" in list_datasets()

    def test_includes_synthetic(self):
        from tfm_benchmark import list_datasets
        assert "synthetic" in list_datasets()


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_nonempty_list(self):
        from tfm_benchmark import list_models
        result = list_models()
        assert isinstance(result, list) and len(result) > 0

    def test_sklearn_always_present(self):
        from tfm_benchmark import list_models
        result = list_models()
        assert "random_forest" in result
        assert "logistic_regression" in result


# ---------------------------------------------------------------------------
# Benchmarker.fit_evaluate
# ---------------------------------------------------------------------------

class TestBenchmarkerIntegration:
    @pytest.fixture
    def splits(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=150, n_features=6, random_state=3)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        y_s = pd.Series(y, name="target")
        return X_df[:120], y_s[:120], X_df[120:], y_s[120:]

    def test_returns_dataframe(self, splits):
        from tfm_benchmark import Benchmarker
        results = Benchmarker(models=SKLEARN_MODELS, verbose=False).fit_evaluate(*splits)
        assert isinstance(results, pd.DataFrame)

    def test_expected_columns(self, splits):
        from tfm_benchmark import Benchmarker
        results = Benchmarker(models=SKLEARN_MODELS, verbose=False).fit_evaluate(*splits)
        for col in ["model_name", "auc_roc", "success"]:
            assert col in results.columns

    def test_one_row_per_model(self, splits):
        from tfm_benchmark import Benchmarker
        results = Benchmarker(models=SKLEARN_MODELS, verbose=False).fit_evaluate(*splits)
        assert len(results) == len(SKLEARN_MODELS)

    def test_sklearn_models_succeed(self, splits):
        from tfm_benchmark import Benchmarker
        results = Benchmarker(models=SKLEARN_MODELS, verbose=False).fit_evaluate(*splits)
        assert results["success"].all()


# ---------------------------------------------------------------------------
# run_benchmark (same structure as Benchmarker)
# ---------------------------------------------------------------------------

class TestRunBenchmarkIntegration:
    @pytest.fixture
    def splits(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=150, n_features=6, random_state=5)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        y_s = pd.Series(y, name="target")
        return X_df[:120], y_s[:120], X_df[120:], y_s[120:]

    def test_returns_dataframe(self, splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*splits, models=["random_forest"], verbose=False)
        assert isinstance(results, pd.DataFrame)

    def test_expected_columns(self, splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*splits, models=["random_forest"], verbose=False)
        for col in ["model_name", "auc_roc", "success"]:
            assert col in results.columns

    def test_one_row(self, splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*splits, models=["random_forest"], verbose=False)
        assert len(results) == 1
