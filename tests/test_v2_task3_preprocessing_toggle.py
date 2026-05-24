"""
v2 Task 3 tests — preprocessing toggle in Benchmarker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

SKLEARN_MODELS = ["random_forest", "logistic_regression"]


@pytest.fixture
def clean_splits():
    """Pure-numeric splits with no missing values."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y_s = pd.Series(y, name="target")
    return X_df[:160], y_s[:160], X_df[160:], y_s[160:]


@pytest.fixture
def mixed_splits():
    """Mixed dtype splits with missing values and categorical columns."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=["age", "income", "score", "weight"])

    # Introduce missing values
    df.loc[0, "age"] = np.nan
    df.loc[5, "income"] = np.nan

    # Add a categorical column
    cats = ["NY", "LA", "SF", "CH"]
    df["city"] = [cats[i % 4] for i in range(len(df))]

    y_s = pd.Series(y, name="target")
    return df[:160], y_s[:160], df[160:], y_s[160:]


# ---------------------------------------------------------------------------
# Constructor accepts preprocessing arg
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_preprocessing_default_false(self):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_MODELS, verbose=False)
        assert b.preprocessing is False

    def test_preprocessing_can_be_set_true(self):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_MODELS, preprocessing=True, verbose=False)
        assert b.preprocessing is True

    def test_preprocessing_false_no_overhead(self, clean_splits):
        """preprocessing=False must behave identically to not passing the arg."""
        from tfm_benchmark import Benchmarker
        b1 = Benchmarker(models=["random_forest"], verbose=False)
        b2 = Benchmarker(models=["random_forest"], preprocessing=False, verbose=False)
        r1 = b1.fit_evaluate(*clean_splits)
        r2 = b2.fit_evaluate(*clean_splits)
        # Both should succeed and return same columns
        assert set(r1.columns) == set(r2.columns)


# ---------------------------------------------------------------------------
# Preprocessing=True runs without errors
# ---------------------------------------------------------------------------

class TestPreprocessingTrue:
    def test_runs_with_clean_numeric(self, clean_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_MODELS, preprocessing=True, verbose=False)
        results = b.fit_evaluate(*clean_splits)
        assert len(results) == 2
        assert results["success"].all()

    def test_runs_with_mixed_dtype(self, mixed_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=["random_forest"], preprocessing=True, verbose=False)
        results = b.fit_evaluate(*mixed_splits)
        assert results["success"].all()

    def test_results_have_preprocessing_column(self, clean_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_MODELS, preprocessing=True, verbose=False)
        results = b.fit_evaluate(*clean_splits)
        assert "preprocessing" in results.columns

    def test_preprocessing_column_true_when_enabled(self, clean_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_MODELS, preprocessing=True, verbose=False)
        results = b.fit_evaluate(*clean_splits)
        assert results["preprocessing"].all()  # all rows True


# ---------------------------------------------------------------------------
# preprocessing=False results also have the column (set to False)
# ---------------------------------------------------------------------------

class TestPreprocessingFalseColumn:
    def test_results_have_preprocessing_column_when_false(self, clean_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_MODELS, preprocessing=False, verbose=False)
        results = b.fit_evaluate(*clean_splits)
        assert "preprocessing" in results.columns

    def test_preprocessing_column_false_when_disabled(self, clean_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_MODELS, preprocessing=False, verbose=False)
        results = b.fit_evaluate(*clean_splits)
        assert not results["preprocessing"].any()  # all rows False


# ---------------------------------------------------------------------------
# Preprocessor fit ONCE per fit_evaluate() call
# ---------------------------------------------------------------------------

class TestFitOnce:
    def test_preprocessor_fit_once_not_per_model(self, clean_splits):
        """
        With preprocessing=True and 2 models, the preprocessor should be fit
        once.  We verify by patching BasicPreprocessor.fit and counting calls.
        """
        from unittest.mock import patch, MagicMock
        from tfm_benchmark import Benchmarker
        from src.data import preprocessor as pp_module

        fit_call_count = []
        original_fit = pp_module.BasicPreprocessor.fit

        def counting_fit(self, X, y=None):
            fit_call_count.append(1)
            return original_fit(self, X, y)

        with patch.object(pp_module.BasicPreprocessor, "fit", counting_fit):
            b = Benchmarker(models=SKLEARN_MODELS, preprocessing=True, verbose=False)
            b.fit_evaluate(*clean_splits)

        assert len(fit_call_count) == 1, (
            f"BasicPreprocessor.fit should be called once, "
            f"but was called {len(fit_call_count)} times"
        )


# ---------------------------------------------------------------------------
# run_benchmark API respects preprocessing flag
# ---------------------------------------------------------------------------

class TestRunBenchmarkAPI:
    def test_run_benchmark_preprocessing_false_default(self, clean_splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*clean_splits, models=SKLEARN_MODELS, verbose=False)
        assert "preprocessing" in results.columns
        assert not results["preprocessing"].any()

    def test_run_benchmark_preprocessing_true(self, clean_splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(
            *clean_splits, models=SKLEARN_MODELS, preprocessing=True, verbose=False
        )
        assert "preprocessing" in results.columns
        assert results["preprocessing"].all()
