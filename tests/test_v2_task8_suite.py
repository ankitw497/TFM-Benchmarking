"""
v2 Task 8 tests — run_benchmark_suite() multi-dataset aggregated benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

SKLEARN_MODELS = ["random_forest", "logistic_regression"]


def _make_split(n=200, n_features=6, seed=0):
    """Return (X_train, y_train, X_test, y_test) as DataFrames/Series."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=n_features, random_state=seed)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y_s = pd.Series(y, name="target")
    n_train = int(n * 0.8)
    return X_df[:n_train], y_s[:n_train], X_df[n_train:], y_s[n_train:]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class TestImport:
    def test_importable_from_tfm_benchmark(self):
        from tfm_benchmark import run_benchmark_suite
        assert callable(run_benchmark_suite)

    def test_in_all(self):
        import tfm_benchmark
        assert "run_benchmark_suite" in tfm_benchmark.__all__


# ---------------------------------------------------------------------------
# Tuple-based dataset input
# ---------------------------------------------------------------------------

class TestTupleDatasets:
    def test_single_tuple_dataset(self):
        from tfm_benchmark import run_benchmark_suite
        ds1 = ("ds_a",) + _make_split(seed=1)
        results = run_benchmark_suite(
            datasets=[ds1],
            models=SKLEARN_MODELS,
            verbose=False,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # 2 models

    def test_multi_tuple_datasets(self):
        from tfm_benchmark import run_benchmark_suite
        ds1 = ("ds_a",) + _make_split(seed=1)
        ds2 = ("ds_b",) + _make_split(seed=2)
        results = run_benchmark_suite(
            datasets=[ds1, ds2],
            models=SKLEARN_MODELS,
            verbose=False,
        )
        assert len(results) == 4  # 2 datasets × 2 models

    def test_dataset_name_column_present(self):
        from tfm_benchmark import run_benchmark_suite
        ds1 = ("my_dataset",) + _make_split(seed=0)
        results = run_benchmark_suite(datasets=[ds1], models=["random_forest"], verbose=False)
        assert "dataset_name" in results.columns

    def test_dataset_name_values_match(self):
        from tfm_benchmark import run_benchmark_suite
        ds1 = ("alpha",) + _make_split(seed=1)
        ds2 = ("beta",) + _make_split(seed=2)
        results = run_benchmark_suite(datasets=[ds1, ds2], models=["random_forest"], verbose=False)
        names = set(results["dataset_name"].tolist())
        assert names == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Standard result columns still present
# ---------------------------------------------------------------------------

class TestResultColumns:
    def test_has_model_key_column(self):
        from tfm_benchmark import run_benchmark_suite
        ds = ("test_ds",) + _make_split(seed=42)
        results = run_benchmark_suite(datasets=[ds], models=SKLEARN_MODELS, verbose=False)
        assert "model_key" in results.columns

    def test_has_auc_roc_column(self):
        from tfm_benchmark import run_benchmark_suite
        ds = ("test_ds",) + _make_split(seed=42)
        results = run_benchmark_suite(datasets=[ds], models=SKLEARN_MODELS, verbose=False)
        assert "auc_roc" in results.columns

    def test_has_success_column(self):
        from tfm_benchmark import run_benchmark_suite
        ds = ("test_ds",) + _make_split(seed=42)
        results = run_benchmark_suite(datasets=[ds], models=SKLEARN_MODELS, verbose=False)
        assert "success" in results.columns


# ---------------------------------------------------------------------------
# Per-dataset errors don't abort the whole suite
# ---------------------------------------------------------------------------

class TestErrorIsolation:
    def test_failing_dataset_does_not_abort_suite(self):
        """A dataset that triggers an error should not abort remaining datasets."""
        from tfm_benchmark import run_benchmark_suite

        # Dataset 1: valid
        ds_good = ("good_ds",) + _make_split(seed=0)

        # Dataset 2: intentionally broken — wrong shapes
        bad_X = pd.DataFrame({"a": [1, 2, 3]})
        bad_y = pd.Series([0, 1])  # mismatched length
        ds_bad = ("bad_ds", bad_X, bad_y, bad_X, bad_y)

        # Dataset 3: valid
        ds_also_good = ("also_good",) + _make_split(seed=5)

        results = run_benchmark_suite(
            datasets=[ds_good, ds_bad, ds_also_good],
            models=["random_forest"],
            verbose=False,
        )

        # Should have results for both good datasets
        dataset_names_in_results = set(results["dataset_name"].tolist())
        assert "good_ds" in dataset_names_in_results
        assert "also_good" in dataset_names_in_results


# ---------------------------------------------------------------------------
# Named datasets route through load_dataset (integration smoke test)
# ---------------------------------------------------------------------------

class TestNamedDatasets:
    def test_named_dataset_key_accepted(self):
        """'synthetic' should be accepted as a named dataset key."""
        from tfm_benchmark import run_benchmark_suite
        results = run_benchmark_suite(
            datasets=["synthetic"],
            models=["random_forest"],
            verbose=False,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert "dataset_name" in results.columns
        assert results.iloc[0]["dataset_name"] == "synthetic"
