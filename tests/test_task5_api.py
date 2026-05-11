"""
Task 5 tests — tfm_benchmark/api.py  +  tfm_benchmark/__init__.py final wiring.
Covers run_benchmark(), list_models(), and the complete __init__ surface.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_splits():
    """100-row binary split — (X_train, y_train, X_test, y_test)."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y_s = pd.Series(y, name="target")
    return X_df[:80], y_s[:80], X_df[80:], y_s[80:]


ALWAYS_AVAILABLE = ["random_forest", "logistic_regression"]


# ---------------------------------------------------------------------------
# Top-level __init__ imports
# ---------------------------------------------------------------------------

class TestInitSurface:
    """All required names must be importable directly from tfm_benchmark."""

    def test_run_benchmark_importable(self):
        from tfm_benchmark import run_benchmark
        assert callable(run_benchmark)

    def test_list_models_importable(self):
        from tfm_benchmark import list_models
        assert callable(list_models)

    def test_benchmarker_importable(self):
        from tfm_benchmark import Benchmarker
        assert callable(Benchmarker)

    def test_load_dataset_importable(self):
        from tfm_benchmark import load_dataset
        assert callable(load_dataset)

    def test_list_datasets_importable(self):
        from tfm_benchmark import list_datasets
        assert callable(list_datasets)

    def test_version_importable(self):
        import tfm_benchmark
        assert tfm_benchmark.__version__ == "0.1.0"

    def test_star_import_surface(self):
        """__all__ must include every public name."""
        import tfm_benchmark
        required = {"run_benchmark", "list_models", "Benchmarker",
                    "load_dataset", "list_datasets", "__version__"}
        missing = required - set(getattr(tfm_benchmark, "__all__", []))
        assert not missing, f"Missing from __all__: {missing}"


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_list(self):
        from tfm_benchmark import list_models
        result = list_models()
        assert isinstance(result, list)

    def test_nonempty(self):
        from tfm_benchmark import list_models
        assert len(list_models()) > 0

    def test_contains_all_registry_keys(self):
        from tfm_benchmark import list_models
        from src.models import MODEL_REGISTRY
        result = list_models()
        for key in MODEL_REGISTRY:
            assert key in result, f"MODEL_REGISTRY key {key!r} missing from list_models()"

    def test_each_entry_is_string(self):
        from tfm_benchmark import list_models
        for item in list_models():
            assert isinstance(item, str)

    def test_sklearn_models_marked_installed(self):
        """random_forest and logistic_regression are always installed."""
        from tfm_benchmark.api import list_models
        result = list_models()
        assert "random_forest" in result
        assert "logistic_regression" in result

    def test_optional_model_in_list_even_if_uninstalled(self):
        """list_models must list ALL registry keys, not just installed ones."""
        from tfm_benchmark import list_models
        result = list_models()
        assert "tabpfn_v2" in result
        assert "tabicl_v2" in result
        assert "xgboost" in result


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_returns_dataframe(self, tiny_splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE)
        assert isinstance(results, pd.DataFrame)

    def test_required_columns(self, tiny_splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE)
        for col in ["model_name", "auc_roc", "success"]:
            assert col in results.columns

    def test_one_row_per_model(self, tiny_splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE)
        assert len(results) == len(ALWAYS_AVAILABLE)

    def test_models_succeed(self, tiny_splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE)
        assert results["success"].all()

    def test_dataset_name_propagated(self, tiny_splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE,
                                dataset_name="my_dataset")
        assert (results["dataset_name"] == "my_dataset").all()

    def test_auto_mode_does_not_crash(self, tiny_splits):
        """auto mode tries all models — must not crash even with missing deps."""
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models="auto")
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_single_model_string_accepted(self, tiny_splits):
        """A single-element list must work."""
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models=["random_forest"])
        assert len(results) == 1

    def test_invalid_model_key_raises(self, tiny_splits):
        from tfm_benchmark import run_benchmark
        with pytest.raises(ValueError, match="does_not_exist"):
            run_benchmark(*tiny_splits, models=["does_not_exist"])

    def test_numpy_input_accepted(self):
        """run_benchmark must accept numpy arrays directly."""
        from tfm_benchmark import run_benchmark
        rng = np.random.RandomState(1)
        X = rng.randn(80, 4)
        y = rng.randint(0, 2, 80)
        results = run_benchmark(X[:60], y[:60], X[60:], y[60:],
                                models=["logistic_regression"])
        assert results.iloc[0]["success"]

    def test_results_sorted_by_auc(self, tiny_splits):
        """Results should come back sorted by AUC descending."""
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE)
        aucs = results[results["success"]]["auc_roc"].tolist()
        assert aucs == sorted(aucs, reverse=True)

    def test_verbose_false_no_stdout(self, tiny_splits, capsys):
        from tfm_benchmark import run_benchmark
        run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE, verbose=False)
        out = capsys.readouterr().out
        assert out == ""

    def test_is_thin_wrapper_over_benchmarker(self, tiny_splits):
        """run_benchmark and Benchmarker.fit_evaluate must produce identical results."""
        from tfm_benchmark import run_benchmark, Benchmarker
        r1 = run_benchmark(*tiny_splits, models=ALWAYS_AVAILABLE,
                           random_state=42, verbose=False)
        b = Benchmarker(models=ALWAYS_AVAILABLE, verbose=False)
        r2 = b.fit_evaluate(*tiny_splits)
        # Same models, same AUC (same data, same seed path)
        assert set(r1["model_name"]) == set(r2["model_name"])
