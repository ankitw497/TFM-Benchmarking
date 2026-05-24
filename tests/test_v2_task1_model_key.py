"""
v2 Task 1 tests — model_key field in BenchmarkResult and propagation through pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

SKLEARN_MODELS = ["random_forest", "logistic_regression"]


@pytest.fixture
def splits():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y_s = pd.Series(y, name="target")
    return X_df[:120], y_s[:120], X_df[120:], y_s[120:]


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResultField:
    def test_model_key_field_exists(self):
        from src.models.base import BenchmarkResult
        r = BenchmarkResult(model_name="MyModel", dataset_name="ds", phase="zero_shot")
        assert hasattr(r, "model_key")

    def test_model_key_defaults_to_empty_string(self):
        from src.models.base import BenchmarkResult
        r = BenchmarkResult(model_name="MyModel", dataset_name="ds", phase="zero_shot")
        assert r.model_key == ""

    def test_model_key_can_be_set(self):
        from src.models.base import BenchmarkResult
        r = BenchmarkResult(model_name="MyModel", dataset_name="ds", phase="zero_shot",
                            model_key="random_forest")
        assert r.model_key == "random_forest"

    def test_to_dict_includes_model_key(self):
        from src.models.base import BenchmarkResult
        r = BenchmarkResult(model_name="M", dataset_name="d", phase="p",
                            model_key="xgboost")
        d = r.to_dict()
        assert "model_key" in d
        assert d["model_key"] == "xgboost"

    def test_backwards_compatible_no_model_key(self):
        """Existing code that omits model_key must not break."""
        from src.models.base import BenchmarkResult
        r = BenchmarkResult(model_name="X", dataset_name="d", phase="p")
        assert r.model_key == ""


# ---------------------------------------------------------------------------
# BaseModelWrapper.evaluate() accepts model_key kwarg
# ---------------------------------------------------------------------------

class TestEvaluateAcceptsModelKey:
    def test_evaluate_accepts_model_key(self):
        from src.models.sklearn_wrapper import RandomForestWrapper
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, random_state=1)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)
        m = RandomForestWrapper()
        result = m.evaluate(X_df[:80], y_s[:80], X_df[80:], y_s[80:],
                            model_key="random_forest")
        assert result.model_key == "random_forest"

    def test_evaluate_model_key_default_empty(self):
        """Callers that don't pass model_key get "" — no breakage."""
        from src.models.sklearn_wrapper import RandomForestWrapper
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, random_state=2)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)
        m = RandomForestWrapper()
        result = m.evaluate(X_df[:80], y_s[:80], X_df[80:], y_s[80:])
        assert result.model_key == ""


# ---------------------------------------------------------------------------
# run_benchmark / Benchmarker propagates model_key into results DataFrame
# ---------------------------------------------------------------------------

class TestModelKeyInResults:
    def test_run_benchmark_has_model_key_column(self, splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*splits, models=SKLEARN_MODELS, verbose=False)
        assert "model_key" in results.columns

    def test_model_key_matches_registry_key(self, splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*splits, models=SKLEARN_MODELS, verbose=False)
        keys_in_results = set(results["model_key"].tolist())
        assert keys_in_results == set(SKLEARN_MODELS)

    def test_model_key_nonempty_for_all_rows(self, splits):
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*splits, models=SKLEARN_MODELS, verbose=False)
        assert (results["model_key"] != "").all()

    def test_benchmarker_has_model_key_column(self, splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=["random_forest"], verbose=False)
        results = b.fit_evaluate(*splits)
        assert "model_key" in results.columns
        assert results.iloc[0]["model_key"] == "random_forest"

    def test_model_name_still_present(self, splits):
        """model_name (display name) must still exist alongside model_key."""
        from tfm_benchmark import run_benchmark
        results = run_benchmark(*splits, models=["random_forest"], verbose=False)
        assert "model_name" in results.columns
        assert results.iloc[0]["model_name"] != ""
