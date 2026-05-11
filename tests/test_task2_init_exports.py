"""
Task 2 tests — src/ __init__.py public exports.
All imports below MUST work after Task 2 is implemented.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# src/__init__.py
# ---------------------------------------------------------------------------

class TestSrcRoot:
    def test_src_version(self):
        import src
        assert hasattr(src, "__version__")
        assert src.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# src/data/__init__.py
# ---------------------------------------------------------------------------

class TestDataInit:
    def test_load_credit_dataset_importable(self):
        from src.data import load_credit_dataset
        assert callable(load_credit_dataset)

    def test_get_dataset_info_importable(self):
        from src.data import get_dataset_info
        assert callable(get_dataset_info)

    def test_get_cv_splits_importable(self):
        from src.data import get_cv_splits
        assert callable(get_cv_splits)

    def test_star_import_exports(self):
        """__all__ must cover the three main public functions."""
        import src.data as d
        assert hasattr(d, "load_credit_dataset")
        assert hasattr(d, "get_dataset_info")
        assert hasattr(d, "get_cv_splits")


# ---------------------------------------------------------------------------
# src/models/__init__.py
# ---------------------------------------------------------------------------

class TestModelsInit:
    def test_base_classes_importable(self):
        from src.models import BaseModelWrapper, BenchmarkResult, ModelLimitations
        assert BaseModelWrapper is not None
        assert BenchmarkResult is not None
        assert ModelLimitations is not None

    def test_model_registry_exists(self):
        from src.models import MODEL_REGISTRY
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0

    def test_model_registry_keys_are_strings(self):
        from src.models import MODEL_REGISTRY
        for key in MODEL_REGISTRY:
            assert isinstance(key, str), f"Non-string key: {key!r}"

    def test_model_registry_contains_sklearn_baselines(self):
        """Sklearn wrappers must always be in registry — no optional deps."""
        from src.models import MODEL_REGISTRY
        assert "random_forest" in MODEL_REGISTRY
        assert "logistic_regression" in MODEL_REGISTRY

    def test_model_registry_contains_known_keys(self):
        from src.models import MODEL_REGISTRY
        expected_keys = [
            "random_forest", "logistic_regression",
            "xgboost", "catboost", "lightgbm",
            "tabpfn_v2", "tabicl_v2",
        ]
        for k in expected_keys:
            assert k in MODEL_REGISTRY, f"Missing key: {k}"

    def test_registry_values_are_callables(self):
        """Each registry value must be a zero/few-arg callable (factory function)."""
        from src.models import MODEL_REGISTRY
        for key, factory in MODEL_REGISTRY.items():
            assert callable(factory), f"Registry value for {key!r} is not callable"

    def test_sklearn_registry_factories_produce_wrapper(self):
        """Sklearn factories must return BaseModelWrapper instances."""
        from src.models import MODEL_REGISTRY, BaseModelWrapper
        for key in ("random_forest", "logistic_regression"):
            instance = MODEL_REGISTRY[key]()
            assert isinstance(instance, BaseModelWrapper), (
                f"{key} factory returned {type(instance)}, expected BaseModelWrapper"
            )


# ---------------------------------------------------------------------------
# src/evaluation/__init__.py
# ---------------------------------------------------------------------------

class TestEvaluationInit:
    def test_compute_all_metrics_importable(self):
        from src.evaluation import compute_all_metrics
        assert callable(compute_all_metrics)

    def test_compute_ece_importable(self):
        from src.evaluation import compute_ece
        assert callable(compute_ece)

    def test_aggregate_cv_results_importable(self):
        from src.evaluation import aggregate_cv_results
        assert callable(aggregate_cv_results)

    def test_create_comparison_table_importable(self):
        from src.evaluation import create_comparison_table
        assert callable(create_comparison_table)

    def test_compute_all_metrics_works(self):
        """Basic smoke test — function must return dict with expected keys."""
        import numpy as np
        from src.evaluation import compute_all_metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])
        result = compute_all_metrics(y_true, y_prob)
        assert isinstance(result, dict)
        assert "auc_roc" in result
        assert "log_loss" in result
        assert "ece" in result


# ---------------------------------------------------------------------------
# src/visualization/__init__.py
# ---------------------------------------------------------------------------

class TestVisualizationInit:
    def test_plot_leaderboard_importable(self):
        from src.visualization import plot_leaderboard
        assert callable(plot_leaderboard)

    def test_generate_all_plots_importable(self):
        from src.visualization import generate_all_plots
        assert callable(generate_all_plots)

    def test_plot_scaling_curves_importable(self):
        from src.visualization import plot_scaling_curves
        assert callable(plot_scaling_curves)

    def test_plot_finetuning_impact_importable(self):
        from src.visualization import plot_finetuning_impact
        assert callable(plot_finetuning_impact)
