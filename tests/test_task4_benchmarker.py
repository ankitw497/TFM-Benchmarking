"""
Task 4 tests — tfm_benchmark/benchmarker.py
Benchmarker class: fit_evaluate(), plot_leaderboard(), save_results().
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures — lightweight data that exercises the full pipeline without GPU
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_splits():
    """200-row binary classification split using always-available sklearn data.
    Returns (X_train, y_train, X_test, y_test) matching fit_evaluate signature.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=4,
        random_state=42, n_classes=2
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y, name="target")
    return X_df[:160], y_s[:160], X_df[160:], y_s[160:]


SKLEARN_ONLY = ["random_forest", "logistic_regression"]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBenchmarkerInit:
    def test_importable(self):
        from tfm_benchmark import Benchmarker
        assert callable(Benchmarker)

    def test_default_models_auto(self):
        from tfm_benchmark import Benchmarker
        b = Benchmarker()
        assert b.models == "auto"

    def test_explicit_model_strings(self):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=["random_forest", "logistic_regression"])
        assert b.models == ["random_forest", "logistic_regression"]

    def test_unknown_model_key_raises(self):
        from tfm_benchmark import Benchmarker
        with pytest.raises(ValueError, match="not_a_real_model"):
            Benchmarker(models=["not_a_real_model"])

    def test_wrapper_instances_accepted(self):
        from tfm_benchmark import Benchmarker
        from src.models.sklearn_wrapper import RandomForestWrapper
        b = Benchmarker(models=[RandomForestWrapper()])
        assert len(b.models) == 1

    def test_dataset_name_stored(self):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY, dataset_name="my_data")
        assert b.dataset_name == "my_data"

    def test_results_none_before_run(self):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        assert b.results_ is None


# ---------------------------------------------------------------------------
# fit_evaluate — return shape and columns
# ---------------------------------------------------------------------------

class TestFitEvaluate:
    def test_returns_dataframe(self, sample_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        results = b.fit_evaluate(*sample_splits)
        assert isinstance(results, pd.DataFrame)

    def test_results_stored_on_instance(self, sample_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        results = b.fit_evaluate(*sample_splits)
        assert b.results_ is not None
        pd.testing.assert_frame_equal(results, b.results_)

    def test_one_row_per_model(self, sample_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        results = b.fit_evaluate(*sample_splits)
        assert len(results) == len(SKLEARN_ONLY)

    def test_required_columns_present(self, sample_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        results = b.fit_evaluate(*sample_splits)
        required = ["model_name", "auc_roc", "accuracy", "log_loss_val",
                    "fit_time", "predict_time", "success"]
        for col in required:
            assert col in results.columns, f"Missing column: {col}"

    def test_sklearn_models_succeed(self, sample_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        results = b.fit_evaluate(*sample_splits)
        assert results["success"].all(), (
            f"Expected all sklearn models to succeed:\n{results[['model_name','success','error_message']]}"
        )

    def test_auc_roc_in_valid_range(self, sample_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        results = b.fit_evaluate(*sample_splits)
        successful = results[results["success"]]
        assert (successful["auc_roc"] >= 0.0).all()
        assert (successful["auc_roc"] <= 1.0).all()

    def test_dataset_name_in_results(self, sample_splits):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY, dataset_name="test_ds")
        results = b.fit_evaluate(*sample_splits)
        assert (results["dataset_name"] == "test_ds").all()

    def test_missing_optional_model_skipped(self, sample_splits):
        """A model whose library isn't installed must appear with success=False, not crash."""
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=["tabpfn_v2", "random_forest"])
        results = b.fit_evaluate(*sample_splits)
        # random_forest must always succeed
        rf_row = results[results["model_name"].str.contains("Random|random", case=False)]
        assert len(rf_row) == 1
        assert rf_row.iloc[0]["success"]
        # Overall result is still a DataFrame — no crash
        assert isinstance(results, pd.DataFrame)

    def test_accepts_numpy_arrays(self):
        """fit_evaluate must accept numpy arrays as well as DataFrames."""
        from tfm_benchmark import Benchmarker
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, 100)
        b = Benchmarker(models=["logistic_regression"])
        # Order: X_train, y_train, X_test, y_test
        results = b.fit_evaluate(X[:80], y[:80], X[80:], y[80:])
        assert isinstance(results, pd.DataFrame)
        assert results.iloc[0]["success"]

    def test_results_sorted_by_auc_desc(self, sample_splits):
        """Results should be sorted by AUC-ROC descending (best first)."""
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        results = b.fit_evaluate(*sample_splits)
        successful = results[results["success"]].reset_index(drop=True)
        if len(successful) > 1:
            aucs = successful["auc_roc"].tolist()
            assert aucs == sorted(aucs, reverse=True), "Results not sorted by AUC descending"


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_save_creates_file(self, sample_splits, tmp_path):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        b.fit_evaluate(*sample_splits)
        out = tmp_path / "results.csv"
        b.save_results(str(out))
        assert out.exists()

    def test_saved_csv_readable(self, sample_splits, tmp_path):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        b.fit_evaluate(*sample_splits)
        out = tmp_path / "results.csv"
        b.save_results(str(out))
        df = pd.read_csv(out)
        assert "model_name" in df.columns
        assert "auc_roc" in df.columns

    def test_save_creates_parent_dirs(self, sample_splits, tmp_path):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        b.fit_evaluate(*sample_splits)
        out = tmp_path / "nested" / "deep" / "results.csv"
        b.save_results(str(out))
        assert out.exists()

    def test_save_before_run_raises(self, tmp_path):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        with pytest.raises(RuntimeError, match="fit_evaluate"):
            b.save_results(str(tmp_path / "out.csv"))


# ---------------------------------------------------------------------------
# plot_leaderboard
# ---------------------------------------------------------------------------

class TestPlotLeaderboard:
    def test_returns_figure(self, sample_splits):
        import matplotlib
        matplotlib.use("Agg")  # headless
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        b.fit_evaluate(*sample_splits)
        fig = b.plot_leaderboard(show=False)
        import matplotlib.pyplot as plt
        assert fig is not None

    def test_plot_before_run_raises(self):
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        with pytest.raises(RuntimeError, match="fit_evaluate"):
            b.plot_leaderboard(show=False)

    def test_plot_saves_file(self, sample_splits, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=SKLEARN_ONLY)
        b.fit_evaluate(*sample_splits)
        out = tmp_path / "leaderboard.png"
        b.plot_leaderboard(save_path=str(out), show=False)
        assert out.exists()
        assert out.stat().st_size > 0
