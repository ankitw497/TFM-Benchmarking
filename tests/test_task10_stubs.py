"""
Task 10 tests — stub modules in src/.

All five stubs must be importable without error and expose their public API.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# src/data/preprocessor.py — BasicPreprocessor
# ---------------------------------------------------------------------------

class TestBasicPreprocessor:
    def test_importable(self):
        from src.data.preprocessor import BasicPreprocessor
        assert callable(BasicPreprocessor)

    def test_instantiable(self):
        from src.data.preprocessor import BasicPreprocessor
        bp = BasicPreprocessor()
        assert bp is not None

    def test_fit_transform_returns_dataframe(self):
        from src.data.preprocessor import BasicPreprocessor
        bp = BasicPreprocessor()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = bp.fit_transform(X)
        assert isinstance(result, (pd.DataFrame, np.ndarray))

    def test_fit_then_transform(self):
        from src.data.preprocessor import BasicPreprocessor
        bp = BasicPreprocessor()
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        bp.fit(X)
        result = bp.transform(X)
        assert result is not None


# ---------------------------------------------------------------------------
# src/data/splitter.py — stratified_split
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def test_importable(self):
        from src.data.splitter import stratified_split
        assert callable(stratified_split)

    def test_returns_four_tuple(self):
        from src.data.splitter import stratified_split
        X = pd.DataFrame({"f": range(100)})
        y = pd.Series([0] * 50 + [1] * 50)
        result = stratified_split(X, y)
        assert len(result) == 4

    def test_correct_sizes(self):
        from src.data.splitter import stratified_split
        X = pd.DataFrame({"f": range(100)})
        y = pd.Series([0] * 50 + [1] * 50)
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == 100

    def test_preserves_class_ratio(self):
        from src.data.splitter import stratified_split
        X = pd.DataFrame({"f": range(200)})
        y = pd.Series([0] * 150 + [1] * 50)
        _, _, y_train, y_test = stratified_split(X, y, test_size=0.2)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05


# ---------------------------------------------------------------------------
# src/evaluation/timing.py — TimingContext
# ---------------------------------------------------------------------------

class TestTimingContext:
    def test_importable(self):
        from src.evaluation.timing import TimingContext
        assert callable(TimingContext)

    def test_context_manager(self):
        from src.evaluation.timing import TimingContext
        with TimingContext() as tc:
            _ = sum(range(1000))
        assert hasattr(tc, "elapsed") or hasattr(tc, "duration") or hasattr(tc, "elapsed_s")

    def test_records_positive_time(self):
        from src.evaluation.timing import TimingContext
        with TimingContext() as tc:
            _ = [i**2 for i in range(5000)]
        duration = getattr(tc, "elapsed", None) or getattr(tc, "duration", None) or getattr(tc, "elapsed_s", 0)
        assert duration >= 0.0


# ---------------------------------------------------------------------------
# src/evaluation/memory.py — get_peak_memory
# ---------------------------------------------------------------------------

class TestGetPeakMemory:
    def test_importable(self):
        from src.evaluation.memory import get_peak_memory
        assert callable(get_peak_memory)

    def test_returns_float(self):
        from src.evaluation.memory import get_peak_memory
        result = get_peak_memory()
        assert isinstance(result, float)

    def test_returns_nonnegative(self):
        from src.evaluation.memory import get_peak_memory
        assert get_peak_memory() >= 0.0


# ---------------------------------------------------------------------------
# src/visualization/leaderboard.py — generate_leaderboard_table
# ---------------------------------------------------------------------------

class TestGenerateLeaderboardTable:
    def test_importable(self):
        from src.visualization.leaderboard import generate_leaderboard_table
        assert callable(generate_leaderboard_table)

    def test_returns_dataframe(self):
        from src.visualization.leaderboard import generate_leaderboard_table
        results = {
            "ModelA": {"auc_roc": 0.85, "accuracy": 0.80},
            "ModelB": {"auc_roc": 0.90, "accuracy": 0.85},
        }
        df = generate_leaderboard_table(results)
        assert isinstance(df, pd.DataFrame)

    def test_has_model_names(self):
        from src.visualization.leaderboard import generate_leaderboard_table
        results = {
            "ModelA": {"auc_roc": 0.85},
            "ModelB": {"auc_roc": 0.90},
        }
        df = generate_leaderboard_table(results)
        combined = str(df)
        assert "ModelA" in combined or "ModelB" in combined
