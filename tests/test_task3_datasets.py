"""
Task 3 tests — tfm_benchmark/datasets.py
load_dataset() and list_datasets() public API.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_returns_list(self):
        from tfm_benchmark.datasets import list_datasets
        result = list_datasets()
        assert isinstance(result, list)

    def test_contains_bundled_datasets(self):
        from tfm_benchmark.datasets import list_datasets
        result = list_datasets()
        assert "german_credit" in result
        assert "taiwan_credit" in result
        assert "synthetic" in result

    def test_does_not_require_kaggle(self):
        """give_me_credit requires Kaggle — must NOT appear in the no-Kaggle list."""
        from tfm_benchmark.datasets import list_datasets
        result = list_datasets()
        assert "give_me_credit" not in result


# ---------------------------------------------------------------------------
# load_dataset — bundled datasets
# ---------------------------------------------------------------------------

class TestLoadBundledDatasets:
    def test_load_german_credit_returns_four_splits(self):
        from tfm_benchmark.datasets import load_dataset
        result = load_dataset("german_credit")
        assert len(result) == 4
        X_train, X_test, y_train, y_test = result
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_load_german_credit_shapes_consistent(self):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, y_train, y_test = load_dataset("german_credit")
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_load_german_credit_default_split(self):
        """Default 80/20 split on ~1000 rows."""
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, y_train, y_test = load_dataset("german_credit")
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.15 <= test_ratio <= 0.25  # ~20%

    def test_load_taiwan_credit_returns_four_splits(self):
        from tfm_benchmark.datasets import load_dataset
        result = load_dataset("taiwan_credit")
        assert len(result) == 4

    def test_load_synthetic_returns_four_splits(self):
        from tfm_benchmark.datasets import load_dataset
        result = load_dataset("synthetic")
        assert len(result) == 4

    def test_load_synthetic_binary_target(self):
        from tfm_benchmark.datasets import load_dataset
        _, _, y_train, y_test = load_dataset("synthetic")
        all_values = pd.concat([y_train, y_test])
        assert set(all_values.unique()).issubset({0, 1})

    def test_max_rows_respected(self):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, y_train, y_test = load_dataset("german_credit", max_rows=200)
        assert len(X_train) + len(X_test) <= 200

    def test_custom_test_size(self):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, y_train, y_test = load_dataset(
            "german_credit", test_size=0.3
        )
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.25 <= test_ratio <= 0.35

    def test_reproducible_with_random_state(self):
        from tfm_benchmark.datasets import load_dataset
        result_a = load_dataset("german_credit", random_state=42)
        result_b = load_dataset("german_credit", random_state=42)
        pd.testing.assert_frame_equal(result_a[0], result_b[0])  # X_train identical

    def test_different_seeds_give_different_splits(self):
        from tfm_benchmark.datasets import load_dataset
        X_train_42, *_ = load_dataset("german_credit", random_state=42)
        X_train_99, *_ = load_dataset("german_credit", random_state=99)
        # Different seeds → different rows in train set (first rows differ)
        assert not X_train_42.iloc[0].equals(X_train_99.iloc[0])

    def test_unknown_dataset_raises_value_error(self):
        from tfm_benchmark.datasets import load_dataset
        with pytest.raises(ValueError, match="unknown_dataset"):
            load_dataset("unknown_dataset")


# ---------------------------------------------------------------------------
# load_dataset — BYO: CSV file path
# ---------------------------------------------------------------------------

class TestLoadCSVFile:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Write a small CSV with a known target column."""
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "feature_a": rng.randn(100),
            "feature_b": rng.randn(100),
            "feature_c": rng.randint(0, 5, 100).astype(float),
            "label": rng.randint(0, 2, 100),
        })
        csv_path = tmp_path / "sample.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_csv_returns_four_splits(self, sample_csv):
        from tfm_benchmark.datasets import load_dataset
        result = load_dataset(sample_csv, target="label")
        assert len(result) == 4

    def test_csv_target_column_removed_from_features(self, sample_csv):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, _, _ = load_dataset(sample_csv, target="label")
        assert "label" not in X_train.columns
        assert "label" not in X_test.columns

    def test_csv_target_is_series(self, sample_csv):
        from tfm_benchmark.datasets import load_dataset
        _, _, y_train, y_test = load_dataset(sample_csv, target="label")
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_csv_shapes_consistent(self, sample_csv):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, y_train, y_test = load_dataset(sample_csv, target="label")
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_csv_requires_target_when_ambiguous(self):
        """Loading a CSV without specifying target should raise ValueError."""
        from tfm_benchmark.datasets import load_dataset
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            path = f.name
        with pytest.raises(ValueError, match="target"):
            load_dataset(path)

    def test_csv_missing_target_column_raises(self, sample_csv):
        from tfm_benchmark.datasets import load_dataset
        with pytest.raises((ValueError, KeyError)):
            load_dataset(sample_csv, target="nonexistent_column")

    def test_nonexistent_file_raises(self):
        from tfm_benchmark.datasets import load_dataset
        with pytest.raises((FileNotFoundError, ValueError)):
            load_dataset("/tmp/does_not_exist_xyz_abc.csv", target="label")


# ---------------------------------------------------------------------------
# load_dataset — BYO: pandas DataFrame
# ---------------------------------------------------------------------------

class TestLoadDataFrame:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(7)
        return pd.DataFrame({
            "x1": rng.randn(150),
            "x2": rng.randn(150),
            "target": rng.randint(0, 2, 150),
        })

    def test_dataframe_returns_four_splits(self, sample_df):
        from tfm_benchmark.datasets import load_dataset
        result = load_dataset(sample_df, target="target")
        assert len(result) == 4

    def test_dataframe_target_removed_from_features(self, sample_df):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, _, _ = load_dataset(sample_df, target="target")
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_dataframe_shapes_consistent(self, sample_df):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, y_train, y_test = load_dataset(sample_df, target="target")
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_dataframe_default_split(self, sample_df):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, _, _ = load_dataset(sample_df, target="target")
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_dataframe_requires_target(self, sample_df):
        from tfm_benchmark.datasets import load_dataset
        with pytest.raises(ValueError, match="target"):
            load_dataset(sample_df)

    def test_dataframe_max_rows(self, sample_df):
        from tfm_benchmark.datasets import load_dataset
        X_train, X_test, _, _ = load_dataset(sample_df, target="target", max_rows=80)
        assert len(X_train) + len(X_test) <= 80
