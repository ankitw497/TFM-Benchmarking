"""
v2 Task 2 tests — BasicPreprocessor: imputation, scaling, one-hot encoding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mixed_df():
    """DataFrame with numeric + categorical columns and some missing values."""
    return pd.DataFrame({
        "age":      [25.0, np.nan, 35.0, 45.0, np.nan],
        "income":   [50000.0, 60000.0, np.nan, 80000.0, 55000.0],
        "city":     ["NY", "LA", None, "NY", "SF"],
        "edu":      ["BSc", "MSc", "BSc", None, "PhD"],
    })


@pytest.fixture
def numeric_df():
    """Pure numeric DataFrame with some missing values."""
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": [10.0, np.nan, 30.0, 40.0],
    })


@pytest.fixture
def categorical_df():
    """Pure categorical DataFrame with some missing values."""
    return pd.DataFrame({
        "color": ["red", "blue", None, "red"],
        "size":  ["S", "M", "L", None],
    })


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_basic_preprocessor(self):
        from src.data.preprocessor import BasicPreprocessor
        assert BasicPreprocessor is not None


# ---------------------------------------------------------------------------
# No missing values after fit_transform
# ---------------------------------------------------------------------------

class TestNoMissingValues:
    def test_numeric_no_nans(self, numeric_df):
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(numeric_df)
        assert not out.isnull().any().any(), "Output should have no NaN values"

    def test_mixed_no_nans(self, mixed_df):
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(mixed_df)
        assert not out.isnull().any().any(), "Mixed df output should have no NaN values"

    def test_categorical_no_nans(self, categorical_df):
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(categorical_df)
        assert not out.isnull().any().any(), "Categorical df output should have no NaN values"


# ---------------------------------------------------------------------------
# Numeric columns are scaled (zero mean, unit variance)
# ---------------------------------------------------------------------------

class TestScaling:
    def test_numeric_mean_near_zero(self, numeric_df):
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(numeric_df)
        # Identify numeric output columns (OHE columns are binary 0/1)
        num_cols = [c for c in out.columns if c in ["a", "b"]]
        for col in num_cols:
            mean = out[col].mean()
            assert abs(mean) < 1e-9, f"Column {col} mean should be ~0, got {mean}"

    def test_numeric_std_near_one(self, numeric_df):
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(numeric_df)
        num_cols = [c for c in out.columns if c in ["a", "b"]]
        for col in num_cols:
            std = out[col].std(ddof=0)
            assert abs(std - 1.0) < 1e-6, f"Column {col} std should be ~1, got {std}"


# ---------------------------------------------------------------------------
# Categorical columns are one-hot encoded
# ---------------------------------------------------------------------------

class TestOneHotEncoding:
    def test_categorical_columns_expanded(self, categorical_df):
        """OHE should create more columns than the original categorical df."""
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(categorical_df)
        # 2 categorical cols with 3 values each → at least 4 OHE columns
        assert out.shape[1] > 2

    def test_ohe_values_are_binary(self, mixed_df):
        """OHE columns should only contain 0 and 1."""
        from src.data.preprocessor import BasicPreprocessor
        pp = BasicPreprocessor()
        out = pp.fit_transform(mixed_df)
        # Numeric cols (age, income) are scaled — may have negative values.
        # Identify OHE columns as those not in original numeric columns.
        original_num = mixed_df.select_dtypes(include="number").columns.tolist()
        ohe_cols = [c for c in out.columns if c not in original_num]
        if ohe_cols:
            ohe_vals = out[ohe_cols].values.flatten()
            assert set(ohe_vals).issubset({0, 1, 0.0, 1.0}), \
                f"OHE values should be 0/1 only, got: {set(ohe_vals)}"

    def test_output_column_names_are_strings(self, mixed_df):
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(mixed_df)
        for col in out.columns:
            assert isinstance(col, str), f"Column name {col!r} should be a string"

    def test_output_column_names_deterministic(self, mixed_df):
        """Two fit_transform calls on the same data → identical columns."""
        from src.data.preprocessor import BasicPreprocessor
        out1 = BasicPreprocessor().fit_transform(mixed_df)
        out2 = BasicPreprocessor().fit_transform(mixed_df)
        assert list(out1.columns) == list(out2.columns)


# ---------------------------------------------------------------------------
# fit / transform separation (no data leakage)
# ---------------------------------------------------------------------------

class TestNoDataLeakage:
    def test_fit_transform_same_as_fit_then_transform(self, numeric_df):
        from src.data.preprocessor import BasicPreprocessor
        pp1 = BasicPreprocessor()
        out1 = pp1.fit_transform(numeric_df)

        pp2 = BasicPreprocessor()
        pp2.fit(numeric_df)
        out2 = pp2.transform(numeric_df)

        pd.testing.assert_frame_equal(out1, out2)

    def test_transform_uses_train_stats(self):
        """Scaler mean/std must come from train set, not test set."""
        from src.data.preprocessor import BasicPreprocessor
        train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        test  = pd.DataFrame({"x": [100.0, 200.0, 300.0]})

        pp = BasicPreprocessor()
        pp.fit(train)
        out_train = pp.transform(train)
        out_test  = pp.transform(test)

        # Train mean should be ~0 after scaling
        assert abs(out_train["x"].mean()) < 1e-9

        # Test values should be very large (not re-scaled to mean=0)
        # because we used train stats (mean=1, std=~0.816)
        assert out_test["x"].mean() > 50  # sanity: 100/0.816 >> 1

    def test_median_imputation_from_train(self):
        """Imputation median must be learned on train, not recomputed on test."""
        from src.data.preprocessor import BasicPreprocessor
        train = pd.DataFrame({"x": [10.0, 20.0, 30.0]})  # median = 20
        test  = pd.DataFrame({"x": [np.nan, np.nan]})

        pp = BasicPreprocessor()
        pp.fit(train)
        out_test = pp.transform(test)

        # NaN in test should be filled with train median (20), scaled to 0
        assert not out_test.isnull().any().any()


# ---------------------------------------------------------------------------
# Unseen categories in test set → all-zero row (handle_unknown="ignore")
# ---------------------------------------------------------------------------

class TestUnseenCategories:
    def test_unseen_category_no_error(self):
        """Unknown category in test should not raise an error."""
        from src.data.preprocessor import BasicPreprocessor
        train = pd.DataFrame({"cat": ["a", "b", "a"]})
        test  = pd.DataFrame({"cat": ["c", "a"]})  # "c" is unseen

        pp = BasicPreprocessor()
        pp.fit(train)
        out = pp.transform(test)
        assert out is not None

    def test_unseen_category_all_zero_row(self):
        """Unseen category → all OHE columns for that row should be 0."""
        from src.data.preprocessor import BasicPreprocessor
        train = pd.DataFrame({"cat": ["a", "b", "a"]})
        test  = pd.DataFrame({"cat": ["c", "a"]})  # row 0 = unseen "c"

        pp = BasicPreprocessor()
        pp.fit(train)
        out = pp.transform(test)

        # Row 0 ("c") should have all zeros in OHE output
        ohe_row = out.iloc[0].values
        assert (ohe_row == 0).all(), f"Unseen category row should be all zeros, got {ohe_row}"


# ---------------------------------------------------------------------------
# Returns a DataFrame (not numpy array)
# ---------------------------------------------------------------------------

class TestOutputType:
    def test_returns_dataframe(self, mixed_df):
        from src.data.preprocessor import BasicPreprocessor
        out = BasicPreprocessor().fit_transform(mixed_df)
        assert isinstance(out, pd.DataFrame), f"Expected DataFrame, got {type(out)}"

    def test_returns_dataframe_transform(self, numeric_df):
        from src.data.preprocessor import BasicPreprocessor
        pp = BasicPreprocessor()
        pp.fit(numeric_df)
        out = pp.transform(numeric_df)
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Works on already-clean data (no NaN, no categoricals)
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_clean_numeric_passthrough_shape(self):
        from src.data.preprocessor import BasicPreprocessor
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        out = BasicPreprocessor().fit_transform(df)
        # Only numeric cols → same number of columns
        assert out.shape[1] == df.shape[1]
        assert out.shape[0] == df.shape[0]
