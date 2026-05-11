"""
src.data.splitter — thin wrappers over sklearn's stratified split.

Provides a single consistent entry point for all train/test splitting
used across the benchmark so callers don't import sklearn directly.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with index reset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (used for stratification).
    test_size : float
        Proportion of rows to include in the test set (default: 0.2).
    random_state : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    (X_train, X_test, y_train, y_test)
        All four splits as DataFrames/Series with reset integer indexes.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
