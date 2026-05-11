"""
tfm_benchmark.datasets — unified data loading for benchmarking.

Supports:
  - Bundled datasets (german_credit, taiwan_credit, synthetic) — no credentials needed
  - Bring-Your-Own data:  a CSV file path  OR  a pandas DataFrame
  - give_me_credit: still routed through the existing Kaggle-backed loader

All entry points return the same four-tuple:
    (X_train, X_test, y_train, y_test)
as pandas DataFrames / Series with consistent dtypes and index resets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional

import pandas as pd

from sklearn.model_selection import train_test_split

# Re-used from src for bundled datasets
from src.data.loader import (
    load_credit_dataset as _load_credit_dataset,
    _generate_synthetic_credit,
)

# Public type alias
SplitTuple = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

# Datasets available without Kaggle credentials
_BUNDLED = ["german_credit", "taiwan_credit", "synthetic"]

# Datasets that still go through the Kaggle-backed path
_KAGGLE_DATASETS = ["give_me_credit"]


def list_datasets() -> list[str]:
    """Return names of bundled datasets usable without Kaggle credentials."""
    return list(_BUNDLED)


def load_dataset(
    source: Union[str, pd.DataFrame, Path],
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
) -> SplitTuple:
    """
    Load a dataset and return a stratified train/test split.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        One of:
        - A bundled dataset name: ``"german_credit"``, ``"taiwan_credit"``,
          ``"synthetic"``, ``"give_me_credit"``
        - An absolute or relative path to a CSV file (string or Path)
        - A pandas DataFrame containing both features and the target column
    target : str, optional
        Name of the target column.  Required when ``source`` is a DataFrame
        or a CSV file path.  Ignored for named bundled datasets.
    test_size : float
        Fraction of data to use as the test set (default: 0.2).
    random_state : int
        Random seed for reproducibility (default: 42).
    max_rows : int, optional
        Sub-sample to at most this many rows before splitting (stratified).

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames and Series

    Raises
    ------
    ValueError
        - Unknown dataset name
        - CSV/DataFrame source provided without ``target``
        - ``target`` column not present in the data
    FileNotFoundError
        CSV path does not exist
    """
    # ── Route to the right loader ─────────────────────────────────────────

    if isinstance(source, pd.DataFrame):
        return _from_dataframe(source, target, test_size, random_state, max_rows)

    source_str = str(source)

    # Named bundled/kaggle dataset
    if source_str in _BUNDLED or source_str in _KAGGLE_DATASETS:
        return _from_named_dataset(source_str, test_size, random_state, max_rows)

    # CSV file path (anything that looks like a file path)
    path = Path(source_str)
    if path.suffix.lower() == ".csv" or path.exists():
        return _from_csv(path, target, test_size, random_state, max_rows)

    raise ValueError(
        f"Unknown dataset or unrecognised source: {source_str!r}.\n"
        f"  • Bundled names: {_BUNDLED + _KAGGLE_DATASETS}\n"
        f"  • Or pass a path to a CSV file or a pandas DataFrame."
    )


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _from_named_dataset(
    name: str,
    test_size: float,
    random_state: int,
    max_rows: Optional[int],
) -> SplitTuple:
    """Load a named bundled dataset or the Kaggle-backed give_me_credit."""
    if name == "synthetic":
        n = min(max_rows, 10_000) if max_rows else 10_000
        df = _generate_synthetic_credit(n=n, seed=random_state)
        target_col = "SeriousDlqin2yrs"
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
        return _split(X, y, test_size=test_size, random_state=random_state, max_rows=None)

    # german_credit, taiwan_credit, give_me_credit → delegate to existing loader
    result = _load_credit_dataset(
        name=name,
        test_size=test_size,
        random_state=random_state,
        max_rows=max_rows,
        return_val=False,
    )
    X_train, X_test, y_train, y_test = result
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def _from_csv(
    path: Path,
    target: Optional[str],
    test_size: float,
    random_state: int,
    max_rows: Optional[int],
) -> SplitTuple:
    """Load a CSV file, split off the target column, and return train/test."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if target is None:
        raise ValueError(
            "A 'target' column name is required when loading from a CSV file.\n"
            "Example: load_dataset('data.csv', target='my_label_column')"
        )

    df = pd.read_csv(path)
    return _from_dataframe(df, target, test_size, random_state, max_rows)


def _from_dataframe(
    df: pd.DataFrame,
    target: Optional[str],
    test_size: float,
    random_state: int,
    max_rows: Optional[int],
) -> SplitTuple:
    """Split a DataFrame into features + target, then train/test."""
    if target is None:
        raise ValueError(
            "A 'target' column name is required when loading from a DataFrame.\n"
            "Example: load_dataset(df, target='my_label_column')"
        )

    if target not in df.columns:
        raise ValueError(
            f"Target column {target!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    return _split(X, y, test_size=test_size, random_state=random_state, max_rows=max_rows)


def _split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    max_rows: Optional[int],
) -> SplitTuple:
    """Stratified train/test split with optional row cap."""
    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y,
            train_size=max_rows,
            stratify=y,
            random_state=random_state,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
