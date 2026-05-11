"""
Unified dataset loader for credit scoring benchmarks.
Downloads, caches, and prepares credit datasets with consistent splits.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Optional, Dict, Any

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_credit_dataset(
    name: str = "give_me_credit",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
    return_val: bool = False,
) -> Tuple:
    """
    Load a credit dataset with consistent train/val/test splits.

    Parameters
    ----------
    name : str
        One of: 'give_me_credit', 'german_credit', 'taiwan_credit'
    test_size : float
        Fraction for test set
    val_size : float
        Fraction for validation set (from remaining after test split)
    random_state : int
        Seed for reproducibility
    max_rows : int, optional
        Subsample to this many rows (for scaling experiments)
    return_val : bool
        If True, return (X_train, X_val, X_test, y_train, y_val, y_test)

    Returns
    -------
    tuple of DataFrames / Series
    """
    loaders = {
        "give_me_credit": _load_give_me_credit,
        "german_credit": _load_german_credit,
        "taiwan_credit": _load_taiwan_credit,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")

    X, y, metadata = loaders[name]()

    # Subsample if requested (stratified)
    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, stratify=y, random_state=random_state
        )

    # Split: first separate test, then split remaining into train/val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if return_val:
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=relative_val_size,
            stratify=y_trainval,
            random_state=random_state,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_trainval, X_test, y_trainval, y_test


def get_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
):
    """Yield stratified K-Fold train/test indices."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        yield fold_idx, train_idx, test_idx


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Return metadata about a dataset without loading it."""
    info = {
        "give_me_credit": {
            "n_samples": 150_000,
            "n_features": 10,
            "target": "SeriousDlqin2yrs",
            "positive_rate": 0.067,
            "has_missing": True,
            "has_categorical": False,
            "source": "Kaggle",
            "url": "https://www.kaggle.com/c/GiveMeSomeCredit",
            "description": "Predict probability of financial distress in next 2 years",
        },
        "german_credit": {
            "n_samples": 1_000,
            "n_features": 20,
            "target": "class",
            "positive_rate": 0.30,
            "has_missing": False,
            "has_categorical": True,
            "source": "UCI",
            "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            "description": "Classify credit applicants as good or bad risk",
        },
        "taiwan_credit": {
            "n_samples": 30_000,
            "n_features": 23,
            "target": "default.payment.next.month",
            "positive_rate": 0.221,
            "has_missing": False,
            "has_categorical": True,
            "source": "UCI",
            "url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "description": "Predict default payment in Taiwan",
        },
    }
    return info.get(name, {})


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def _load_give_me_credit() -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Load 'Give Me Some Credit' dataset from Kaggle.

    If not cached locally, attempts to download via Kaggle API.
    Fallback: generate a synthetic version for testing.
    """
    cache_path = DATA_DIR / "give_me_credit"
    csv_path = cache_path / "cs-training.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
    else:
        # Try Kaggle API download
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            import kaggle
            kaggle.api.competition_download_file(
                "GiveMeSomeCredit", "cs-training.csv", path=str(cache_path)
            )
            df = pd.read_csv(csv_path, index_col=0)
        except (Exception, SystemExit) as e:
            # kaggle>=2.0 calls exit(1) on missing credentials (raises SystemExit,
            # a BaseException subclass not caught by bare `except Exception`).
            print(f"⚠️  Could not download from Kaggle: {e}")
            print("   Generating synthetic credit data for testing...")
            df = _generate_synthetic_credit(n=150_000, seed=42)

    target_col = "SeriousDlqin2yrs"
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    metadata = {"name": "give_me_credit", "task": "binary_classification"}
    return X, y, metadata


def _load_german_credit() -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Load German Credit dataset from UCI."""
    cache_path = DATA_DIR / "german_credit"
    csv_path = cache_path / "german_credit.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=144)
            X_uci = dataset.data.features
            y_uci = dataset.data.targets
            df = pd.concat([X_uci, y_uci], axis=1)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"⚠️  Could not download German Credit: {e}")
            df = _generate_synthetic_credit(n=1_000, n_features=20, seed=43)

    target_col = df.columns[-1]
    y = df[target_col].astype(int)
    # Remap: 1=good(0), 2=bad(1) in UCI encoding
    if y.nunique() == 2 and set(y.unique()) == {1, 2}:
        y = (y == 2).astype(int)
    X = df.drop(columns=[target_col])

    metadata = {"name": "german_credit", "task": "binary_classification"}
    return X, y, metadata


def _load_taiwan_credit() -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Load Taiwan Credit Default dataset from UCI."""
    cache_path = DATA_DIR / "taiwan_credit"
    csv_path = cache_path / "taiwan_credit.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=350)
            X_uci = dataset.data.features
            y_uci = dataset.data.targets
            df = pd.concat([X_uci, y_uci], axis=1)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"⚠️  Could not download Taiwan Credit: {e}")
            df = _generate_synthetic_credit(n=30_000, n_features=23, seed=44)

    target_col = "default.payment.next.month" if "default.payment.next.month" in df.columns else df.columns[-1]
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    metadata = {"name": "taiwan_credit", "task": "binary_classification"}
    return X, y, metadata


def _generate_synthetic_credit(
    n: int = 10_000,
    n_features: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic credit-like data as fallback."""
    rng = np.random.RandomState(seed)

    data = {}
    # Target: ~7% positive rate
    y = rng.binomial(1, 0.07, size=n)

    feature_names = [
        "RevolvingUtilization", "Age", "Num30-59DaysPastDue",
        "DebtRatio", "MonthlyIncome", "NumOpenCreditLines",
        "Num90DaysPastDue", "NumRealEstateLoans", "Num60-89DaysPastDue",
        "NumDependents",
    ]

    for i in range(min(n_features, len(feature_names))):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        base = rng.randn(n) + y * 0.3  # slight signal
        if i == 4:  # MonthlyIncome: inject ~20% missing
            mask = rng.random(n) < 0.20
            base[mask] = np.nan
            base = np.abs(base) * 5000
        elif i == 9:  # NumDependents: inject ~2.5% missing
            mask = rng.random(n) < 0.025
            base[mask] = np.nan
            base = np.abs(base).astype(float).round()
        elif i in [2, 6, 8]:  # Count features
            base = np.abs(base).round().clip(0, 10)
        elif i == 0:  # Utilization
            base = np.abs(base).clip(0, 2)
        elif i == 3:  # DebtRatio
            base = np.abs(base).clip(0, 5)
        data[name] = base

    for i in range(len(feature_names), n_features):
        data[f"feature_{i}"] = rng.randn(n)

    data["SeriousDlqin2yrs"] = y
    return pd.DataFrame(data)
