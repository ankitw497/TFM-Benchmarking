"""
tfm_benchmark.api — functional entry points.

These are thin wrappers over Benchmarker / MODEL_REGISTRY so users can call
a single function without instantiating any class.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from src.models import MODEL_REGISTRY


def list_models() -> List[str]:
    """Return all model keys registered in MODEL_REGISTRY.

    Lists every supported model, including those whose optional dependency
    is not currently installed.  Use this to discover valid string keys for
    ``run_benchmark(models=[...])``.

    Returns
    -------
    list[str]
        Sorted list of model key strings (e.g. ``["logistic_regression",
        "random_forest", "tabicl_v2", "tabpfn_v2", ...]``).
    """
    return list(MODEL_REGISTRY.keys())


def run_benchmark(
    X_train,
    y_train,
    X_test,
    y_test,
    models: Union[str, List[str]] = "auto",
    dataset_name: str = "custom",
    random_state: int = 42,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Benchmark one or more models on a pre-split dataset.

    Thin functional wrapper over :class:`~tfm_benchmark.Benchmarker`.  Accepts
    either pandas DataFrames/Series or numpy arrays.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and labels.
    X_test, y_test : array-like
        Test features and labels.
    models : "auto" | list[str]
        - ``"auto"``: run every model in MODEL_REGISTRY, skipping those whose
          library is not installed.
        - list of string keys: only those models are benchmarked.  Unknown
          keys raise ``ValueError``.
    dataset_name : str
        Label written into the ``dataset_name`` column of the results
        (default: ``"custom"``).
    random_state : int
        Passed through to Benchmarker for reproducibility (default: 42).
    verbose : bool
        Print per-model progress lines (default: ``True``).
    **kwargs
        Extra keyword arguments forwarded to
        :class:`~tfm_benchmark.Benchmarker`.

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by AUC-ROC descending.  Always contains
        at minimum: ``model_name``, ``auc_roc``, ``success``,
        ``dataset_name``.

    Raises
    ------
    ValueError
        If any element of *models* is not a valid MODEL_REGISTRY key.

    Examples
    --------
    >>> from tfm_benchmark import run_benchmark
    >>> results = run_benchmark(X_train, y_train, X_test, y_test,
    ...                         models=["random_forest", "logistic_regression"])
    >>> print(results[["model_name", "auc_roc"]])
    """
    from tfm_benchmark.benchmarker import Benchmarker

    b = Benchmarker(
        models=models,
        dataset_name=dataset_name,
        verbose=verbose,
        **kwargs,
    )
    return b.fit_evaluate(X_train, y_train, X_test, y_test)
