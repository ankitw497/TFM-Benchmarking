"""
tfm_benchmark.api — functional entry points.

These are thin wrappers over Benchmarker / MODEL_REGISTRY so users can call
a single function without instantiating any class.
"""

from __future__ import annotations

import traceback
import warnings
from typing import List, Sequence, Tuple, Union

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
    preprocessing: bool = False,
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
    preprocessing : bool
        When ``True``, apply :class:`~src.data.preprocessor.BasicPreprocessor`
        (median imputation + scaling + OHE) before passing data to any model.
        The preprocessor is fit on training data only (default: ``False``).
    **kwargs
        Extra keyword arguments forwarded to
        :class:`~tfm_benchmark.Benchmarker`.

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by AUC-ROC descending.  Always contains
        at minimum: ``model_name``, ``auc_roc``, ``success``,
        ``dataset_name``, ``preprocessing``.

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
        preprocessing=preprocessing,
        **kwargs,
    )
    return b.fit_evaluate(X_train, y_train, X_test, y_test)


# ---------------------------------------------------------------------------
# Multi-dataset suite
# ---------------------------------------------------------------------------

def run_benchmark_suite(
    datasets: Sequence,
    models: Union[str, List[str]] = "auto",
    verbose: bool = True,
    preprocessing: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Benchmark one or more models across multiple datasets.

    Each dataset can be specified as:
    - A **string** key (e.g. ``"german_credit"``, ``"synthetic"``) — loaded
      automatically via :func:`~tfm_benchmark.load_dataset`.
    - A **5-tuple** ``(name, X_train, y_train, X_test, y_test)`` for custom
      pre-split data.

    Errors from a single dataset do not abort the rest of the suite; they are
    caught, warned about, and the suite continues.

    Parameters
    ----------
    datasets : sequence of str | (name, X_train, y_train, X_test, y_test)
        Ordered list of datasets to benchmark on.
    models : "auto" | list[str]
        Model keys or ``"auto"`` — same semantics as :func:`run_benchmark`.
    verbose : bool
        Print per-dataset / per-model progress (default: ``True``).
    preprocessing : bool
        Apply :class:`~src.data.preprocessor.BasicPreprocessor` before each
        dataset's model evaluation (default: ``False``).
    **kwargs
        Extra keyword arguments forwarded to
        :class:`~tfm_benchmark.Benchmarker`.

    Returns
    -------
    pd.DataFrame
        All model rows from all datasets concatenated, with an additional
        ``dataset_name`` column that identifies which dataset each row came from.

    Examples
    --------
    >>> from tfm_benchmark import run_benchmark_suite
    >>> results = run_benchmark_suite(
    ...     datasets=["synthetic", ("my_data", X_tr, y_tr, X_te, y_te)],
    ...     models=["random_forest", "logistic_regression"],
    ... )
    >>> print(results.groupby("dataset_name")["auc_roc"].mean())
    """
    from tfm_benchmark.benchmarker import Benchmarker
    from tfm_benchmark.datasets import load_dataset

    all_frames: list[pd.DataFrame] = []

    for ds in datasets:
        # Resolve the dataset
        if isinstance(ds, str):
            dataset_name = ds
            try:
                X_train, X_test, y_train, y_test = load_dataset(ds)
            except Exception as exc:
                warnings.warn(
                    f"run_benchmark_suite: failed to load named dataset {ds!r}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
        elif isinstance(ds, (tuple, list)) and len(ds) == 5:
            dataset_name, X_train, y_train, X_test, y_test = ds
        else:
            warnings.warn(
                f"run_benchmark_suite: unrecognised dataset entry {ds!r}. "
                "Expected a string name or a (name, X_train, y_train, X_test, y_test) tuple.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if verbose:
            print(f"\n=== Dataset: {dataset_name} ===")

        try:
            b = Benchmarker(
                models=models,
                dataset_name=dataset_name,
                verbose=verbose,
                preprocessing=preprocessing,
                **kwargs,
            )
            df = b.fit_evaluate(X_train, y_train, X_test, y_test)
            # Ensure dataset_name column is present (Benchmarker stores it in results)
            if "dataset_name" not in df.columns:
                df = df.copy()
                df["dataset_name"] = dataset_name
            all_frames.append(df)
        except Exception as exc:
            warnings.warn(
                f"run_benchmark_suite: error on dataset {dataset_name!r}: {exc}\n"
                f"{traceback.format_exc()}",
                UserWarning,
                stacklevel=2,
            )

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)
