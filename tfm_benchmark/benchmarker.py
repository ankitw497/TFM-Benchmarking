"""
tfm_benchmark.benchmarker — high-level class-based API.

Usage
-----
    from tfm_benchmark import Benchmarker

    # Class-based (fluent) API
    b = Benchmarker(models="auto")
    results = b.fit_evaluate(X_train, y_train, X_test, y_test)
    b.plot_leaderboard()
    b.save_results("results/run.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.models import MODEL_REGISTRY, BaseModelWrapper, BenchmarkResult


class Benchmarker:
    """
    High-level benchmarking interface for comparing tabular models.

    Parameters
    ----------
    models : "auto" | list[str] | list[BaseModelWrapper]
        - ``"auto"``: try every model in MODEL_REGISTRY; silently skip models
          whose optional library is not installed.
        - list of string keys (e.g. ``["random_forest", "xgboost"]``): only
          those models are benchmarked.  Unknown keys raise ``ValueError``
          immediately.
        - list of ``BaseModelWrapper`` instances: used directly.
    dataset_name : str
        Label written into results (default: ``"custom"``).
    phase : str
        Phase label written into results (default: ``"zero_shot"``).
    verbose : bool
        Print per-model progress lines (default: ``True``).
    """

    def __init__(
        self,
        models: Union[str, List] = "auto",
        dataset_name: str = "custom",
        phase: str = "zero_shot",
        verbose: bool = True,
    ):
        self.dataset_name = dataset_name
        self.phase = phase
        self.verbose = verbose
        self.results_: Optional[pd.DataFrame] = None

        self.models = self._validate_models(models)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> pd.DataFrame:
        """
        Fit every model on the training data and evaluate on the test set.

        Accepts pandas DataFrames/Series **or** numpy arrays — numpy inputs
        are automatically converted to DataFrames with generic column names.

        Returns
        -------
        pd.DataFrame
            One row per model, sorted by AUC-ROC descending.
            Stored as ``self.results_`` for subsequent calls to
            ``plot_leaderboard()`` / ``save_results()``.
        """
        X_train, X_test, y_train, y_test = _coerce_to_pandas(
            X_train, X_test, y_train, y_test
        )

        # Resolve "auto" lazily here so registry is queried at run-time.
        # Returns list of (registry_key, wrapper) pairs so the key can be
        # stored alongside the wrapper's own display name in results.
        keyed_wrappers = self._resolve_wrappers()

        records = []
        for model_key, wrapper in keyed_wrappers:
            if self.verbose:
                print(f"  [{wrapper.name}]", end=" ", flush=True)

            result = wrapper.evaluate(
                X_train, y_train, X_test, y_test,
                dataset_name=self.dataset_name,
                phase=self.phase,
                model_key=model_key,
            )

            if self.verbose:
                if result.success:
                    print(f"AUC={result.auc_roc:.4f}  t={result.total_time:.1f}s")
                else:
                    print(f"SKIP  {result.error_message[:60]}")

            records.append(result.to_dict())

        df = pd.DataFrame(records)

        # Sort successful rows by AUC desc; failed rows go to bottom
        if not df.empty and "auc_roc" in df.columns:
            df_ok = df[df["success"]].sort_values("auc_roc", ascending=False)
            df_fail = df[~df["success"]]
            df = pd.concat([df_ok, df_fail], ignore_index=True)

        self.results_ = df
        return df

    def plot_leaderboard(
        self,
        metric: str = "auc_roc",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Render a horizontal bar chart ranking models by *metric*.

        Parameters
        ----------
        metric : str
            Column name to rank by (default: ``"auc_roc"``).
        title : str, optional
            Chart title.  Defaults to ``"Leaderboard — <dataset_name>"``.
        save_path : str, optional
            File path to save the figure (PNG/PDF/SVG).
        show : bool
            Call ``plt.show()`` after rendering (default: ``True``).
            Set to ``False`` in headless / test environments.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_results("plot_leaderboard")

        from src.visualization.plots import plot_leaderboard as _plot

        if title is None:
            title = f"Leaderboard — {self.dataset_name}"

        fig = _plot(
            self.results_,
            metric=metric,
            title=title,
            save_path=save_path,
        )

        if show:
            import matplotlib.pyplot as plt
            plt.show()

        return fig

    def save_results(self, path: Union[str, Path]) -> Path:
        """
        Write the results DataFrame to a CSV file.

        Parent directories are created automatically.

        Returns the resolved Path to the saved file.
        """
        self._require_results("save_results")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.results_.to_csv(out, index=False)

        if self.verbose:
            print(f"Results saved → {out}")

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_models(self, models):
        """
        Validate the ``models`` constructor argument and return it normalised.

        - ``"auto"`` → stored as the string ``"auto"`` (resolved lazily).
        - list[str] → each key checked against MODEL_REGISTRY; raises
          ``ValueError`` for unknown keys.
        - list[BaseModelWrapper] → returned as-is.
        - Mixed list (strings + instances) is also supported.
        """
        if models == "auto":
            return "auto"

        if not isinstance(models, list):
            raise TypeError(
                f"'models' must be 'auto', a list of string keys, or a list of "
                f"BaseModelWrapper instances. Got: {type(models)!r}"
            )

        # Validate string keys; leave instances alone
        bad_keys = [
            m for m in models
            if isinstance(m, str) and m not in MODEL_REGISTRY
        ]
        if bad_keys:
            raise ValueError(
                f"Unknown model key(s): {bad_keys}. "
                f"Valid keys: {sorted(MODEL_REGISTRY.keys())}"
            )

        return models

    def _resolve_wrappers(self):
        """
        Turn ``self.models`` into a list of ``(registry_key, BaseModelWrapper)``
        pairs, trying to import each factory and skipping quietly on
        ``ImportError``.

        Returns
        -------
        list of (str, BaseModelWrapper)
            Each element is ``(model_key, wrapper)`` so the registry key is
            carried alongside the wrapper and can be stored in results.
        """
        if isinstance(self.models, list):
            keyed = []
            for m in self.models:
                if isinstance(m, BaseModelWrapper):
                    # Instance passed directly — use the wrapper's name as key
                    keyed.append((m.name, m))
                else:
                    wrapper = _try_instantiate(m)
                    if wrapper is not None:
                        keyed.append((m, wrapper))
                    elif self.verbose:
                        print(f"  [{m}] SKIP (optional dependency not installed)")
            return keyed

        # "auto" mode: try every registry entry, skip on ImportError
        keyed = []
        for key in MODEL_REGISTRY:
            wrapper = _try_instantiate(key)
            if wrapper is not None:
                keyed.append((key, wrapper))
        return keyed

    def _require_results(self, method_name: str) -> None:
        if self.results_ is None:
            raise RuntimeError(
                f"No results available. Call fit_evaluate() before {method_name}()."
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _try_instantiate(key: str) -> Optional[BaseModelWrapper]:
    """Call the MODEL_REGISTRY factory; return None on ImportError."""
    try:
        return MODEL_REGISTRY[key]()
    except ImportError:
        return None


def _coerce_to_pandas(X_train, X_test, y_train, y_test):
    """Convert numpy arrays to DataFrames/Series, leave pandas objects alone."""
    if isinstance(X_train, np.ndarray):
        cols = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train, name="target")
        y_test = pd.Series(y_test, name="target")
    return X_train, X_test, y_train, y_test
