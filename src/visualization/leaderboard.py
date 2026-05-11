"""
src.visualization.leaderboard — leaderboard table generation.

Thin wrapper over src.evaluation.metrics.create_comparison_table so
callers can import from a visualization-facing module without knowing
where the logic lives.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def generate_leaderboard_table(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    sort_by: str = "auc_roc",
    ascending: bool = False,
) -> pd.DataFrame:
    """Generate a sorted leaderboard DataFrame from model results.

    Parameters
    ----------
    results : dict
        Mapping of ``model_name -> {metric_name: value}``.
    metrics : list[str], optional
        Metrics to include as columns.  ``None`` includes all.
    sort_by : str
        Column to sort by (default: ``"auc_roc"``).
    ascending : bool
        Sort direction (default: ``False`` → best first).

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by *sort_by*.
    """
    from src.evaluation.metrics import create_comparison_table
    return create_comparison_table(
        results,
        metrics=metrics,
        sort_by=sort_by,
        ascending=ascending,
    )
