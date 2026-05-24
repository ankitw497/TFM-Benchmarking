"""
examples/03_multi_dataset.py — Cross-dataset benchmark demo using run_benchmark_suite().

Benchmarks two sklearn models across two datasets (German Credit + a synthetic dataset),
prints a grouped leaderboard per dataset and an overall average-AUC summary row,
then saves results to results/multi_dataset_results.csv.

Usage
-----
    python examples/03_multi_dataset.py

Requirements
------------
Core install is sufficient (no GPU / no Kaggle account needed for 'synthetic'):

    pip install -e "."

For German Credit data (downloaded from UCI on first run):

    pip install -e ".[dev]"   # already includes ucimlrepo in core deps

"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running from any directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
import pandas as pd

from tfm_benchmark import run_benchmark_suite, load_dataset

# ── Configuration ────────────────────────────────────────────────────────────

MODELS = ["random_forest", "logistic_regression"]
RESULTS_PATH = ROOT / "results" / "multi_dataset_results.csv"


# ── Dataset definitions ──────────────────────────────────────────────────────

def build_datasets():
    """Return a list of dataset entries for run_benchmark_suite()."""
    datasets = []

    # Dataset 1: Synthetic (always available, no download needed)
    datasets.append("synthetic")

    # Dataset 2: German Credit (UCI download — may be slow on first run)
    try:
        datasets.append("german_credit")
    except Exception as exc:
        warnings.warn(f"Skipping German Credit: {exc}")

    return datasets


# ── Leaderboard formatting ───────────────────────────────────────────────────

def print_grouped_leaderboard(results: pd.DataFrame) -> None:
    """Print a leaderboard grouped by dataset_name, then overall averages."""
    success = results[results["success"]].copy()
    if success.empty:
        print("No successful results to display.")
        return

    metric_cols = ["auc_roc", "accuracy", "f1_macro", "fit_time"]
    available = [c for c in metric_cols if c in success.columns]

    print("\n" + "=" * 70)
    print(f"{'BENCHMARK RESULTS':^70}")
    print("=" * 70)

    for dataset_name in sorted(success["dataset_name"].unique()):
        subset = success[success["dataset_name"] == dataset_name].sort_values(
            "auc_roc", ascending=False
        )
        print(f"\n── Dataset: {dataset_name} ──")
        display_cols = ["model_name", "model_key"] + available
        display_cols = [c for c in display_cols if c in subset.columns]
        print(subset[display_cols].to_string(index=False))

    # Overall average AUC per model
    print("\n── Overall Average AUC-ROC per Model ──")
    avg = (
        success.groupby("model_key")["auc_roc"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg.columns = ["model_key", "avg_auc_roc"]
    print(avg.to_string(index=False))
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("tfm-benchmark — Multi-Dataset Suite Example")
    print(f"Models : {MODELS}")

    datasets = build_datasets()
    if not datasets:
        print("No datasets available. Exiting.")
        sys.exit(1)

    print(f"Datasets: {datasets}\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = run_benchmark_suite(
            datasets=datasets,
            models=MODELS,
            verbose=True,
            preprocessing=False,
        )

    for w in caught:
        print(f"[WARNING] {w.message}", file=sys.stderr)

    if results.empty:
        print("No results returned. Check warnings above.")
        sys.exit(1)

    print_grouped_leaderboard(results)

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
