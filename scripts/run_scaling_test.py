#!/usr/bin/env python3
"""
run_scaling_test.py — Test how each model handles increasing dataset sizes.

Gracefully handles:
- Hard row limits (skips model at that size and beyond)
- GPU Out-of-Memory errors (marks failure, skips larger sizes)
- Per-model timeout (marks failure, continues)

Usage:
    python scripts/run_scaling_test.py --dataset give_me_credit
    python scripts/run_scaling_test.py --dataset give_me_credit --row-counts 500 1000 5000 10000
    python scripts/run_scaling_test.py --dataset all --timeout 300
"""

import argparse
import signal
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_credit_dataset
from src.models.base import BenchmarkResult
from scripts.run_benchmark import get_zero_shot_models

DEFAULT_ROW_COUNTS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 150_000]


# ---------------------------------------------------------------------------
# Timeout context manager (Unix-only; no-op on Windows)
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._supported = hasattr(signal, "SIGALRM")

    def __enter__(self):
        if self._supported and self.seconds > 0:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._supported:
            signal.alarm(0)
        return False

    @staticmethod
    def _handler(signum, frame):
        raise _TimeoutError("Model evaluation timed out")


# ---------------------------------------------------------------------------
# Scaling experiment
# ---------------------------------------------------------------------------

def _is_oom_error(e: Exception) -> bool:
    """Return True if the exception looks like an out-of-memory error."""
    msg = str(e).lower()
    return isinstance(e, MemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in msg
    )


def run_scaling_test(
    dataset_name: str,
    row_counts: list,
    results_dir: Path,
    timeout_seconds: int = 0,
) -> pd.DataFrame:
    """Test all models at increasing dataset sizes."""
    print(f"\n{'='*70}")
    print(f"  SCALING EXPERIMENT: {dataset_name.upper()}")
    print(f"  Row counts  : {row_counts}")
    if timeout_seconds > 0:
        print(f"  Per-model timeout: {timeout_seconds}s")
    print(f"{'='*70}\n")

    all_results = []
    models = get_zero_shot_models()

    # Track which models have already failed due to OOM — skip larger sizes
    oom_failed: set = set()

    for n_rows in row_counts:
        print(f"\n  --- {n_rows:,} rows ---")

        try:
            X_train, X_test, y_train, y_test = load_credit_dataset(
                dataset_name, max_rows=n_rows
            )
        except ValueError as e:
            print(f"  Cannot subsample to {n_rows}: {e}")
            continue

        actual_train = len(X_train)
        print(f"  Actual train: {actual_train:,}  |  Test: {len(X_test):,}")

        for model in models:
            lim = model.get_limitations()
            print(f"    [{model.name}]", end=" ", flush=True)

            # Skip if this model already hit OOM at a smaller size
            if model.name in oom_failed:
                print("SKIPPED (OOM at smaller size)")
                all_results.append(_make_skip_row(
                    model.name, n_rows, actual_train, X_train.shape[1],
                    "Skipped: OOM at smaller dataset size",
                ))
                continue

            # Skip hard row limits
            if lim.max_rows and actual_train > lim.max_rows:
                print(f"SKIPPED (hard limit: {lim.max_rows:,} rows)")
                all_results.append(_make_skip_row(
                    model.name, n_rows, actual_train, X_train.shape[1],
                    f"Exceeds max_rows={lim.max_rows}",
                ))
                continue

            # Warn on soft limits
            warn = ""
            if lim.recommended_max_rows and actual_train > lim.recommended_max_rows:
                warn = f" [above recommended {lim.recommended_max_rows:,}]"

            # Run with timeout and OOM protection
            try:
                with _Timeout(timeout_seconds):
                    result = model.evaluate(
                        X_train, y_train, X_test, y_test,
                        dataset_name, phase="scaling",
                    )
            except _TimeoutError:
                print(f"TIMEOUT ({timeout_seconds}s)")
                all_results.append(_make_skip_row(
                    model.name, n_rows, actual_train, X_train.shape[1],
                    f"Timeout after {timeout_seconds}s",
                ))
                continue
            except (MemoryError, RuntimeError) as e:
                if _is_oom_error(e):
                    oom_failed.add(model.name)
                    print(f"OOM — will skip at larger sizes")
                    all_results.append(_make_skip_row(
                        model.name, n_rows, actual_train, X_train.shape[1],
                        f"OOM: {str(e)[:80]}",
                    ))
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                raise

            if result.success:
                print(f"AUC={result.auc_roc:.4f}  Time={result.total_time:.1f}s{warn}")
            else:
                print(f"FAILED: {result.error_message[:50]}")

            row = result.to_dict()
            row["n_rows_requested"] = n_rows
            all_results.append(row)

    df = pd.DataFrame(all_results)
    output_path = results_dir / f"scaling_{dataset_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Scaling results saved: {output_path}")

    _print_scaling_summary(df)
    return df


def _make_skip_row(
    model_name: str,
    n_rows_requested: int,
    n_rows_actual: int,
    n_features: int,
    reason: str,
) -> dict:
    """Build a failure/skip result row for the scaling DataFrame."""
    return {
        "model_name": model_name,
        "n_rows_requested": n_rows_requested,
        "n_train": n_rows_actual,
        "n_features": n_features,
        "success": False,
        "error_message": reason,
        "auc_roc": None,
        "log_loss_val": None,
        "total_time": None,
        "peak_memory_mb": None,
        "phase": "scaling",
    }


def _print_scaling_summary(df: pd.DataFrame) -> None:
    """Print which models succeeded at which row counts."""
    print(f"\n  {'─'*70}")
    print(f"  SCALING SUMMARY  (AUC shown if succeeded, -- otherwise)")
    print(f"  {'─'*70}")

    if df.empty:
        print("  No results.")
        return

    row_counts = sorted(df["n_rows_requested"].dropna().unique().astype(int))
    models = df["model_name"].unique()

    header = f"  {'Model':<28}" + "".join(f"{n:>9,}" for n in row_counts)
    print(header)
    print(f"  {'─'*70}")

    for model in models:
        line = f"  {model:<28}"
        for n in row_counts:
            mask = (df["model_name"] == model) & (df["n_rows_requested"] == n)
            subset = df[mask]
            if subset.empty or not subset.iloc[0].get("success", False):
                line += f"{'--':>9}"
            else:
                auc = subset.iloc[0].get("auc_roc")
                line += f"{auc:>9.3f}" if auc is not None else f"{'--':>9}"
        print(line)

    print(f"  {'─'*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TFM Scaling Test")
    parser.add_argument(
        "--dataset",
        type=str,
        default="give_me_credit",
        choices=["give_me_credit", "german_credit", "taiwan_credit", "all"],
    )
    parser.add_argument(
        "--row-counts",
        type=int,
        nargs="+",
        default=DEFAULT_ROW_COUNTS,
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Max seconds per model per row count. 0 = no timeout (default).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        ["give_me_credit", "german_credit", "taiwan_credit"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset in datasets:
        run_scaling_test(dataset, args.row_counts, results_dir, args.timeout)

    print("\nScaling test complete.")


if __name__ == "__main__":
    main()
