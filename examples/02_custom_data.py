"""
Custom Data — benchmark on your own CSV (or a generated synthetic one).

Requires only:
    pip install -e ".[gbdt]"

No Kaggle credentials or GPU needed.

Run:
    python examples/02_custom_data.py
    python examples/02_custom_data.py --csv path/to/your_data.csv --target label_col
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tfm_benchmark import load_dataset, run_benchmark, list_models


def _generate_csv(path: Path, n: int = 300, seed: int = 0) -> None:
    """Write a simple synthetic binary-classification CSV."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age":        rng.randint(18, 80, n),
        "income":     rng.exponential(50_000, n).round(2),
        "balance":    rng.normal(5_000, 3_000, n).round(2),
        "loan_amount": rng.exponential(10_000, n).round(2),
        "num_accounts": rng.randint(1, 10, n),
        "default":    (rng.rand(n) > 0.75).astype(int),
    })
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark on custom data")
    parser.add_argument("--csv",    default=None, help="Path to CSV file (optional)")
    parser.add_argument("--target", default="default", help="Target column name")
    parser.add_argument("--models", nargs="+",
                        default=["random_forest", "logistic_regression"],
                        help="Model keys to benchmark")
    args = parser.parse_args()

    # ── 1. Prepare data ───────────────────────────────────────────────────────
    if args.csv:
        csv_path = args.csv
        print(f"Loading data from {csv_path} ...")
    else:
        # Generate a synthetic CSV so the example runs without any extra files
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv_path = tmp.name
        _generate_csv(Path(csv_path))
        print(f"Generated synthetic CSV → {csv_path}")

    X_train, X_test, y_train, y_test = load_dataset(csv_path, target=args.target)
    print(f"  train: {X_train.shape}  test: {X_test.shape}")

    # ── 2. Show available models ──────────────────────────────────────────────
    print(f"\nAll registered models: {list_models()}")

    # ── 3. Benchmark ──────────────────────────────────────────────────────────
    print(f"\nRunning benchmark with: {args.models}")
    results = run_benchmark(
        X_train, y_train, X_test, y_test,
        models=args.models,
        dataset_name=Path(csv_path).stem,
        verbose=True,
    )

    # ── 4. Print results ──────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────")
    display_cols = [c for c in ["model_name", "auc_roc", "accuracy", "fit_time"] if c in results.columns]
    print(results[display_cols].to_string(index=False))

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "custom_data_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
