"""
Quick Start — benchmark on bundled German Credit data.

Requires only:
    pip install -e ".[gbdt]"

No Kaggle credentials or GPU needed.

Run:
    python examples/01_quick_start.py
"""

import matplotlib
matplotlib.use("Agg")  # headless — remove if you want an interactive window

from pathlib import Path

from tfm_benchmark import load_dataset, run_benchmark

# ── 1. Load bundled data ────────────────────────────────────────────────────
print("Loading German Credit dataset...")
X_train, X_test, y_train, y_test = load_dataset("german_credit")
print(f"  train: {X_train.shape}  test: {X_test.shape}")

# ── 2. Run benchmark ─────────────────────────────────────────────────────────
print("\nBenchmarking models...")
results = run_benchmark(
    X_train, y_train, X_test, y_test,
    models=["random_forest", "logistic_regression"],
    dataset_name="german_credit",
    verbose=True,
)

# ── 3. Print leaderboard ─────────────────────────────────────────────────────
print("\n── Leaderboard ──────────────────────────────────────────")
display_cols = [c for c in ["model_name", "auc_roc", "accuracy", "fit_time"] if c in results.columns]
print(results[display_cols].to_string(index=False))

# ── 4. Save leaderboard chart ────────────────────────────────────────────────
from tfm_benchmark import Benchmarker

b = Benchmarker(models=["random_forest", "logistic_regression"],
                dataset_name="german_credit", verbose=False)
b.fit_evaluate(X_train, y_train, X_test, y_test)

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
png_path = out_dir / "quick_start_leaderboard.png"
b.plot_leaderboard(save_path=str(png_path), show=False)
print(f"\nLeaderboard chart saved → {png_path}")

# ── 5. Save results CSV ──────────────────────────────────────────────────────
csv_path = out_dir / "quick_start_results.csv"
b.save_results(str(csv_path))
