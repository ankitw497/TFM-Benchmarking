#!/usr/bin/env python3
"""
run_missing_value_experiment.py — Measure AUC degradation vs MCAR missingness rate.

Injects Missing Completely At Random (MCAR) missingness at 5 rates and evaluates
models that handle missing values (natively or via internal imputation).

Usage:
    python scripts/run_missing_value_experiment.py --dataset german_credit
    python scripts/run_missing_value_experiment.py --dataset all
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_credit_dataset

MISSING_RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def inject_missingness(
    X: pd.DataFrame, rate: float, random_state: int = 42
) -> pd.DataFrame:
    """Inject MCAR missingness at the given rate across all numeric columns."""
    if rate == 0.0:
        return X.copy()
    rng = np.random.RandomState(random_state)
    X_out = X.copy()
    for col in X_out.select_dtypes(include="number").columns:
        mask = rng.random(len(X_out)) < rate
        X_out.loc[mask, col] = np.nan
    return X_out


def get_missing_capable_models():
    """Return models that support missing values (natively or via imputation)."""
    models = []

    # sklearn wrappers — handle missing via internal median imputation
    try:
        from src.models.sklearn_wrapper import RandomForestWrapper, LogisticRegressionWrapper
        models.append(RandomForestWrapper())
        models.append(LogisticRegressionWrapper())
    except ImportError:
        pass

    # GBDTs — native missing value support
    try:
        from src.models.gbdt_wrapper import XGBoostModel, LightGBMModel, CatBoostModel
        models.append(XGBoostModel(tuned=False))
        models.append(LightGBMModel())
        models.append(CatBoostModel(tuned=False))
    except ImportError:
        pass

    # TabPFN v2 — native missing support
    try:
        from src.models.tabpfn_wrapper import TabPFNModel
        models.append(TabPFNModel(version="v2", device="auto"))
    except ImportError:
        pass

    # TabICL v2 — native missing support
    try:
        from src.models.tabicl_wrapper import TabICLModel
        models.append(TabICLModel(version="v2"))
    except ImportError:
        pass

    return models


def run_missing_value_experiment(
    dataset_name: str, results_dir: Path
) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"  MISSING VALUE EXPERIMENT: {dataset_name.upper()}")
    print(f"{'='*70}\n")

    X_train, X_test, y_train, y_test = load_credit_dataset(dataset_name)
    models = get_missing_capable_models()

    if not models:
        print("  No models available. Install at least one model package.")
        return pd.DataFrame()

    print(f"  Dataset : {dataset_name}  ({len(X_train):,} train / {len(X_test):,} test)")
    print(f"  Models  : {[m.name for m in models]}")
    print(f"  Rates   : {MISSING_RATES}\n")

    all_results = []

    for rate in MISSING_RATES:
        print(f"  --- Missing rate: {rate:.0%} ---")
        X_train_m = inject_missingness(X_train, rate, random_state=42)
        X_test_m = inject_missingness(X_test, rate, random_state=43)

        for model in models:
            print(f"    [{model.name}]", end=" ", flush=True)
            result = model.evaluate(
                X_train_m, y_train, X_test_m, y_test,
                dataset_name, phase="missing_value",
            )
            row = result.to_dict()
            row["missing_rate"] = rate
            all_results.append(row)

            if result.success:
                print(f"AUC={result.auc_roc:.4f}")
            else:
                print(f"FAILED: {result.error_message[:50]}")

    df = pd.DataFrame(all_results)
    output_path = results_dir / f"missing_value_{dataset_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved: {output_path}")

    # Summary pivot table
    successful = df[df["success"]]
    if not successful.empty:
        print("\n  AUC by missing rate:")
        pivot = successful.pivot_table(
            index="model_name",
            columns="missing_rate",
            values="auc_roc",
            aggfunc="first",
        )
        print(pivot.round(4).to_string())

    return df


def main():
    parser = argparse.ArgumentParser(description="TFM Missing Value Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german_credit",
        choices=["give_me_credit", "german_credit", "taiwan_credit", "all"],
    )
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        ["give_me_credit", "german_credit", "taiwan_credit"]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset in datasets:
        run_missing_value_experiment(dataset, results_dir)

    print("\nMissing value experiment complete.")


if __name__ == "__main__":
    main()
