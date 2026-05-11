#!/usr/bin/env python3
"""
run_ensemble_experiment.py — Measure AUC gains from ensembling TFMs with GBDTs.

Requires zero-shot benchmark to have been run with --save-probas:
    python scripts/run_benchmark.py --dataset <name> --save-probas

Three ensemble strategies:
  1. Simple average         — (proba_A + proba_B) / 2
  2. Rank average           — average of rank-transformed probabilities
  3. Stacking               — LogisticRegression meta-learner on held-out train probas

Usage:
    python scripts/run_ensemble_experiment.py --dataset give_me_credit
    python scripts/run_ensemble_experiment.py --dataset all
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_credit_dataset
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ENSEMBLE_PAIRS = [
    ["TabPFN-v2", "XGBoost-Tuned"],
    ["TabICL-v2", "CatBoost-Tuned"],
    ["TabPFN-v2", "TabICL-v2"],
    ["TabPFN-v2", "LightGBM-Default"],
    ["TabPFN-v2", "XGBoost-Tuned", "TabICL-v2"],  # Triple ensemble
]


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------

def ensemble_average(probas: list) -> np.ndarray:
    """Arithmetic mean of probability arrays."""
    return np.mean(probas, axis=0)


def ensemble_rank_average(probas: list) -> np.ndarray:
    """
    Rank-based ensemble — less sensitive to miscalibrated models.
    Each model's probabilities are rank-normalised before averaging.
    """
    from scipy.stats import rankdata
    ranked = [rankdata(p) / len(p) for p in probas]
    return np.mean(ranked, axis=0)


def ensemble_stacking(
    test_probas: list,
    train_probas: list,
    y_train: np.ndarray,
) -> np.ndarray:
    """
    Stacking: fit a LogisticRegression meta-learner on held-out train probas,
    then predict on test probas.
    """
    X_meta_train = np.column_stack(train_probas)
    X_meta_test = np.column_stack(test_probas)
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(X_meta_train, y_train)
    return meta.predict_proba(X_meta_test)[:, 1]


# ---------------------------------------------------------------------------
# Probability loading helpers
# ---------------------------------------------------------------------------

def load_probas(
    probas_dir: Path, model_name: str, dataset_name: str, phase: str = "zero_shot"
) -> tuple:
    """Load saved probabilities. Returns (y_proba, y_test) or (None, None)."""
    fname = probas_dir / f"probas_{model_name}_{dataset_name}_{phase}.npz"
    if not fname.exists():
        return None, None
    data = np.load(fname)
    return data["y_proba"], data["y_test"]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_ensemble_experiment(
    dataset_name: str, results_dir: Path
) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE EXPERIMENT: {dataset_name.upper()}")
    print(f"{'='*70}\n")

    probas_dir = results_dir / ".cache" / "probas"
    if not probas_dir.exists():
        print(f"  ERROR: No saved probabilities found at {probas_dir}")
        print("  Run first:")
        print(f"    python scripts/run_benchmark.py --dataset {dataset_name} --save-probas")
        return pd.DataFrame()

    _, X_test, y_train, y_test = load_credit_dataset(dataset_name)
    y_test_np = y_test.values

    all_results = []

    for pair in ENSEMBLE_PAIRS:
        pair_name = " + ".join(pair)

        # Load probas for each model in the pair
        pair_test_probas = []
        pair_train_probas = []
        missing = []

        for model_name in pair:
            y_proba_test, _ = load_probas(probas_dir, model_name, dataset_name)
            if y_proba_test is None:
                missing.append(model_name)
            else:
                pair_test_probas.append(y_proba_test)
                # Train probas (optional — only needed for stacking)
                y_proba_train, _ = load_probas(
                    probas_dir, model_name, dataset_name, phase="train"
                )
                pair_train_probas.append(y_proba_train)

        if missing:
            print(f"  Skipping [{pair_name}] — missing probas for: {missing}")
            continue

        print(f"  Ensemble: {pair_name}")

        # Individual model AUCs (baseline)
        for i, model_name in enumerate(pair):
            auc = roc_auc_score(y_test_np, pair_test_probas[i])
            all_results.append({
                "ensemble_pair": pair_name,
                "method": "individual",
                "model_name": model_name,
                "auc_roc": auc,
                "dataset": dataset_name,
                "n_models": 1,
            })
            print(f"    {model_name} (individual): AUC={auc:.4f}")

        # Simple average
        avg_proba = ensemble_average(pair_test_probas)
        avg_auc = roc_auc_score(y_test_np, avg_proba)
        all_results.append({
            "ensemble_pair": pair_name,
            "method": "average",
            "model_name": pair_name,
            "auc_roc": avg_auc,
            "dataset": dataset_name,
            "n_models": len(pair),
        })
        print(f"    Average ensemble           : AUC={avg_auc:.4f}")

        # Rank average
        rank_proba = ensemble_rank_average(pair_test_probas)
        rank_auc = roc_auc_score(y_test_np, rank_proba)
        all_results.append({
            "ensemble_pair": pair_name,
            "method": "rank_average",
            "model_name": pair_name,
            "auc_roc": rank_auc,
            "dataset": dataset_name,
            "n_models": len(pair),
        })
        print(f"    Rank-avg ensemble          : AUC={rank_auc:.4f}")

        # Stacking (requires train probas)
        if all(p is not None for p in pair_train_probas):
            try:
                stack_proba = ensemble_stacking(
                    pair_test_probas, pair_train_probas, y_train.values
                )
                stack_auc = roc_auc_score(y_test_np, stack_proba)
                all_results.append({
                    "ensemble_pair": pair_name,
                    "method": "stacking",
                    "model_name": pair_name,
                    "auc_roc": stack_auc,
                    "dataset": dataset_name,
                    "n_models": len(pair),
                })
                print(f"    Stacking ensemble          : AUC={stack_auc:.4f}")
            except Exception as e:
                print(f"    Stacking failed: {e}")
        else:
            print(
                "    Stacking skipped (no train probas — re-run with "
                "--save-probas on train split)"
            )

        print()

    if not all_results:
        print("  No ensemble results produced.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    output_path = results_dir / f"ensemble_{dataset_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"  Results saved: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="TFM Ensemble Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="give_me_credit",
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
        run_ensemble_experiment(dataset, results_dir)

    print("\nEnsemble experiment complete.")


if __name__ == "__main__":
    main()
