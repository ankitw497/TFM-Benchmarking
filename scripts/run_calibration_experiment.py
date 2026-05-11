#!/usr/bin/env python3
"""
run_calibration_experiment.py — Post-hoc calibration analysis.

Applies three calibration methods to each model's saved predictions:
  1. Temperature scaling  — single learnable parameter (best for neural models)
  2. Platt scaling        — sigmoid (LogisticRegression on logits)
  3. Isotonic regression  — non-parametric monotone (more flexible, can overfit)

Requires zero-shot benchmark to have been run with --save-probas:
    python scripts/run_benchmark.py --dataset <name> --save-probas

Usage:
    python scripts/run_calibration_experiment.py --dataset give_me_credit
    python scripts/run_calibration_experiment.py --dataset all
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_credit_dataset
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize_scalar
from scipy.special import logit, expit


# ---------------------------------------------------------------------------
# Calibration metrics and methods
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (weighted average over bins)."""
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 1e-7, 1 - 1e-7)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)


def temperature_scale(
    y_proba: np.ndarray,
    y_val: np.ndarray,
) -> tuple:
    """
    Find optimal temperature T that minimises NLL on y_val.
    scaled_prob = sigmoid(logit(prob) / T)

    Returns (calibrated_proba, T_optimal).
    """
    logits = logit(np.clip(y_proba, 1e-7, 1 - 1e-7))

    def neg_log_likelihood(log_T):
        T = np.exp(log_T)
        return log_loss(y_val, expit(logits / T))

    result = minimize_scalar(neg_log_likelihood, bounds=(-3.0, 3.0), method="bounded")
    T_opt = float(np.exp(result.x))
    return expit(logits / T_opt), T_opt


def platt_scale(
    y_proba_cal: np.ndarray,
    y_cal: np.ndarray,
    y_proba_test: np.ndarray,
) -> np.ndarray:
    """Platt scaling: fit sigmoid to (logit(proba), label) pairs."""
    from sklearn.linear_model import LogisticRegression
    logits_cal = logit(np.clip(y_proba_cal, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    logits_test = logit(np.clip(y_proba_test, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, max_iter=1000)  # High C = minimal regularisation
    lr.fit(logits_cal, y_cal)
    return lr.predict_proba(logits_test)[:, 1]


def isotonic_scale(
    y_proba_cal: np.ndarray,
    y_cal: np.ndarray,
    y_proba_test: np.ndarray,
) -> np.ndarray:
    """Isotonic regression calibration (non-parametric, monotone)."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_proba_cal, y_cal)
    return ir.predict(y_proba_test)


def metrics_dict(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute ECE, Brier score, and Log-Loss."""
    y_proba = np.clip(y_proba, 1e-7, 1 - 1e-7)
    return {
        "ece": compute_ece(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_calibration_experiment(
    dataset_name: str, results_dir: Path
) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"  CALIBRATION EXPERIMENT: {dataset_name.upper()}")
    print(f"{'='*70}\n")

    probas_dir = results_dir / ".cache" / "probas"
    if not probas_dir.exists():
        print(f"  ERROR: No saved probabilities at {probas_dir}")
        print("  Run first:")
        print(f"    python scripts/run_benchmark.py --dataset {dataset_name} --save-probas")
        return pd.DataFrame()

    proba_files = sorted(probas_dir.glob(f"probas_*_{dataset_name}_zero_shot.npz"))
    if not proba_files:
        print(f"  No zero-shot probas found for dataset: {dataset_name}")
        return pd.DataFrame()

    # We need a calibration set (use the val split)
    _, X_val, X_test, _, y_val, y_test = load_credit_dataset(
        dataset_name, return_val=True
    )
    y_val_np = y_val.values
    y_test_np = y_test.values

    all_results = []
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for proba_file in proba_files:
        # Extract model name from filename
        stem = proba_file.stem  # probas_<model>_<dataset>_zero_shot
        model_name = stem.replace(f"probas_", "").replace(f"_{dataset_name}_zero_shot", "")

        data = np.load(proba_file)
        y_proba_test = np.clip(data["y_proba"].astype(float), 1e-7, 1 - 1e-7)
        y_true_test = data["y_test"]

        print(f"  [{model_name}]")

        # Baseline (uncalibrated)
        base = metrics_dict(y_true_test, y_proba_test)
        row_base = {"model_name": model_name, "dataset": dataset_name,
                    "method": "uncalibrated", **base}
        all_results.append(row_base)
        print(
            f"    Uncalibrated : ECE={base['ece']:.4f}  "
            f"Brier={base['brier']:.4f}  LogLoss={base['log_loss']:.4f}"
        )

        # Load val-set probas for calibration methods that need them
        val_proba_file = probas_dir / f"probas_{model_name}_{dataset_name}_val.npz"
        if val_proba_file.exists():
            val_data = np.load(val_proba_file)
            y_proba_val = np.clip(val_data["y_proba"].astype(float), 1e-7, 1 - 1e-7)
        else:
            # Fall back to using test set as calibration set (slight optimism)
            y_proba_val = y_proba_test
            y_val_np = y_true_test

        # 1. Temperature scaling
        try:
            cal_temp, T_opt = temperature_scale(y_proba_test, y_val_np)
            m = metrics_dict(y_true_test, cal_temp)
            all_results.append({
                "model_name": model_name, "dataset": dataset_name,
                "method": f"temperature (T={T_opt:.3f})", **m,
            })
            print(
                f"    Temperature  : ECE={m['ece']:.4f}  "
                f"Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  T={T_opt:.3f}"
            )
        except Exception as e:
            print(f"    Temperature scaling failed: {e}")

        # 2. Platt scaling
        try:
            cal_platt = platt_scale(y_proba_val, y_val_np, y_proba_test)
            m = metrics_dict(y_true_test, cal_platt)
            all_results.append({
                "model_name": model_name, "dataset": dataset_name,
                "method": "platt_scaling", **m,
            })
            print(
                f"    Platt scaling: ECE={m['ece']:.4f}  "
                f"Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}"
            )
        except Exception as e:
            print(f"    Platt scaling failed: {e}")

        # 3. Isotonic regression
        try:
            cal_iso = isotonic_scale(y_proba_val, y_val_np, y_proba_test)
            m = metrics_dict(y_true_test, cal_iso)
            all_results.append({
                "model_name": model_name, "dataset": dataset_name,
                "method": "isotonic", **m,
            })
            print(
                f"    Isotonic     : ECE={m['ece']:.4f}  "
                f"Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}"
            )
        except Exception as e:
            print(f"    Isotonic regression failed: {e}")

        print()

    df = pd.DataFrame(all_results)
    output_path = results_dir / f"calibration_{dataset_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"  Results saved: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="TFM Calibration Experiment")
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
        run_calibration_experiment(dataset, results_dir)

    print("\nCalibration experiment complete.")


if __name__ == "__main__":
    main()
