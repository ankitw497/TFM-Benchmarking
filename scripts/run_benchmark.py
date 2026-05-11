#!/usr/bin/env python3
"""
run_benchmark.py — Main CLI for running TFM benchmarks.

Usage:
    python scripts/run_benchmark.py --dataset give_me_credit --phase zero_shot
    python scripts/run_benchmark.py --dataset all --phase zero_shot --save-probas
    python scripts/run_benchmark.py --dataset german_credit --phase all --no-cache
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_credit_dataset, get_dataset_info
from src.models.base import BaseModelWrapper, BenchmarkResult


# ---------------------------------------------------------------------------
# Result cache — avoids re-running expensive models on crash/resume
# ---------------------------------------------------------------------------

class ResultCache:
    """
    File-based result cache. Each evaluation is stored as a JSON file
    keyed by (model_name, dataset_name, phase, n_train_rows).

    Use --no-cache to disable and force full re-evaluation.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model_name: str, dataset_name: str, phase: str, n_train: int) -> str:
        raw = f"{model_name}|{dataset_name}|{phase}|{n_train}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(
        self, model_name: str, dataset_name: str, phase: str, n_train: int
    ) -> Optional[BenchmarkResult]:
        key = self._key(model_name, dataset_name, phase, n_train)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            # BenchmarkResult fields only — strip extras from old runs
            valid_fields = BenchmarkResult.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return BenchmarkResult(**filtered)
        except Exception:
            return None  # Corrupted cache entry; re-run

    def put(self, result: BenchmarkResult) -> None:
        key = self._key(
            result.model_name, result.dataset_name, result.phase, result.n_train
        )
        path = self.cache_dir / f"{key}.json"
        d = result.to_dict()
        # Ensure warnings is serializable
        d["warnings"] = list(d.get("warnings", []))
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Probabilities saver
# ---------------------------------------------------------------------------

def _save_probas(
    y_proba: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    dataset_name: str,
    phase: str,
    probas_dir: Path,
) -> None:
    """Save raw probability arrays for ensemble/calibration experiments."""
    probas_dir.mkdir(parents=True, exist_ok=True)
    fname = f"probas_{model_name}_{dataset_name}_{phase}.npz"
    np.savez(probas_dir / fname, y_proba=y_proba, y_test=y_test)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_zero_shot_models() -> List[BaseModelWrapper]:
    """Instantiate all models for zero-shot benchmarking."""
    models = []

    # ── Tabular Foundation Models ─────────────────────────────────────────
    try:
        from src.models.tabpfn_wrapper import TabPFNModel
        models.append(TabPFNModel(version="v1", device="cpu"))   # v1: CPU only, 1K row limit
        models.append(TabPFNModel(version="v2", device="auto"))
        models.append(TabPFNModel(version="v2.5", device="auto"))
        models.append(TabPFNModel(version="v2.5-real", device="auto"))
    except ImportError as e:
        print(f"  Warning: TabPFN not available: {e}")

    try:
        from src.models.tabicl_wrapper import TabICLModel
        models.append(TabICLModel(version="v2"))
        models.append(TabICLModel(version="v1.1"))
    except ImportError as e:
        print(f"  Warning: TabICL not available: {e}")

    try:
        from src.models.mitra_wrapper import MitraModel
        models.append(MitraModel())
    except ImportError as e:
        print(f"  Warning: Mitra/AutoGluon not available: {e}")

    try:
        from src.models.tabdpt_wrapper import TabDPTModel
        models.append(TabDPTModel())
    except ImportError as e:
        print(f"  Warning: TabDPT not available (install from GitHub): {e}")

    # ── Deep Learning ─────────────────────────────────────────────────────
    try:
        from src.models.tabnet_wrapper import TabNetModel
        models.append(TabNetModel(pretrain=False))
    except ImportError as e:
        print(f"  Warning: TabNet not available: {e}")

    try:
        from src.models.ft_transformer_wrapper import FTTransformerModel
        models.append(FTTransformerModel())
    except ImportError as e:
        print(f"  Warning: FT-Transformer not available (install rtdl): {e}")

    # ── GBDT Baselines ────────────────────────────────────────────────────
    try:
        from src.models.gbdt_wrapper import XGBoostModel, CatBoostModel, LightGBMModel
        models.append(XGBoostModel(tuned=False))
        models.append(XGBoostModel(tuned=True))
        models.append(CatBoostModel(tuned=False))
        models.append(CatBoostModel(tuned=True))
        models.append(LightGBMModel())
    except ImportError as e:
        print(f"  Warning: GBDT models not available: {e}")

    # ── sklearn Baselines ─────────────────────────────────────────────────
    try:
        from src.models.sklearn_wrapper import RandomForestWrapper, LogisticRegressionWrapper
        models.append(RandomForestWrapper())
        models.append(LogisticRegressionWrapper())
    except ImportError as e:
        print(f"  Warning: sklearn wrappers not available: {e}")

    return models


def get_finetune_models() -> List[BaseModelWrapper]:
    """Instantiate models for fine-tuning experiments."""
    models = []

    try:
        from src.models.tabpfn_wrapper import FinetunedTabPFNModel
        models.append(FinetunedTabPFNModel(version="v2", epochs=30, lr=1e-5))
        models.append(FinetunedTabPFNModel(version="v2.5", epochs=30, lr=1e-5))
    except ImportError as e:
        print(f"  Warning: TabPFN fine-tuning not available: {e}")

    return models


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def run_zero_shot(
    dataset_name: str,
    results_dir: Path,
    cache: Optional[ResultCache],
    save_probas: bool = False,
) -> pd.DataFrame:
    """Run zero-shot evaluation on all registered models."""
    print(f"\n{'='*70}")
    print(f"  ZERO-SHOT BENCHMARK: {dataset_name.upper()}")
    print(f"{'='*70}\n")

    X_train, X_test, y_train, y_test = load_credit_dataset(dataset_name)
    print(f"  Dataset : {dataset_name}")
    print(f"  Train   : {len(X_train):,} rows  x  {X_train.shape[1]} features")
    print(f"  Test    : {len(X_test):,} rows")
    print(f"  Pos rate: {y_train.mean():.3f}\n")

    probas_dir = (results_dir / ".cache" / "probas") if save_probas else None
    models = get_zero_shot_models()
    results = []

    for model in models:
        lim = model.get_limitations()
        print(f"  [{model.name}]", end=" ", flush=True)

        # Check hard limits
        if not model.can_handle_dataset(len(X_train), X_train.shape[1]):
            print(
                f"SKIPPED  (limit: max_rows={lim.max_rows}, "
                f"max_features={lim.max_features})"
            )
            result = BenchmarkResult(
                model_name=model.name,
                dataset_name=dataset_name,
                phase="zero_shot",
                success=False,
                error_message="Exceeds model limits",
                n_train=len(X_train),
                n_test=len(X_test),
                n_features=X_train.shape[1],
            )
            results.append(result)
            continue

        # Serve from cache if available
        if cache:
            cached = cache.get(model.name, dataset_name, "zero_shot", len(X_train))
            if cached:
                print(f"[cached]  AUC={cached.auc_roc:.4f}")
                results.append(cached)
                continue

        result = model.evaluate(
            X_train, y_train, X_test, y_test, dataset_name, "zero_shot"
        )

        if result.success:
            print(
                f"AUC={result.auc_roc:.4f}  LogLoss={result.log_loss_val:.4f}  "
                f"Time={result.total_time:.1f}s  License={lim.license[:30]}"
            )
        else:
            print(f"FAILED  {result.error_message[:60]}")

        if cache and result.success:
            cache.put(result)

        # Optionally save raw probabilities for ensemble / calibration experiments
        if save_probas and result.success and probas_dir:
            try:
                y_proba = model.predict_proba(X_test)
                y_proba_1d = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                _save_probas(
                    y_proba_1d, y_test.values,
                    model.name, dataset_name, "zero_shot",
                    probas_dir,
                )
            except Exception:
                pass  # Don't fail the benchmark just because proba saving failed

        results.append(result)

    df = pd.DataFrame([r.to_dict() for r in results])
    output_path = results_dir / f"zero_shot_{dataset_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved: {output_path}")
    _print_leaderboard(df)
    return df


def run_finetuning(
    dataset_name: str,
    results_dir: Path,
    cache: Optional[ResultCache],
    save_probas: bool = False,
) -> pd.DataFrame:
    """Run fine-tuning experiments."""
    print(f"\n{'='*70}")
    print(f"  FINE-TUNING BENCHMARK: {dataset_name.upper()}")
    print(f"{'='*70}\n")

    X_train, X_val, X_test, y_train, y_val, y_test = load_credit_dataset(
        dataset_name, return_val=True
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}\n")

    models = get_finetune_models()
    results = []

    for model in models:
        lim = model.get_limitations()
        print(f"  [{model.name}]", end=" ", flush=True)

        if cache:
            cached = cache.get(model.name, dataset_name, "finetuned", len(X_train))
            if cached:
                print(f"[cached]  AUC={cached.auc_roc:.4f}")
                results.append(cached)
                continue

        result = model.evaluate(
            X_train, y_train, X_test, y_test, dataset_name, "finetuned"
        )

        if result.success:
            print(f"AUC={result.auc_roc:.4f}  Time={result.total_time:.1f}s")
        else:
            print(f"FAILED  {result.error_message[:60]}")

        if cache and result.success:
            cache.put(result)

        results.append(result)

    df = pd.DataFrame([r.to_dict() for r in results])
    output_path = results_dir / f"finetuned_{dataset_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved: {output_path}")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_leaderboard(df: pd.DataFrame) -> None:
    """Print sorted leaderboard of successful results."""
    successful = df[df["success"] == True].copy()
    if successful.empty:
        print("\n  No successful results to display.")
        return

    successful = successful.sort_values("auc_roc", ascending=False)
    print(f"\n  {'─'*78}")
    print(
        f"  {'Rank':<5} {'Model':<28} {'AUC-ROC':<10} "
        f"{'LogLoss':<10} {'Time(s)':<10} {'Mem(MB)':<8}"
    )
    print(f"  {'─'*78}")
    for rank, (_, row) in enumerate(successful.iterrows(), 1):
        print(
            f"  {rank:<5} {str(row['model_name']):<28} {row['auc_roc']:<10.4f} "
            f"{row['log_loss_val']:<10.4f} {row['total_time']:<10.2f} "
            f"{row['peak_memory_mb']:<8.0f}"
        )
    print(f"  {'─'*78}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TFM Benchmark Runner — Tabular Foundation Models on Credit Data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="give_me_credit",
        choices=["give_me_credit", "german_credit", "taiwan_credit", "all"],
        help="Dataset to benchmark. Use 'all' to run all three.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "finetune", "all"],
        help="Benchmark phase to run.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to write CSV results and figures.",
    )
    parser.add_argument(
        "--save-probas",
        action="store_true",
        help=(
            "Save raw probability arrays to results/.cache/probas/. "
            "Required for run_ensemble_experiment.py and run_calibration_experiment.py."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching and re-run all models from scratch.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cache = (
        None
        if args.no_cache
        else ResultCache(results_dir / ".cache" / "results")
    )

    if cache:
        print("Result caching enabled. Use --no-cache to force full re-evaluation.")
    if args.save_probas:
        print("Probability saving enabled (results/.cache/probas/).")

    datasets = (
        ["give_me_credit", "german_credit", "taiwan_credit"]
        if args.dataset == "all"
        else [args.dataset]
    )

    start = time.perf_counter()
    for dataset in datasets:
        if args.phase in ("zero_shot", "all"):
            run_zero_shot(dataset, results_dir, cache, save_probas=args.save_probas)
        if args.phase in ("finetune", "all"):
            run_finetuning(dataset, results_dir, cache, save_probas=args.save_probas)

    elapsed = time.perf_counter() - start
    print(f"\nBenchmark complete in {elapsed:.1f}s.  Results in: {results_dir}/")


if __name__ == "__main__":
    main()
