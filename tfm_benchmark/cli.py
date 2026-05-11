"""
tfm-benchmark CLI — entry point declared in pyproject.toml [project.scripts].

Usage:
    tfm-benchmark --help
    tfm-benchmark list-models
    tfm-benchmark list-datasets
    tfm-benchmark run --data <path_or_dataset> --target <col> [--models MODEL ...] [--output DIR]
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Static registries (expanded by Tasks 2 & 5 when wrappers are wired up)
# ---------------------------------------------------------------------------

_BUNDLED_DATASETS = [
    ("german_credit", "1 000 rows · 20 features · UCI · binary classification"),
    ("taiwan_credit", "30 000 rows · 23 features · UCI · binary classification"),
    ("synthetic", "Synthetic credit data — no download required"),
]

_KNOWN_MODELS = [
    ("tabpfn_v1",       "TabPFN v1",             "pip install tabpfn>=2.0"),
    ("tabpfn_v2",       "TabPFN v2",             "pip install tabpfn>=2.0"),
    ("tabpfn_v2_5",     "TabPFN v2.5",           "pip install tabpfn>=2.0  (+ HuggingFace login)"),
    ("tabpfn_v2_5_real","TabPFN v2.5 (real)",    "pip install tabpfn>=2.0  (+ HuggingFace login)"),
    ("tabicl_v2",       "TabICL v2",             "pip install tabicl>=0.2"),
    ("tabicl_v1_1",     "TabICL v1.1",           "pip install tabicl>=0.2"),
    ("mitra",           "Mitra (AutoGluon)",     "pip install autogluon.tabular>=1.4"),
    ("tabdpt",          "TabDPT",                "pip install git+https://github.com/layer6ai-labs/TabDPT.git"),
    ("tabnet",          "TabNet",                "pip install pytorch-tabnet torch>=2.1"),
    ("ft_transformer",  "FT-Transformer",        "pip install rtdl"),
    ("xgboost",         "XGBoost",               "pip install xgboost>=2.0"),
    ("catboost",        "CatBoost",              "pip install catboost>=1.2"),
    ("lightgbm",        "LightGBM",              "pip install lightgbm>=4.0"),
    ("random_forest",   "Random Forest",         "included in base install"),
    ("logistic_regression", "Logistic Regression", "included in base install"),
]


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_list_models(_args: argparse.Namespace) -> None:
    from tfm_benchmark.api import list_models as _list_models

    # Build lookup from static table for display names / install hints
    _display = {key: (display, note) for key, display, note in _KNOWN_MODELS}

    print("Available models (✅ = installed, ❌ = needs optional dep):\n")
    for key in _list_models():
        installed = _check_model_installed(key)
        status = "✅" if installed else "❌"
        display_name, install_note = _display.get(key, (key, "see docs"))
        line = f"  {status}  {display_name:<28}  [{key}]"
        if not installed:
            line += f"\n       install: {install_note}"
        print(line)
    print()


def _cmd_list_datasets(_args: argparse.Namespace) -> None:
    print("Bundled datasets (usable without Kaggle credentials):\n")
    for name, description in _BUNDLED_DATASETS:
        print(f"  • {name:<20}  {description}")
    print()
    print("  • give_me_credit        150 000 rows · 10 features · Kaggle (requires API key)")
    print()
    print("Pass any of these names to --data, or supply a CSV file path.")
    print()


def _cmd_run(args: argparse.Namespace) -> None:
    try:
        from tfm_benchmark.api import run_benchmark
        from tfm_benchmark.datasets import load_dataset
    except ImportError as exc:
        print(f"Error: required modules not yet available — {exc}", file=sys.stderr)
        print("Run `pip install -e .` and try again.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data: {args.data}")
    X_train, X_test, y_train, y_test = load_dataset(
        args.data,
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        max_rows=args.max_rows,
    )
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows  |  Features: {X_train.shape[1]}\n")

    models = args.models if args.models else "auto"
    results = run_benchmark(X_train, y_train, X_test, y_test, models=models)

    print(results[["model_name", "auc_roc", "log_loss_val", "total_time", "success"]].to_string(index=False))

    if args.output:
        import pathlib
        out = pathlib.Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / f"results_{args.data if isinstance(args.data, str) else 'custom'}.csv"
        results.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_model_installed(model_key: str) -> bool:
    """Best-effort check whether a model's backing library is importable."""
    checks = {
        "tabpfn_v1": "tabpfn",
        "tabpfn_v2": "tabpfn",
        "tabpfn_v2_5": "tabpfn",
        "tabpfn_v2_5_real": "tabpfn",
        "tabicl_v2": "tabicl",
        "tabicl_v1_1": "tabicl",
        "mitra": "autogluon",
        "tabdpt": "tabdpt",
        "tabnet": "pytorch_tabnet",
        "ft_transformer": "rtdl",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "lightgbm": "lightgbm",
        "random_forest": "sklearn",
        "logistic_regression": "sklearn",
    }
    lib = checks.get(model_key)
    if lib is None:
        return False
    try:
        __import__(lib)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfm-benchmark",
        description="TFM-Bench: Benchmark Tabular Foundation Models vs. classical ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tfm-benchmark list-models\n"
            "  tfm-benchmark list-datasets\n"
            "  tfm-benchmark run --data german_credit --models xgboost random_forest\n"
            "  tfm-benchmark run --data my_data.csv --target label --output results/\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"tfm-benchmark 0.1.0")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    subparsers.add_parser("list-models", help="List all supported models and installation status")
    subparsers.add_parser("list-datasets", help="List bundled datasets")

    run_p = subparsers.add_parser("run", help="Run a benchmark and print leaderboard")
    run_p.add_argument("--data", required=True, metavar="DATASET_OR_CSV",
                       help="Bundled dataset name (e.g. german_credit) or path to a CSV file")
    run_p.add_argument("--target", default=None, metavar="COLUMN",
                       help="Target column name (required for CSV input)")
    run_p.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                       help="Model keys to benchmark (default: all installed). "
                            "Run list-models for valid keys.")
    run_p.add_argument("--output", default=None, metavar="DIR",
                       help="Directory to save results CSV (optional)")
    run_p.add_argument("--test-size", type=float, default=0.2, metavar="FLOAT",
                       help="Fraction of data to use as test set (default: 0.2)")
    run_p.add_argument("--max-rows", type=int, default=None, metavar="N",
                       help="Cap total rows before splitting (useful for quick tests)")
    run_p.add_argument("--seed", type=int, default=42, metavar="INT",
                       help="Random seed for reproducibility (default: 42)")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "list-models": _cmd_list_models,
        "list-datasets": _cmd_list_datasets,
        "run": _cmd_run,
    }
    dispatch[args.command](args)
    sys.exit(0)


if __name__ == "__main__":
    main()
