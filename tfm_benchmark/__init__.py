"""
tfm_benchmark — Benchmarking Tabular Foundation Models.

High-level public API. Full internals live in src/.
"""

__version__ = "0.1.0"

# Populated progressively as sub-modules are implemented (Tasks 3-5).
# Import guards keep the package importable even if optional dependencies
# (tabpfn, xgboost, torch, etc.) are not installed.

from tfm_benchmark import cli  # noqa: F401 — make `from tfm_benchmark import cli` work
from tfm_benchmark.datasets import load_dataset, list_datasets

__all__ = [
    "__version__",
    "cli",
    "load_dataset",
    "list_datasets",
]
