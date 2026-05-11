# Sprint v1 — PRD: Make TFM-Benchmark Installable and Usable

## Sprint Overview

The codebase has solid internals — 14 model wrappers, metrics, visualization, and scripts —
but is not yet usable as a package. Users must clone the repo, understand internal import paths,
and write their own orchestration code. This sprint turns it into a proper, pip-installable Python
package with a clean high-level API that works in under 5 minutes for both practitioners and
researchers — with support for bundled datasets and bring-your-own data.

## Goals

- `pip install tfm-benchmark` works and gives users `from tfm_benchmark import Benchmarker, run_benchmark`
- A user can benchmark 10+ models on their own CSV in 5 lines of code
- Bundled datasets (German Credit, Taiwan Credit, synthetic) work out of the box without Kaggle credentials
- Both class-style (`Benchmarker`) and functional-style (`run_benchmark`) APIs are available
- At least one runnable example in `examples/` validates the full pipeline end-to-end
- Tests pass in CI without optional heavy dependencies (TabPFN, TabICL, etc.)

## User Stories

- As a **practitioner**, I want to point the benchmarker at my CSV file and get a leaderboard comparing 10+ models, so I can pick the best model for my tabular data without reading internal code.
- As a **researcher**, I want to call `run_benchmark(X_train, y_train, X_test, y_test)` and get back a structured results DataFrame, so I can integrate benchmarking into my own analysis pipeline.
- As a **new user**, I want to run a working example from the `examples/` directory to verify my installation and understand the API.
- As a **developer**, I want `import tfm_benchmark` to expose all public classes and functions, so I don't need to navigate the `src/` directory structure.

## Technical Architecture

**Current state (broken):**
```
src/
  data/loader.py          # imports as: from src.data.loader import ...
  models/tabpfn_wrapper.py
  evaluation/metrics.py
  visualization/plots.py
  finetuning/trainer.py
```
All `__init__.py` files are empty. Package installs as `tfm-benchmark` but only exposes `src.*` paths.

**Target state (this sprint):**
```
tfm_benchmark/              ← NEW: thin public API package
  __init__.py               ← exposes Benchmarker, run_benchmark, load_dataset, list_models
  benchmarker.py            ← Benchmarker class (high-level, class-based API)
  api.py                    ← run_benchmark() functional API
  datasets.py               ← load_dataset(): bundled + BYO CSV/DataFrame support

src/
  __init__.py               ← updated: re-exports key public symbols
  data/__init__.py          ← updated: exposes load_credit_dataset, _generate_synthetic_credit
  models/__init__.py        ← updated: exposes all wrapper classes + MODEL_REGISTRY dict
  evaluation/__init__.py    ← updated: exposes compute_all_metrics, aggregate_cv_results
  visualization/__init__.py ← updated: exposes plot_leaderboard, generate_all_plots

examples/
  01_quick_start.py         ← NEW: 5-line demo with bundled dataset
  02_custom_data.py         ← NEW: BYO CSV demo with auto-split

tests/
  test_core.py              ← updated: fix references to now-exported functions
  test_benchmarker.py       ← NEW: tests for Benchmarker + run_benchmark API
```

**Data flow — BYO data path:**
```
User provides CSV / DataFrame
        ↓
load_dataset(path_or_df, target='label', test_size=0.2)
        ↓
Benchmarker.fit_evaluate(X_train, y_train, X_test, y_test, models='auto')
        ↓  (internally iterates model wrappers, catches ImportError gracefully)
BenchmarkResult list  →  DataFrame  →  plot_leaderboard()
```

**Package installation (pyproject.toml change):**
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "tfm_benchmark*"]   # add tfm_benchmark*

[project.scripts]
tfm-benchmark = "tfm_benchmark.cli:main"   # add CLI entry point
```

**Public API surface (`tfm_benchmark/__init__.py`):**
```python
from tfm_benchmark import (
    Benchmarker,        # class-based API
    run_benchmark,      # functional API
    load_dataset,       # data loading (bundled + BYO)
    list_models,        # returns list of available model names
    list_datasets,      # returns list of bundled dataset names
)
```

## Out of Scope (v2+)

- Implementing missing `src/` files referenced in README but not yet needed by the new API:
  `preprocessor.py`, `splitter.py`, `timing.py`, `memory.py`, `statistical_tests.py`,
  `leaderboard.py`, `report.py`
- Fine-tuning experiments (complex GPU dependency — deferred to v2)
- Scaling experiments CLI
- Hyperparameter optimization (Optuna)
- Multi-dataset cross-benchmark reports
- Calibration / ensemble / missing-value experiment scripts (they exist but are untested)
- PyPI publishing (out of scope until tests pass)

## Dependencies

- All existing `src/` code must remain importable at `src.*` paths (scripts + notebooks use these)
- The new `tfm_benchmark.*` package sits on top of `src.*` — no circular imports
- Heavy optional deps (tabpfn, tabicl, torch, xgboost, etc.) remain optional — all wrappers
  already catch `ImportError` gracefully; the Benchmarker must do the same
- Python 3.10+, pandas, numpy, scikit-learn (already in core deps)
