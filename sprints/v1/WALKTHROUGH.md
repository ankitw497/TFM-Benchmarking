# Sprint v1 — Walkthrough

## Summary

Sprint v1 transformed a research codebase with solid internals (14 model wrappers, metrics,
visualization) but no usable public API into a proper pip-installable Python package. A new
`tfm_benchmark/` package layer was built on top of the existing `src/` internals, exposing
`Benchmarker`, `run_benchmark`, `load_dataset`, `list_models`, and `list_datasets` as
first-class citizens. Users can now benchmark 10+ tabular models on bundled or bring-your-own
data in 5 lines of Python, using either a fluent class API or a single function call — with
no Kaggle credentials, no GPU, and no knowledge of the internal `src.*` import paths.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User-facing layer                            │
│                                                                     │
│   from tfm_benchmark import run_benchmark, Benchmarker,            │
│                              load_dataset, list_models              │
│                                                                     │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │  tfm_benchmark/  │  │  tfm_benchmark/ │  │  tfm_benchmark/  │   │
│  │   api.py         │  │  benchmarker.py │  │  datasets.py     │   │
│  │  run_benchmark() │  │  Benchmarker    │  │  load_dataset()  │   │
│  │  list_models()   │  │  fit_evaluate() │  │  list_datasets() │   │
│  └────────┬─────────┘  └───────┬─────────┘  └───────┬──────────┘   │
│           │ thin wrapper        │ delegates          │ routes       │
└───────────┼─────────────────────┼────────────────────┼─────────────┘
            │                     │                    │
┌───────────▼─────────────────────▼────────────────────▼─────────────┐
│                        Internal src/ layer                          │
│                                                                     │
│  ┌─────────────────────┐       ┌────────────────────────────────┐   │
│  │  src/models/        │       │  src/data/loader.py            │   │
│  │  MODEL_REGISTRY     │       │  load_credit_dataset()         │   │
│  │  (17 lazy lambdas)  │       │  _generate_synthetic_credit()  │   │
│  │                     │       └────────────────────────────────┘   │
│  │  BaseModelWrapper   │       ┌────────────────────────────────┐   │
│  │  BenchmarkResult    │       │  src/evaluation/metrics.py     │   │
│  │  14 wrapper classes │       │  compute_all_metrics()         │   │
│  └─────────────────────┘       └────────────────────────────────┘   │
│                                ┌────────────────────────────────┐   │
│                                │  src/visualization/plots.py    │   │
│                                │  plot_leaderboard()            │   │
│                                └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         CLI layer                                │
│  tfm-benchmark list-models  → delegates to list_models()        │
│  tfm-benchmark list-datasets → delegates to list_datasets()     │
│  tfm-benchmark run --data german_credit --models random_forest  │
│                                   → delegates to run_benchmark() │
└──────────────────────────────────────────────────────────────────┘

Data flow (BYO CSV path):
  user CSV / DataFrame
       ↓  load_dataset(path, target="label")
  X_train, X_test, y_train, y_test  (stratified 80/20 split)
       ↓  run_benchmark(..., models=["random_forest", "xgboost"])
  Benchmarker._resolve_wrappers()  →  for each wrapper: wrapper.evaluate()
       ↓  BenchmarkResult.to_dict() × N
  pd.DataFrame  (sorted by AUC-ROC desc)
       ↓  plot_leaderboard() / save_results()
  PNG chart + CSV file
```

---

## Files Created / Modified

### `pyproject.toml`
**Purpose**: Package discovery and CLI entry-point registration.

**Key changes**:
- Added `"tfm_benchmark*"` to `[tool.setuptools.packages.find]` so the new
  package is included in `pip install -e .`
- Added `[project.scripts]` entry: `tfm-benchmark = "tfm_benchmark.cli:main"`

Before this change `pip install -e .` only installed `src.*`; the new
`tfm_benchmark` package was invisible to Python's import system.

---

### `tfm_benchmark/__init__.py`
**Purpose**: Single import surface for the entire public API.

```python
from tfm_benchmark.datasets   import load_dataset, list_datasets
from tfm_benchmark.benchmarker import Benchmarker
from tfm_benchmark.api         import run_benchmark, list_models

__all__ = ["__version__", "Benchmarker", "run_benchmark",
           "load_dataset", "list_datasets", "list_models"]
```

Every public symbol is re-exported here so users only ever need
`from tfm_benchmark import X` — they never touch `src.*` paths.

---

### `tfm_benchmark/datasets.py`
**Purpose**: Unified data loading that accepts a bundled dataset name, a CSV path, or a raw DataFrame — always returning the same `(X_train, X_test, y_train, y_test)` four-tuple.

**Key functions**:
- `list_datasets()` — returns `["german_credit", "taiwan_credit", "synthetic"]`
- `load_dataset(source, target, test_size, random_state, max_rows)` — routes to the right private loader
- `_from_named_dataset()` — delegates to `src.data.loader.load_credit_dataset()`
- `_from_csv()` — reads with `pd.read_csv()`, then calls `_from_dataframe()`
- `_from_dataframe()` — splits off the target column, calls `_split()`
- `_split()` — stratified `train_test_split` + `reset_index(drop=True)` on all four outputs

**How it works**:

The routing logic in `load_dataset()` inspects the type of `source` first: if
it's a `pd.DataFrame` it goes straight to `_from_dataframe`. If it's a string,
the function checks whether it matches a known bundled dataset name, then falls
back to treating it as a file path. This means the same call signature handles
all three BYO patterns without the caller needing to know which path runs.

Every loader ultimately calls `_split()`, which always calls
`reset_index(drop=True)` on all four outputs. This ensures the returned splits
always have a clean `RangeIndex(0, n)` regardless of how the source data was
indexed — important because downstream wrappers and tests rely on consistent
index alignment.

The `synthetic` dataset is generated on-the-fly via
`_generate_synthetic_credit()` from `src.data.loader` — no files or downloads
needed, making it useful for quick tests and CI.

---

### `tfm_benchmark/benchmarker.py`
**Purpose**: High-level class-based API that orchestrates the full benchmark pipeline.

**Key methods**:
- `__init__(models, dataset_name, phase, verbose)` — validates model keys against `MODEL_REGISTRY` immediately so bad keys fail fast, before any data is loaded
- `fit_evaluate(X_train, y_train, X_test, y_test)` — the core method: coerces numpy arrays to DataFrames, resolves model wrappers, calls `wrapper.evaluate()` on each, assembles a sorted DataFrame, stores it as `self.results_`
- `plot_leaderboard(metric, title, save_path, show)` — delegates to `src.visualization.plots.plot_leaderboard()`
- `save_results(path)` — writes `self.results_` to CSV; creates parent directories automatically
- `_validate_models(models)` — accepts `"auto"`, a list of string keys (checked against `MODEL_REGISTRY`), or a list of `BaseModelWrapper` instances
- `_resolve_wrappers()` — converts string keys to wrapper instances by calling `MODEL_REGISTRY[key]()`; silently skips keys whose optional library raises `ImportError`

**How it works**:

```python
# Inside fit_evaluate():
wrappers = self._resolve_wrappers()   # lazy import: ImportError → skip
for wrapper in wrappers:
    result = wrapper.evaluate(X_train, y_train, X_test, y_test, ...)
    records.append(result.to_dict())

df_ok   = df[df["success"]].sort_values("auc_roc", ascending=False)
df_fail = df[~df["success"]]
return pd.concat([df_ok, df_fail], ignore_index=True)
```

The key design decision is that failures are *rows in the result DataFrame*,
not exceptions. When a model's optional library is absent (e.g. `xgboost` not
installed), `_try_instantiate` returns `None` and the model is silently
skipped. When a model is instantiated but fails at fit/predict time, the
`BenchmarkResult` has `success=False` and `error_message` filled in — still a
row, just at the bottom after sorting. This means `auto` mode never crashes
regardless of which optional deps are installed.

Results are always sorted successful rows by AUC-ROC descending, failed rows
appended at the bottom.

---

### `tfm_benchmark/api.py`
**Purpose**: Thin functional wrapper over `Benchmarker` for one-liner usage.

**Key functions**:
- `list_models()` — returns `list(MODEL_REGISTRY.keys())`, always reflecting the current registry
- `run_benchmark(X_train, y_train, X_test, y_test, models, dataset_name, ...)` — instantiates a `Benchmarker` and calls `fit_evaluate()`; all keyword args pass through

```python
def run_benchmark(X_train, y_train, X_test, y_test,
                  models="auto", dataset_name="custom", verbose=True, **kwargs):
    from tfm_benchmark.benchmarker import Benchmarker
    b = Benchmarker(models=models, dataset_name=dataset_name,
                    verbose=verbose, **kwargs)
    return b.fit_evaluate(X_train, y_train, X_test, y_test)
```

`list_models()` deliberately returns *all* registry keys, including those for
uninstalled optional dependencies. This is by design: users call `list_models()`
to discover what's possible, not just what's currently installed.

---

### `tfm_benchmark/cli.py`
**Purpose**: `argparse`-based CLI registered as the `tfm-benchmark` shell command.

**Subcommands**:
- `list-models` — calls `list_models()`, checks each key's backing library with `_check_model_installed()`, prints `✅` / `❌` with install instructions
- `list-datasets` — calls `list_datasets()`, prints descriptions
- `run` — calls `load_dataset()` then `run_benchmark()`, prints results table, saves CSV if `--output` given

**Key flags for `run`**:
- `--data DATASET_OR_CSV` (required) — bundled name or path to CSV
- `--target COLUMN` — target column for CSV/DataFrame input
- `--models MODEL ...` — one or more registry keys; defaults to `auto`
- `--output DIR` — directory to save the results CSV
- `--test-size`, `--max-rows`, `--seed` — data options

`list-models` delegates to `list_models()` (dynamic, always in sync with
`MODEL_REGISTRY`) rather than a hardcoded list, so new models added to the
registry appear automatically in CLI output.

---

### `src/models/__init__.py`
**Purpose**: Central model registry — maps stable string keys to lazy factory lambdas.

**`MODEL_REGISTRY`** (17 entries):

```python
MODEL_REGISTRY = {
    "tabpfn_v1":    lambda: _make_tabpfn("v1", device="cpu"),
    "tabpfn_v2":    lambda: _make_tabpfn("v2"),
    ...
    "xgboost":      lambda: _make_xgboost(tuned=False),
    "random_forest": lambda: _make_random_forest(),
    "logistic_regression": lambda: _make_logistic_regression(),
}
```

Every factory function wraps the real import inside a module-level function
(not a lambda directly calling an import), so that `import tfm_benchmark`
never fails due to missing optional dependencies. The import only runs when
a factory is **called** — at benchmark time, not at package load time.

---

### `src/data/__init__.py`, `src/evaluation/__init__.py`, `src/visualization/__init__.py`
**Purpose**: Populate previously-empty `__init__.py` files with clean re-exports.

Before this sprint all three files were empty, forcing users to write
`from src.data.loader import load_credit_dataset` (internal path). Now:

```python
# src/data/__init__.py
from .loader import load_credit_dataset, get_cv_splits, get_dataset_info
```

Existing scripts and notebooks that used `src.data.loader` directly still work;
the new exports are additive.

---

### `src/data/loader.py`
**Purpose**: Existing bundled-dataset loader; one bug fixed.

**Bug fixed**: `kaggle>=2.0` calls `exit(1)` (not `raise ImportError`) when
credentials are missing. `exit(1)` raises `SystemExit` which is a `BaseException`,
not caught by `except Exception`. Fixed by changing the catch to
`except (Exception, SystemExit)` so missing Kaggle credentials don't crash the
entire process.

---

### `src/data/preprocessor.py` *(new)*
**Purpose**: `BasicPreprocessor` — sklearn-compatible pass-through stub.

Implements `fit()`, `transform()`, `fit_transform()` following the sklearn
transformer convention. Currently returns data unchanged (no-op). Designed to
be extended in a later sprint with imputation, scaling, and encoding. Having
the module present now means code can `from src.data.preprocessor import BasicPreprocessor`
without `ImportError` even before the real logic is built.

---

### `src/data/splitter.py` *(new)*
**Purpose**: `stratified_split()` — thin wrapper so callers don't import sklearn directly.

```python
def stratified_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True))
```

---

### `src/evaluation/timing.py` *(new)*
**Purpose**: `TimingContext` — wall-clock timing via `time.perf_counter`.

```python
with TimingContext() as tc:
    model.fit(X_train, y_train)
print(f"fit took {tc.elapsed:.3f}s")
```

---

### `src/evaluation/memory.py` *(new)*
**Purpose**: `get_peak_memory()` — returns RSS memory in MB via psutil, or `0.0` if psutil is absent.

Extracted from the private `_get_peak_memory()` in `src/models/base.py` so it
has a stable, importable public path. The original private function in `base.py`
remains (it's used by `BaseModelWrapper.evaluate()` internally).

---

### `src/visualization/leaderboard.py` *(new)*
**Purpose**: `generate_leaderboard_table()` — delegates to `src.evaluation.metrics.create_comparison_table()`.

Provides a visualization-facing import path so notebooks and scripts can
`from src.visualization.leaderboard import generate_leaderboard_table` without
knowing that the actual logic lives in the evaluation layer.

---

### `src/finetuning/trainer.py`
**Purpose**: Existing fine-tuning trainer; two semgrep findings suppressed.

The trainer uses `pickle` for checkpoint save/load (necessary for arbitrary
ML model objects that can't be serialized to JSON). Added inline
`# nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle`
suppressions to acknowledge the known risk. No behavioural change.

---

### `examples/01_quick_start.py`
**Purpose**: End-to-end demo on the bundled German Credit dataset.

1. Calls `load_dataset("german_credit")` — no credentials needed
2. Calls `run_benchmark(...)` with `random_forest` + `logistic_regression`
3. Prints a leaderboard table
4. Uses `Benchmarker` directly to call `plot_leaderboard(save_path=...)` — saves a PNG
5. Calls `save_results(...)` — saves a CSV to `results/`

Requires only `pip install -e ".[gbdt]"`. Uses `matplotlib.use("Agg")` so it
runs headless in CI without a display.

---

### `examples/02_custom_data.py`
**Purpose**: BYO-data demo that generates a synthetic CSV and benchmarks it.

Accepts `--csv`, `--target`, and `--models` CLI arguments. When run without
arguments, it generates a 300-row synthetic credit CSV to a temp file, loads
it via `load_dataset(csv_path, target="default")`, and runs `run_benchmark()`.
Demonstrates `list_models()` by printing all available keys before benchmarking.

---

### `README.md`
**Purpose**: Updated to lead with a working Quick Start for the new API.

A new section was prepended before the detailed research documentation:
- **Installation**: `pip install -e ".[gbdt]"` (no-GPU, base, all extras)
- **5-line example**: uses `from tfm_benchmark import ...` — verified runnable by a test
- **BYO data example**: CSV path and DataFrame variants
- **Class-based API example**: `Benchmarker` flow
- **Example scripts**: how to run `examples/01_quick_start.py`

The stale "Minimal Example" in the old Quick Start section (which used
`from src.data.loader import ...`) was replaced with the research pipeline
CLI invocations, preserving all detailed documentation below.

---

### Test files

| File | Tests | What it covers |
|------|-------|----------------|
| `tests/test_task1_package_structure.py` | 7 | `import tfm_benchmark` works; `__version__` is set; `pyproject.toml` includes both packages; CLI entry-point is wired |
| `tests/test_task2_init_exports.py` | 21 | All public symbols importable from `src.*` sub-packages; `MODEL_REGISTRY` has 17 keys; all factory lambdas call without crash |
| `tests/test_task3_datasets.py` | 27 | `load_dataset` with named dataset / CSV / DataFrame; `list_datasets`; error cases (missing target, bad path, unknown name) |
| `tests/test_task4_benchmarker.py` | 24 | `Benchmarker` construction, validation, `fit_evaluate`, `save_results`, `plot_leaderboard`; numpy input; AUC sort order |
| `tests/test_task5_api.py` | 25 | `run_benchmark` / `list_models` importable from `tfm_benchmark`; `__all__` surface; functional correctness vs `Benchmarker`; `verbose=False` produces no stdout |
| `tests/test_task6_examples.py` | 10 | Both example scripts exit 0 as subprocesses; produce expected output; README mentions both |
| `tests/test_benchmarker.py` | 19 | Integration tests for the full public API using sklearn-only models; no optional deps required |
| `tests/test_core.py` | 15 (13+2 skip) | Pre-existing core tests; GBDT tests fixed to use `pytest.importorskip("xgboost")` instead of `try/except ImportError` (which never fired because wrapper module imports succeed) |
| `tests/test_task8_readme.py` | 11 | README structure (Quick Start before Experimental Design, `[gbdt]` extra, new API in snippet); 5-line example actually runs as subprocess |
| `tests/test_task9_cli.py` | 22 | All CLI subcommands via subprocess; `--help`, `--version`; `list-models` covers all MODEL_REGISTRY keys; `run` saves CSV; invalid model exits non-zero |
| `tests/test_task10_stubs.py` | 17 | All five new `src/` stub modules importable and behave correctly |

**Total: 196 passed, 2 skipped** (xgboost and lightgbm tests skip when those libraries are not installed — correct behaviour, not a failure).

---

## Data Flow

### Functional path (`run_benchmark`)

```
run_benchmark(X_train, y_train, X_test, y_test, models=["random_forest"])
  → Benchmarker(models=["random_forest"], verbose=True)
      → _validate_models(["random_forest"])   # fails fast if key unknown
  → fit_evaluate(X_train, y_train, X_test, y_test)
      → _coerce_to_pandas(...)                # numpy arrays → DataFrames
      → _resolve_wrappers()
          → MODEL_REGISTRY["random_forest"]() # calls factory lambda
          → RandomForestWrapper instance
      → wrapper.evaluate(X_train, y_train, X_test, y_test, ...)
          → wrapper.fit(X_train, y_train)
          → y_prob = wrapper.predict_proba(X_test)
          → compute_all_metrics(y_test, y_prob)
          → BenchmarkResult(auc_roc=..., success=True, ...)
      → pd.DataFrame([result.to_dict()])
      → sort by auc_roc desc (successful rows), failed rows at bottom
  → return DataFrame
```

### BYO data path

```
load_dataset("my_data.csv", target="default")
  → _from_csv(Path("my_data.csv"), target="default", ...)
      → pd.read_csv("my_data.csv")
      → _from_dataframe(df, target="default", ...)
          → y = df["default"],  X = df.drop("default")
          → _split(X, y, test_size=0.2, ...)
              → train_test_split(..., stratify=y)
              → reset_index(drop=True) on all four
  → (X_train, X_test, y_train, y_test)
```

### CLI path

```
$ tfm-benchmark run --data german_credit --models random_forest --output results/
  → load_dataset("german_credit")         # bundled, no credentials
  → run_benchmark(X_train, y_train, X_test, y_test, models=["random_forest"])
  → print results DataFrame
  → results.to_csv("results/results_german_credit.csv")
```

---

## Security Measures

- **No pickle in public API**: The only pickle usage is in `src/finetuning/trainer.py`
  for checkpoint save/load, which is internal and documented with semgrep suppressions.
  The new `tfm_benchmark/` layer uses no serialization.
- **No shell injection in CLI**: All arguments are processed by `argparse` and passed
  to Python functions — no `subprocess.run(shell=True)` or `os.system()` calls.
- **No credential exposure**: Kaggle credentials are only accessed by the existing
  `src/data/loader.py` path (`give_me_credit`). All bundled datasets use local CSV
  files included in the repo.
- **`FileNotFoundError` before read**: `_from_csv()` checks `path.exists()` and
  raises `FileNotFoundError` with a clear message before attempting `pd.read_csv()`.
- **All semgrep scans passed clean** for new files throughout the sprint.

---

## Known Limitations

1. **`model_name` in results is the wrapper's display name, not the registry key.**
   `RandomForestWrapper.name` is `"RandomForest-Default"`, not `"random_forest"`.
   This means you can't directly join results back to registry keys without a mapping.

2. **`_get_peak_memory()` still exists in `src/models/base.py`** alongside the new
   `src/evaluation/memory.get_peak_memory()`. The original private function was not
   removed to avoid touching the pre-existing `evaluate()` hot path. A future sprint
   should consolidate.

3. **`auto` mode silently skips uninstallable models** without counting them.
   If every optional model fails to import, `run_benchmark(models="auto")` returns
   only the two sklearn baselines with no warning that 15 models were skipped.

4. **CLI `run` subcommand does not support DataFrame input** — only named datasets
   and CSV paths. This is consistent with what makes sense from a shell, but
   means the CLI cannot replicate the full Python API surface.

5. **`BasicPreprocessor` is a no-op pass-through.** It follows the sklearn interface
   but performs no actual preprocessing (no imputation, no scaling, no encoding).
   This is intentional for v1 (stub only), but users who call it expecting preprocessing
   will get back unchanged data.

6. **Examples run `matplotlib.use("Agg")` unconditionally.** If a user runs
   `01_quick_start.py` interactively expecting a popup chart they'll see nothing —
   the PNG is saved to `results/` instead.

7. **No `PyPI` publishing.** The package is `pip install -e .` only. The `pyproject.toml`
   metadata (author, URL, description) still contains placeholder values from the
   initial commit.

---

## What's Next (v2 suggestions)

| Priority | Item |
|----------|------|
| P0 | **Registry key in results** — add a `model_key` column to `BenchmarkResult` so the registry key appears alongside the display name in every DataFrame |
| P0 | **`BasicPreprocessor` real implementation** — median imputation, standard scaling for numeric, one-hot for categorics; toggle via `preprocessing=True` on `Benchmarker` |
| P1 | **Cross-dataset report** — `run_benchmark` over a list of dataset names, aggregate results into a single leaderboard table |
| P1 | **PyPI publishing** — fill in `pyproject.toml` metadata, add `CHANGELOG.md`, publish to TestPyPI |
| P1 | **Consolidate `_get_peak_memory`** — remove duplicate from `base.py`, import from `src.evaluation.memory` |
| P1 | **`verbose=False` warning when all models skipped in `auto` mode** — warn if the resolved wrapper list is empty |
| P2 | **Fine-tuning experiments** — `TabPFN v2` fine-tuning pipeline; `Benchmarker(phase="finetune")` |
| P2 | **Scaling experiment CLI** — `tfm-benchmark run --scale 500 1000 5000` to reproduce the row-scaling curves from the PRD |
| P2 | **CI configuration** — GitHub Actions workflow that runs `pytest tests/` and `semgrep` on every PR |
