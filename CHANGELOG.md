# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — 2026-05-24

### Added
- `BenchmarkResult.model_key` field — stable registry key (e.g. `"random_forest"`)
  alongside the human-readable `model_name`, making programmatic post-processing easy.
- `BasicPreprocessor` — real implementation with median imputation (numeric),
  mode imputation (categorical), `StandardScaler`, and `OneHotEncoder(handle_unknown="ignore")`.
  Fit once on training data; no data leakage to test split.
- `Benchmarker(preprocessing=True)` — applies `BasicPreprocessor` once per
  `fit_evaluate()` call before passing data to any model.
  Results DataFrame includes a boolean `preprocessing` column.
- `run_benchmark(..., preprocessing=True)` — forwards new flag to `Benchmarker`.
- Auto-mode skip warnings — `models="auto"` now emits `warnings.warn(UserWarning)` listing
  all skipped model keys and their missing libraries; raises `RuntimeError` if zero models
  can be instantiated.
- `pyproject.toml` — added `authors`, `keywords`, `classifiers`, and `[project.urls]`.
- `CHANGELOG.md` — this file.

### Changed
- `Benchmarker._resolve_wrappers()` now returns `List[(registry_key, wrapper)]` tuples
  (previously `List[wrapper]`); all callers updated.
- `_get_peak_memory()` consolidated into `src.evaluation.memory.get_peak_memory()`;
  the private duplicate in `src/models/base.py` has been removed.
- Version bumped `0.1.0 → 0.2.0`.

### Fixed
- `auto` mode previously swallowed `ImportError` silently — now warns with a clear message.

---

## [0.1.0] — 2026-05-23

### Added
- Public package `tfm_benchmark` with installable `pyproject.toml`.
- `load_dataset(source, ...)` — loads German Credit (UCI), Taiwan Credit, and synthetic
  datasets; supports raw DataFrames and CSV paths; stratified train/test split with
  `reset_index(drop=True)`.
- `list_datasets()` — returns supported named dataset keys.
- `Benchmarker` class — fluent API: `fit_evaluate()`, `plot_leaderboard()`, `save_results()`.
- `run_benchmark(X_train, y_train, X_test, y_test, models=...)` — functional entry point.
- `list_models()` — returns all `MODEL_REGISTRY` keys.
- `MODEL_REGISTRY` with 17 model wrappers:
  - sklearn baselines: `logistic_regression`, `random_forest`, `gradient_boosting`,
    `decision_tree`, `knn`, `naive_bayes`, `svm`
  - GBDT: `xgboost`, `catboost`, `lightgbm`
  - TFMs: `tabpfn_v1`, `tabpfn_v2`, `tabpfn_v2_5_real`, `tabicl_v2`, `mitra`,
    `tabnet`, `ft_transformer`
- `tfm-benchmark` CLI with `list-models`, `list-datasets`, `run` sub-commands.
- Example scripts: `examples/01_quick_start.py`, `examples/02_custom_data.py`.
- Horizontal bar-chart leaderboard via `plot_leaderboard()`.
- `BenchmarkResult` dataclass with AUC-ROC, accuracy, log-loss, Brier score,
  F1-macro, ECE, fit/predict timing, and peak memory.
- 196 tests covering package structure, datasets, benchmarker, API, CLI, and stubs.
