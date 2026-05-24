# Examples

All examples require only the base install (no GPU, no Kaggle account):

```bash
pip install -e "."
```

## 01_quick_start.py

Benchmarks RandomForest and LogisticRegression on the bundled German Credit dataset, prints a leaderboard table, saves a PNG chart to `results/quick_start_leaderboard.png`, and saves a CSV to `results/quick_start_results.csv`.

```bash
python examples/01_quick_start.py
```

## 02_custom_data.py

Generates a synthetic CSV, loads it via `load_dataset()`, and runs `run_benchmark()`.  Pass `--csv` and `--target` to use your own data instead:

```bash
# synthetic data (no arguments needed)
python examples/02_custom_data.py

# your own CSV
python examples/02_custom_data.py --csv path/to/data.csv --target label_column

# choose specific models
python examples/02_custom_data.py --models random_forest logistic_regression
```

## 03_multi_dataset.py

Demonstrates `run_benchmark_suite()` — a single function call that benchmarks across
multiple datasets and returns one aggregated DataFrame.  The example:

- Benchmarks `random_forest` and `logistic_regression` on **synthetic** and **german_credit**
- Prints a grouped leaderboard (one section per dataset)
- Prints an overall average AUC-ROC row across all datasets
- Saves results to `results/multi_dataset_results.csv`

```bash
python examples/03_multi_dataset.py
```

You can also call `run_benchmark_suite()` programmatically with custom pre-split data:

```python
from tfm_benchmark import run_benchmark_suite

results = run_benchmark_suite(
    datasets=[
        "synthetic",                              # named dataset key
        ("my_data", X_train, y_train, X_test, y_test),  # pre-split tuple
    ],
    models=["random_forest", "logistic_regression"],
    preprocessing=True,   # apply BasicPreprocessor before each dataset
)
print(results.groupby("dataset_name")[["model_key", "auc_roc"]])
```
