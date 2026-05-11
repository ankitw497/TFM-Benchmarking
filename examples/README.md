# Examples

Both examples require only the base install (no GPU, no Kaggle account):

```bash
pip install -e ".[gbdt]"
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
