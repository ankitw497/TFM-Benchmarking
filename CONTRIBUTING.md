# Contributing to TFM-Bench

Thanks for your interest! This project aims to be the go-to resource for evaluating Tabular Foundation Models.

## How to Add a New Model

1. **Create a wrapper** in `src/models/` that inherits from `BaseModelWrapper`
2. **Implement** three methods: `fit()`, `predict_proba()`, `get_limitations()`
3. **Register** it in `scripts/run_benchmark.py` → `get_zero_shot_models()`
4. **Document** the license, row/feature limits, and any gotchas
5. **Test** with `python scripts/run_benchmark.py --dataset german_credit`

### Wrapper Template

```python
from src.models.base import BaseModelWrapper, ModelLimitations

class MyNewModel(BaseModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(name="MyModel-v1", **kwargs)

    def fit(self, X_train, y_train):
        # Initialize and fit your model
        self._is_fitted = True

    def predict_proba(self, X_test):
        # Return shape (n_samples, n_classes)
        pass

    def get_limitations(self):
        return ModelLimitations(
            max_rows=10_000,
            max_features=500,
            supports_missing=True,
            license="MIT",
            commercial_use=True,
            notes="Brief description of the model.",
        )
```

## How to Add a New Dataset

1. Add a loader function in `src/data/loader.py`
2. Add metadata to `get_dataset_info()`
3. Register the dataset name in `configs/experiments.yaml`

## Code Style

- Format with `black --line-length 100`
- Lint with `ruff`
- Type hints encouraged but not required

## Reporting Results

If you run the benchmark on new hardware or with new models, we'd love to include your results! Open a PR with your `results/` CSV files and a brief description of your setup (GPU model, RAM, Python version).
